"""Griffin-style cfDNA cell-type deconvolution (aggregate composite dip over
cell-type-specific marker sites). y[k] = GC-corrected composite coverage-dip depth
at cell-type-k's differential accessible sites; R[k,j] = accessibility of k's
markers in cell type j (from ATAC). Solve y = R w on the simplex + unknown."""
import numpy as np, pyarrow.parquet as pq, pyfaidx, argparse, json, os, sys
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref, deconvolve as dc
from chrombpnet.cfdna.run_poc import build_R
import re as _re
# blood/immune classifier: BM+PBMC studies are entirely hematopoietic, + leukemia line, + keywords
# (the Lareau/Granja naming is CD4/CD8/NK/preB/Ery/GMP/HSC/... which keyword-only regexes miss)
_HEME_KW = _re.compile(r"lymph|myeloid|macrophage|erythro|\bmono|dendritic|\bNK\b|\bCD4|\bCD8|"
                       r"preB|proB|\bGMP|\bHSC|\bCLP|\bCMP|\bMEP|\bMPP|\bLMPP|thymocyte|megakaryo|"
                       r"\bEry|baso|\bmast|plasma|T_cell|B_cell|T_Lympho|B_Lympho|Lymphoid|Myeloid|"
                       r"microglia|kupffer", _re.I)
def is_heme(n):
    if any(s in n for s in ("Granja2019", "Lareau2019", "Mimitou2021")): return True
    if "K562" in n: return True  # CML/erythroleukemia line — hematopoietic origin
    return bool(_HEME_KW.search(n))

def load_frags_gc(cfdna, hg38, chroms):
    t=pq.read_table(cfdna,columns=["chromosome","start_position","end_position","sequence"])
    ch=t.column("chromosome").to_numpy(zero_copy_only=False)
    s=t.column("start_position").to_numpy().astype(np.int64); e=t.column("end_position").to_numpy().astype(np.int64)
    seq=t.column("sequence").combine_chunks(); b=seq.buffers()
    offs=np.frombuffer(b[1],dtype=np.int32); vals=np.frombuffer(b[2],dtype=np.uint8)
    n=len(ch); st_,en_=offs[:n],offs[1:n+1]
    isgc=((vals==71)|(vals==67)|(vals==103)|(vals==99)).astype(np.int64); cz=np.concatenate([[0],np.cumsum(isgc)])
    gcf=np.where((en_-st_)>0,(cz[en_]-cz[st_])/np.maximum(en_-st_,1),0.5)
    L=e-s;keep=(L>=100)&(L<=250);ch,s,e,gcf=ch[keep],s[keep],e[keep],gcf[keep];Lk=L[keep]
    # genome-expected GC
    fa=pyfaidx.Fasta(hg38);rng=np.random.default_rng(0);N=150000;sz={c:len(fa[c]) for c in chroms}
    Ls=Lk[rng.integers(0,len(Lk),N)];egc=[]
    for i in range(N):
        c=chroms[rng.integers(0,len(chroms))];ln=int(Ls[i]);p=int(rng.integers(3_000_000,sz[c]-3_000_000))
        ss=str(fa[c][p:p+ln]).upper()
        if 'N' not in ss: egc.append((ss.count('G')+ss.count('C'))/ln)
    egc=np.array(egc);bins=np.linspace(0,1,51)
    oh,_=np.histogram(gcf,bins=bins,density=True);eh,_=np.histogram(egc,bins=bins,density=True)
    wc=np.where(oh>1e-6,eh/oh,0.0);fw=wc[np.clip(np.digitize(gcf,bins)-1,0,len(wc)-1)];fw/=(fw.mean()+1e-9)
    frby={}
    for c in set(chroms):
        m=ch==c
        if m.any(): o=np.argsort(s[m]);frby[c]=(s[m][o],e[m][o],fw[m][o])
    return frby,len(s)

def region_profiles(frby, regions, W, BIN):
    """GC-weighted coverage profile per region (len 2W/BIN)."""
    nb=2*W//BIN; prof=np.zeros((len(regions),nb))
    rchr=regions["chr"].to_numpy(); rc=regions["center"].to_numpy().astype(np.int64)
    for c in np.unique(rchr):
        if c not in frby: continue
        st,en,fw=frby[c]; idx=np.where(rchr==c)[0]
        for i in idx:
            pos=rc[i]; lo,hi=pos-W,pos+W
            j0=np.searchsorted(st,lo-260); j1=np.searchsorted(st,hi)
            cs=st[j0:j1]; ce=en[j0:j1]; cw=fw[j0:j1]; sel=ce>lo
            cs=np.maximum(cs[sel],lo)-lo; ce=np.minimum(ce[sel],hi)-lo; cw=cw[sel]
            if len(cs)==0: continue
            d=np.zeros(nb+1); a=np.clip((cs//BIN).astype(int),0,nb); bb=np.clip((ce//BIN).astype(int),0,nb)
            np.add.at(d,a,cw); np.add.at(d,bb,-cw); prof[i]=np.cumsum(d[:nb])
    return prof

def dip_of(composite):
    # bounded, stable dip: (flank - center)/(flank + center). Positive = NDR dip.
    nb=len(composite); c=composite[nb//2-5:nb//2+5].mean()
    fl=np.r_[composite[nb//2-70:nb//2-40],composite[nb//2+40:nb//2+70]].mean()
    return (fl-c)/(fl+c+1e-6)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--manifest",required=True); ap.add_argument("--manifest-root",default=None)
    ap.add_argument("--cfdna",required=True); ap.add_argument("-g","--genome",required=True)
    ap.add_argument("--chroms",nargs="*",default=["chr1","chr3","chr6"])
    ap.add_argument("--n-regions",type=int,default=8000); ap.add_argument("--top-per-cell",type=int,default=400)
    ap.add_argument("--markers-per-cell",type=int,default=250); ap.add_argument("--half-width",type=int,default=500)
    ap.add_argument("--W",type=int,default=1000); ap.add_argument("--bin",type=int,default=10)
    ap.add_argument("--jobs",type=int,default=12); ap.add_argument("-o","--out-json",default=None)
    a=ap.parse_args()
    root=a.manifest_root or os.path.dirname(os.path.abspath(a.manifest))
    man=parse_manifest(a.manifest,root); cts=[m['cell_type'] for m in man]
    print(f"cells={len(cts)} chroms={a.chroms}",flush=True)
    print("building panel + accessibility matrix A ...",flush=True)
    regions=ref.build_region_panel(man,a.chroms,top_per_cell=a.top_per_cell,n_regions=a.n_regions,half_width=a.half_width,seed=0)
    accA,_=build_R(man,regions,a.jobs)   # A[region, cell] accessibility sums
    print(f"  panel={len(regions)} A={accA.shape}",flush=True)
    # per-cell-type differential markers: top specificity z-score
    col=accA/(accA.sum(0,keepdims=True)+1e-12)
    rm=col.mean(1,keepdims=True); rs=col.std(1,keepdims=True)+1e-12; Z=(col-rm)/rs  # region x cell z
    markers={}
    for j in range(len(cts)):
        markers[j]=np.argsort(Z[:,j])[::-1][:a.markers_per_cell]
    # reference R[k,j] = mean accessibility of k's markers in cell j
    R=np.zeros((len(cts),len(cts)))
    for k in range(len(cts)):
        R[k]=accA[markers[k]].mean(0)
    # cfDNA composite dip per marker set
    print("loading cfDNA + GC ...",flush=True)
    frby,nfr=load_frags_gc(a.cfdna,a.genome,a.chroms); print(f"  frags={nfr:,}",flush=True)
    print("cfDNA region coverage profiles ...",flush=True)
    prof=region_profiles(frby,regions,a.W,a.bin)
    y=np.array([dip_of(prof[markers[k]].sum(0)) for k in range(len(cts))])
    print(f"  y (dip depths): min {y.min():.3f} med {np.median(y):.3f} max {y.max():.3f}",flush=True)
    # KEY DIAGNOSTIC: is openness (y) higher for hematopoietic marker sets? (healthy plasma)
    hm=np.array([is_heme(c) for c in cts])
    yh=y[hm]; yn_=y[~hm]
    print(f"  y[heme] mean={yh.mean():.4f} (n={hm.sum()}) | y[non-heme] mean={yn_.mean():.4f} "
          f"| delta={yh.mean()-yn_.mean():+.4f}",flush=True)
    order_y=np.argsort(y)[::-1]
    print("  top-12 marker sets by openness (y):")
    for k in order_y[:12]:
        print(f"      y={y[k]:+.3f} {'HEME' if hm[k] else '    '}  {cts[k]}",flush=True)
    # normalize columns of R and y to distributions, deconvolve on simplex + unknown
    Rn=R/(R.sum(0,keepdims=True)+1e-12)
    yn=np.clip(y,0,None); yn=yn/(yn.sum()+1e-12)
    u=np.full((Rn.shape[0],1),1.0/Rn.shape[0]); Rw=np.hstack([Rn,u]); names=cts+["__unknown__"]
    w=dc.deconvolve(yn,Rw,l2=1e-4); recon=float(np.corrcoef(yn,Rw@w)[0,1])
    order=np.argsort(w)[::-1]
    heme=sum(w[i] for i,n in enumerate(names) if n!="__unknown__" and is_heme(n))
    unk=w[names.index("__unknown__")]
    print(f"\n=== GRIFFIN DECONVOLUTION: {os.path.basename(a.cfdna)} ===")
    print(f"recon r(y,Rw)={recon:.3f} | hematopoietic mass={heme:.3f} | unknown={unk:.3f}")
    print(f"{'rank':>4} {'weight':>8} {'heme':>5}  cell type")
    for r,k in enumerate(order[:20],1):
        n=names[k]; tag="UNK" if n=="__unknown__" else ("HEME" if is_heme(n) else "")
        print(f"{r:>4} {w[k]:>8.4f} {tag:>5}  {n}")
    if a.out_json:
        json.dump({"cfdna":os.path.basename(a.cfdna),"recon_r":recon,"heme_mass":float(heme),
                   "unknown_mass":float(unk),"top":[(names[k],float(w[k])) for k in order[:40]]},
                  open(a.out_json,"w"),indent=2)
        print("wrote",a.out_json)

if __name__=="__main__": main()
