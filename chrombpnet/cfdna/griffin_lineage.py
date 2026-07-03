"""Lineage-level cfDNA deconvolution: collapse cell types into ~9 lineages to
kill the collinearity that makes fine (200-way) deconvolution unstable at 0.28x.
y[L] = GC-corrected composite dip at lineage-L differential marker sites;
R[L,L'] = accessibility of L's markers in lineage L'. Solve y = R w over lineages."""
import numpy as np, pyarrow.parquet as pq, pyfaidx, argparse, json, os, re
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref, deconvolve as dc
from chrombpnet.cfdna.run_poc import build_R

def lineage_of(n):
    if any(s in n for s in ("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")): return "hematopoietic"
    if "K562" in n: return "hematopoietic"
    if re.search(r"lymph|myeloid|macrophage|erythro|monocyte|dendritic|\bNK\b|\bCD4|\bCD8|B_cell|"
                 r"T_cell|B_Lympho|T_Lympho|Lymphoid|Myeloid|thymocyte|megakaryo|microglia|kupffer|"
                 r"\bmast|plasma_cell|\bHSC|\bGMP|\bCLP|\bCMP|preB|proB|\bEry|regulatory_T|invariant_T",n,re.I): return "hematopoietic"
    if re.search(r"retina|photoreceptor|amacrine|horizontal|bipolar|\brod|\bcone|Mueller|Muller",n,re.I): return "retinal"
    if re.search(r"neuron|neural|oligo|astro|\bOPC|\bglia|excitatory|inhibitory|LAMP5|PVALB|\bVIP|\bSST|pericyte.*brain",n,re.I): return "neural"
    if re.search(r"cardio|cardiac|myocyte|\bCM\b|_CM$|heart|skeletal_muscle|\baCM|\bvCM|Cardiomyocyte",n,re.I): return "cardiac_muscle"
    if re.search(r"epitheli|enterocyte|goblet|colon|esophag|ductal|acinar|hepato|cholangio|\bclub|secretory|\bbasal|keratinocyte|alveolar|ciliated",n,re.I): return "epithelial"
    if re.search(r"fibroblast|pericyte|mesench|stellate|stromal|smooth_muscle|\bSMC|mural|\bFB\d",n,re.I): return "stromal"
    if re.search(r"endothel",n,re.I): return "endothelial"
    if re.search(r"Tumor|tumour|glioblastoma|carcinoma|MCF7|\bTC\b",n,re.I): return "tumor"
    return "other"

def load_frags_gc(cfdna,hg38,chroms):
    t=pq.read_table(cfdna,columns=["chromosome","start_position","end_position","sequence"])
    ch=t.column("chromosome").to_numpy(zero_copy_only=False)
    s=t.column("start_position").to_numpy().astype(np.int64); e=t.column("end_position").to_numpy().astype(np.int64)
    seq=t.column("sequence").combine_chunks(); b=seq.buffers()
    offs=np.frombuffer(b[1],dtype=np.int32); vals=np.frombuffer(b[2],dtype=np.uint8)
    n=len(ch); st_,en_=offs[:n],offs[1:n+1]
    isgc=((vals==71)|(vals==67)|(vals==103)|(vals==99)).astype(np.int64); cz=np.concatenate([[0],np.cumsum(isgc)])
    gcf=np.where((en_-st_)>0,(cz[en_]-cz[st_])/np.maximum(en_-st_,1),0.5)
    L=e-s;keep=(L>=100)&(L<=250);ch,s,e,gcf=ch[keep],s[keep],e[keep],gcf[keep];Lk=L[keep]
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

def region_profiles(frby,regions,W,BIN):
    nb=2*W//BIN; prof=np.zeros((len(regions),nb))
    rchr=regions["chr"].to_numpy(); rc=regions["center"].to_numpy().astype(np.int64)
    for c in np.unique(rchr):
        if c not in frby: continue
        st,en,fw=frby[c]; idx=np.where(rchr==c)[0]
        for i in idx:
            pos=rc[i]; lo,hi=pos-W,pos+W; j0=np.searchsorted(st,lo-260); j1=np.searchsorted(st,hi)
            cs=st[j0:j1]; ce=en[j0:j1]; cw=fw[j0:j1]; sel=ce>lo
            cs=np.maximum(cs[sel],lo)-lo; ce=np.minimum(ce[sel],hi)-lo; cw=cw[sel]
            if len(cs)==0: continue
            d=np.zeros(nb+1); a=np.clip((cs//BIN).astype(int),0,nb); bb=np.clip((ce//BIN).astype(int),0,nb)
            np.add.at(d,a,cw); np.add.at(d,bb,-cw); prof[i]=np.cumsum(d[:nb])
    return prof

def dip_of(comp):
    nb=len(comp); c=comp[nb//2-5:nb//2+5].mean(); fl=np.r_[comp[nb//2-70:nb//2-40],comp[nb//2+40:nb//2+70]].mean()
    return (fl-c)/(fl+c+1e-6)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--manifest",required=True); ap.add_argument("--manifest-root",default=None)
    ap.add_argument("--cfdna",required=True); ap.add_argument("-g","--genome",required=True)
    ap.add_argument("--chroms",nargs="*",default=["chr1","chr3","chr6"])
    ap.add_argument("--n-regions",type=int,default=8000); ap.add_argument("--top-per-cell",type=int,default=400)
    ap.add_argument("--markers-per-lineage",type=int,default=600); ap.add_argument("--half-width",type=int,default=500)
    ap.add_argument("--W",type=int,default=1000); ap.add_argument("--bin",type=int,default=10)
    ap.add_argument("--jobs",type=int,default=12); ap.add_argument("-o","--out-json",default=None)
    a=ap.parse_args()
    root=a.manifest_root or os.path.dirname(os.path.abspath(a.manifest))
    man=parse_manifest(a.manifest,root); cts=[m['cell_type'] for m in man]
    lin=np.array([lineage_of(c) for c in cts]); L=sorted(set(lin))
    print(f"cells={len(cts)} lineages={len(L)}",flush=True)
    for l in L: print(f"  {l}: {(lin==l).sum()} cells",flush=True)
    print("panel + accessibility A ...",flush=True)
    regions=ref.build_region_panel(man,a.chroms,top_per_cell=a.top_per_cell,n_regions=a.n_regions,half_width=a.half_width,seed=0)
    accA,_=build_R(man,regions,a.jobs)
    # lineage-mean accessibility
    Alin=np.zeros((len(regions),len(L)))
    for j,l in enumerate(L): Alin[:,j]=accA[:,lin==l].mean(1)
    # lineage differential markers (specificity z-score)
    col=Alin/(Alin.sum(0,keepdims=True)+1e-12); rm=col.mean(1,keepdims=True); rs=col.std(1,keepdims=True)+1e-12
    Z=(col-rm)/rs; markers={j:np.argsort(Z[:,j])[::-1][:a.markers_per_lineage] for j in range(len(L))}
    R=np.zeros((len(L),len(L)))
    for k in range(len(L)): R[k]=Alin[markers[k]].mean(0)
    print("cfDNA coverage profiles ...",flush=True)
    frby,nfr=load_frags_gc(a.cfdna,a.genome,a.chroms); print(f"  frags={nfr:,}",flush=True)
    prof=region_profiles(frby,regions,a.W,a.bin)
    y=np.array([dip_of(prof[markers[k]].sum(0)) for k in range(len(L))])
    print("\n  lineage openness y (dip depth):")
    for k in np.argsort(y)[::-1]:
        print(f"    y={y[k]:+.3f}  {L[k]}",flush=True)
    Rn=R/(R.sum(0,keepdims=True)+1e-12); yn=np.clip(y,0,None); yn=yn/(yn.sum()+1e-12)
    u=np.full((Rn.shape[0],1),1.0/Rn.shape[0]); Rw=np.hstack([Rn,u]); names=L+["__unknown__"]
    w=dc.deconvolve(yn,Rw,l2=1e-3); recon=float(np.corrcoef(yn,Rw@w)[0,1])
    print(f"\n=== LINEAGE DECONVOLUTION: {os.path.basename(a.cfdna)} ===")
    print(f"recon r={recon:.3f} | cond(R)={np.linalg.cond(Rn):.1f}")
    for k in np.argsort(w)[::-1]:
        print(f"  {w[k]:6.3f}  {names[k]}")
    if a.out_json:
        json.dump({"cfdna":os.path.basename(a.cfdna),"recon_r":recon,
                   "fractions":{names[k]:float(w[k]) for k in range(len(names))},
                   "lineage_y":{L[k]:float(y[k]) for k in range(len(L))}}, open(a.out_json,"w"),indent=2)
        print("wrote",a.out_json)

if __name__=="__main__": main()
