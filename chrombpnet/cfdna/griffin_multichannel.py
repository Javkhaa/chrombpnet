"""Multi-channel cfDNA deconvolution: accessibility (coverage-dip at NDR markers) +
nucleosome-dyad (fragment-midpoint peak at nucleosome-occupied markers). Tests whether
adding the nucleosome channel improves spike-in recovery + resolution over acc-alone.
Uses the cached acc+nuc reference matrices; caches cfDNA coverage + midpoint profiles."""
import numpy as np, pyarrow.parquet as pq, pyfaidx, argparse, os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of, load_frags_gc

def lheme(m): return lineage_of(m)=="hematopoietic" or any(s in m for s in("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")) or "K562" in m

def cov_profiles(frby,regions,W,BIN):
    rchr,rc=regions; nb=2*W//BIN; prof=np.zeros((len(rc),nb))
    for c in np.unique(rchr):
        if c not in frby: continue
        st,en,fw=frby[c]; idx=np.where(rchr==c)[0]
        for i in idx:
            pos=rc[i]; lo,hi=pos-W,pos+W; j0=np.searchsorted(st,lo-260); j1=np.searchsorted(st,hi)
            cs=st[j0:j1]; ce=en[j0:j1]; cw=fw[j0:j1]; sel=ce>lo
            cs=np.maximum(cs[sel],lo)-lo; ce=np.minimum(ce[sel],hi)-lo; cw=cw[sel]
            if len(cs)==0: continue
            d=np.zeros(nb+1); a=np.clip((cs//BIN).astype(int),0,nb); b=np.clip((ce//BIN).astype(int),0,nb)
            np.add.at(d,a,cw); np.add.at(d,b,-cw); prof[i]=np.cumsum(d[:nb])
    return prof

def mid_profiles(frby,regions,W,BIN):
    rchr,rc=regions; nb=2*W//BIN; prof=np.zeros((len(rc),nb))
    for c in np.unique(rchr):
        if c not in frby: continue
        st,en,fw=frby[c]; mid=((st+en)//2); o=np.argsort(mid); mids=mid[o]; fws=fw[o]; idx=np.where(rchr==c)[0]
        for i in idx:
            pos=rc[i]; lo,hi=pos-W,pos+W; j0=np.searchsorted(mids,lo); j1=np.searchsorted(mids,hi)
            rel=((mids[j0:j1]-lo)//BIN).astype(int); w=fws[j0:j1]; m=(rel>=0)&(rel<nb)
            np.add.at(prof[i],rel[m],w[m])
    return prof

def dip(comp):  # NDR openness
    nb=len(comp); c=comp[nb//2-5:nb//2+5].mean(); fl=np.r_[comp[nb//2-70:nb//2-40],comp[nb//2+40:nb//2+70]].mean(); return (fl-c)/(fl+c+1e-6)
def peak(comp): # nucleosome dyad occupancy
    nb=len(comp); c=comp[nb//2-5:nb//2+5].mean(); fl=np.r_[comp[nb//2-70:nb//2-40],comp[nb//2+40:nb//2+70]].mean(); return (c-fl)/(c+fl+1e-6)

def markers(M,G_cols_idx,groups,mpg):
    Agrp=np.stack([M[:,groups[g]].mean(1) for g in G_cols_idx],1)
    cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
    Zm=(cg-rm)/rs; return Agrp,{k:np.argsort(Zm[:,k])[::-1][:mpg] for k in range(len(G_cols_idx))}

def refmat(Agrp,mk):
    G=Agrp.shape[1]; R=np.stack([Agrp[mk[k]].mean(0) for k in range(G)],0); return R/(R.sum(0,keepdims=True)+1e-12)

def solve(Rn,y):
    u=Rn.mean(1,keepdims=True); Rw=np.hstack([Rn,u]); yn=np.clip(y,0,None); yn=yn/(yn.sum()+1e-12)
    return dc.deconvolve(yn,Rw,l2=0.0,iters=8000)[:Rn.shape[1]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--an-cache",required=True); ap.add_argument("--cfdna",required=True)
    ap.add_argument("-g","--genome",required=True); ap.add_argument("--chroms",nargs="*",default=["chr1","chr3","chr6"])
    ap.add_argument("--merge-corr",type=float,default=0.85); ap.add_argument("--mpg",type=int,default=400)
    ap.add_argument("--W",type=int,default=1000); ap.add_argument("--bin",type=int,default=10)
    ap.add_argument("--prof-cache",default=None)
    a=ap.parse_args()
    z=np.load(a.an_cache,allow_pickle=True); accA=z["accA"]; nucA=z["nucA"]; cts=list(z["cts"])
    regions=(z["rchr"],z["rc"].astype(np.int64))
    col=accA/(accA.sum(0,keepdims=True)+1e-12)
    D=1-np.clip(np.corrcoef(np.log1p(col*1e4).T),-1,1); np.fill_diagonal(D,0)
    cl=fcluster(linkage(squareform(D,checks=False),method="average"),t=1-a.merge_corr,criterion="distance")
    groups={};
    for i,g in enumerate(cl): groups.setdefault(g,[]).append(i)
    gid=sorted(groups); G=len(gid); gmem=[[cts[i] for i in groups[g]] for g in gid]
    heme=np.array([any(lheme(x) for x in gmem[k]) for k in range(G)])
    epi=np.array([lineage_of(gmem[k][0])=="epithelial" for k in range(G)])
    AaccG,mk_acc=markers(accA,gid,groups,a.mpg); AnucG,mk_nuc=markers(nucA,gid,groups,a.mpg)
    Racc=refmat(AaccG,mk_acc); Rnuc=refmat(AnucG,mk_nuc)
    print(f"G={G} cond(acc)={np.linalg.cond(Racc):.0f} cond(nuc)={np.linalg.cond(Rnuc):.0f} heme_g={heme.sum()} epi_g={epi.sum()}",flush=True)
    # cfDNA features
    if a.prof_cache and os.path.exists(a.prof_cache):
        zz=np.load(a.prof_cache); covp=zz["cov"]; midp=zz["mid"]; print("loaded prof cache",flush=True)
    else:
        frby,nfr=load_frags_gc(a.cfdna,a.genome,a.chroms); print(f"frags={nfr:,}",flush=True)
        covp=cov_profiles(frby,regions,a.W,a.bin); midp=mid_profiles(frby,regions,a.W,a.bin)
        if a.prof_cache: np.savez(a.prof_cache,cov=covp,mid=midp)
    y_acc=np.array([dip(covp[mk_acc[k]].sum(0)) for k in range(G)])
    y_nuc=np.array([peak(midp[mk_nuc[k]].sum(0)) for k in range(G)])
    Rstack=np.vstack([Racc,Rnuc]); Rstack=Rstack/(Rstack.sum(0,keepdims=True)+1e-12)
    y_stack=np.concatenate([np.clip(y_acc,0,None)/ (np.clip(y_acc,0,None).sum()+1e-9),
                            np.clip(y_nuc,0,None)/ (np.clip(y_nuc,0,None).sum()+1e-9)])
    rng=np.random.default_rng(0)
    def spike_recover(Rn):
        # 20% epithelial + 80% blood, noiseless
        bk=np.where(heme)[0]; ek=np.where(epi)[0]; w=np.zeros(G); w[ek]=0.2/len(ek); w[bk]=0.8/len(bk)
        wh=solve(Rn,Rn@w); return wh[epi].sum()
    print("\n=== channel comparison (real sample) ===")
    for name,Rn,y in [("acc",Racc,y_acc),("nuc",Rnuc,y_nuc),("acc+nuc",Rstack,y_stack)]:
        w=solve(Rn,y); h=w[heme].sum(); e=w[epi].sum()
        sp=spike_recover(Rn)
        print(f"  {name:<8} heme={h:.3f} epi={e:.3f} | spike 20%epi->{sp:.2f}",flush=True)

if __name__=="__main__": main()
