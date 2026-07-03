"""Data-driven similarity-grouped cfDNA deconvolution. Agglomeratively merge
cell types whose accessibility signatures correlate above --merge-corr, giving
adaptive granularity (fine where distinguishable, collapsed where not). Caches
the accessibility matrix A so the granularity knob can be swept in seconds.

  # first run builds + caches A, then deconvolves at merge-corr 0.9
  griffin_grouped.py -m manifest.tsv --manifest-root R --cfdna x.parquet -g hg38.fa \
     --cache /path/A.npz --merge-corr 0.90
  # subsequent runs reuse the cache: instant re-solve at a different threshold
"""
import numpy as np, pyarrow.parquet as pq, pyfaidx, argparse, json, os, re
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref, deconvolve as dc
from chrombpnet.cfdna.run_poc import build_R
from chrombpnet.cfdna.griffin_lineage import lineage_of, load_frags_gc, region_profiles, dip_of  # reuse

def is_heme(n):
    if any(s in n for s in ("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")): return True
    if "K562" in n: return True
    return lineage_of(n)=="hematopoietic"

def group_label(members):
    # name a merged group by its lineage(s) + a representative
    lins=sorted(set(lineage_of(m) for m in members))
    rep=min(members,key=len)  # shortest name as representative
    tag="+".join(lins) if len(lins)<=2 else "mixed"
    return f"[{tag}:{len(members)}] {rep.split('__')[-1] if '__' in rep else rep}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--manifest",required=True); ap.add_argument("--manifest-root",default=None)
    ap.add_argument("--cfdna",required=True); ap.add_argument("-g","--genome",required=True)
    ap.add_argument("--chroms",nargs="*",default=["chr1","chr3","chr6"])
    ap.add_argument("--n-regions",type=int,default=8000); ap.add_argument("--top-per-cell",type=int,default=400)
    ap.add_argument("--markers-per-group",type=int,default=400); ap.add_argument("--half-width",type=int,default=500)
    ap.add_argument("--W",type=int,default=1000); ap.add_argument("--bin",type=int,default=10)
    ap.add_argument("--merge-corr",type=float,default=0.70,help="merge cell types with signature corr above this (0.7 keeps R well-conditioned)")
    ap.add_argument("--jobs",type=int,default=12); ap.add_argument("--cache",default=None)
    ap.add_argument("--prof-cache",default=None,help="cache/load cfDNA region coverage profiles (fast granularity sweeps)")
    ap.add_argument("-o","--out-json",default=None)
    a=ap.parse_args()
    root=a.manifest_root or os.path.dirname(os.path.abspath(a.manifest))
    man=parse_manifest(a.manifest,root); cts=[m['cell_type'] for m in man]
    # --- accessibility matrix A (cached) ---
    if a.cache and os.path.exists(a.cache):
        z=np.load(a.cache,allow_pickle=True); accA=z["accA"]; regions=__import__("pandas").DataFrame(
            {"chr":z["rchr"],"center":z["rc"],"start":z["rc"]-a.half_width,"end":z["rc"]+a.half_width})
        assert list(z["cts"])==cts, "cache manifest mismatch"
        print(f"loaded cached A {accA.shape}",flush=True)
    else:
        print("building panel + A ...",flush=True)
        regions=ref.build_region_panel(man,a.chroms,top_per_cell=a.top_per_cell,n_regions=a.n_regions,half_width=a.half_width,seed=0)
        accA,_=build_R(man,regions,a.jobs)
        if a.cache:
            np.savez(a.cache,accA=accA,rchr=regions["chr"].to_numpy(),rc=regions["center"].to_numpy(),cts=np.array(cts))
            print(f"cached A -> {a.cache}",flush=True)
    # --- similarity grouping: correlation of per-cell accessibility signatures ---
    col=accA/(accA.sum(0,keepdims=True)+1e-12)
    C=np.corrcoef(np.log1p(col*1e4).T)  # cell x cell signature correlation
    C=np.clip(C,-1,1); D=1.0-C; np.fill_diagonal(D,0.0)
    Zc=linkage(squareform(D,checks=False),method="average")
    clusters=fcluster(Zc,t=1.0-a.merge_corr,criterion="distance")
    groups={}
    for i,g in enumerate(clusters): groups.setdefault(g,[]).append(i)
    gids=sorted(groups); G=len(gids)
    print(f"cells={len(cts)} -> groups={G} at merge-corr={a.merge_corr}",flush=True)
    # group-mean accessibility
    Agrp=np.zeros((len(regions),G)); gmembers=[]
    for gi,g in enumerate(gids):
        idx=groups[g]; Agrp[:,gi]=accA[:,idx].mean(1); gmembers.append([cts[i] for i in idx])
    # differential markers per group
    cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
    Zm=(cg-rm)/rs; markers={gi:np.argsort(Zm[:,gi])[::-1][:a.markers_per_group] for gi in range(G)}
    R=np.zeros((G,G))
    for k in range(G): R[k]=Agrp[markers[k]].mean(0)
    # --- cfDNA (region coverage profiles cached per sample; independent of grouping) ---
    print("cfDNA profiles ...",flush=True)
    if a.prof_cache and os.path.exists(a.prof_cache):
        prof=np.load(a.prof_cache)["prof"]; print(f"  loaded cached profiles {prof.shape}",flush=True)
    else:
        frby,nfr=load_frags_gc(a.cfdna,a.genome,a.chroms); print(f"  frags={nfr:,}",flush=True)
        prof=region_profiles(frby,regions,a.W,a.bin)
        if a.prof_cache: np.savez(a.prof_cache,prof=prof); print(f"  cached profiles -> {a.prof_cache}",flush=True)
    y=np.array([dip_of(prof[markers[k]].sum(0)) for k in range(G)])
    Rn=R/(R.sum(0,keepdims=True)+1e-12); yn=np.clip(y,0,None); yn=yn/(yn.sum()+1e-12)
    # l2=0 (ridge biases toward the collinear blood block; validated via spike-in),
    # unknown column = mean-of-R (absorbs missing cell types without breaking recovery)
    u=Rn.mean(1,keepdims=True); Rw=np.hstack([Rn,u]); w=dc.deconvolve(yn,Rw,l2=0.0,iters=8000)
    recon=float(np.corrcoef(yn,Rw@w)[0,1])
    labels=[group_label(m) for m in gmembers]+["__unknown__"]
    heme_g=[is_heme(gmembers[gi][0]) or any(is_heme(m) for m in gmembers[gi]) for gi in range(G)]
    heme_mass=sum(w[gi] for gi in range(G) if heme_g[gi])
    print(f"\n=== GROUPED DECONVOLUTION (merge-corr {a.merge_corr}): {os.path.basename(a.cfdna)} ===")
    print(f"recon r={recon:.3f} | cond(Rn)={np.linalg.cond(Rn):.1f} | HEMATOPOIETIC mass={heme_mass:.3f}")
    for k in np.argsort(w)[::-1][:18]:
        if w[k]<1e-3: continue
        tag="HEME" if (k<G and heme_g[k]) else ("UNK" if labels[k]=="__unknown__" else "    ")
        print(f"  {w[k]:6.3f} {tag}  {labels[k]}")
    if a.out_json:
        json.dump({"cfdna":os.path.basename(a.cfdna),"merge_corr":a.merge_corr,"n_groups":G,
                   "recon_r":recon,"cond":float(np.linalg.cond(Rn)),"heme_mass":float(heme_mass),
                   "top":[(labels[int(k)],float(w[int(k)])) for k in np.argsort(w)[::-1][:30]]},
                  open(a.out_json,"w"),indent=2); print("wrote",a.out_json)

if __name__=="__main__": main()
