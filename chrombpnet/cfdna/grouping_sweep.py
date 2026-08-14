"""Sweep the similarity-grouping threshold; at each granularity test whether the
simplex solver RECOVERS known compositions (noiseless spike-in). Find the coarsest
granularity where R is well-conditioned enough that 100% epithelial -> ~100% epithelial.
Solver: l2=0, no unknown column (pure inverse problem)."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of

CORPUS="/mnt/data/jganbat/scatac_corpus"
z=np.load(f"{CORPUS}/run/griffin_A_adult346.npz",allow_pickle=True)
accA=z["accA"]; cts=list(z["cts"])
col=accA/(accA.sum(0,keepdims=True)+1e-12)
Cmat=np.clip(np.corrcoef(np.log1p(col*1e4).T),-1,1); D=1-Cmat; np.fill_diagonal(D,0)
Zc=linkage(squareform(D,checks=False),method="average")
def lin_heme(m): return lineage_of(m)=="hematopoietic" or any(s in m for s in ("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")) or "K562" in m

def solve(Rn,y,l2=0.0):
    return dc.deconvolve(y/(y.sum()+1e-12),Rn,l2=l2,iters=8000)

def build(merge_corr,mpg=400):
    cl=fcluster(Zc,t=1-merge_corr,criterion="distance")
    groups={};
    for i,g in enumerate(cl): groups.setdefault(g,[]).append(i)
    gids=sorted(groups); G=len(gids); gmem=[[cts[i] for i in groups[g]] for g in gids]
    Agrp=np.stack([accA[:,groups[g]].mean(1) for g in gids],1)
    cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
    Zm=(cg-rm)/rs; mk={k:np.argsort(Zm[:,k])[::-1][:mpg] for k in range(G)}
    R=np.stack([Agrp[mk[k]].mean(0) for k in range(G)],0); Rn=R/(R.sum(0,keepdims=True)+1e-12)
    lin=[lineage_of(gmem[k][0]) for k in range(G)]
    heme=np.array([any(lin_heme(x) for x in gmem[k]) for k in range(G)])
    epi=np.array([lineage_of(gmem[k][0])=="epithelial" for k in range(G)])
    return Rn,G,gmem,heme,epi,lin

print(f"{'mcorr':>6}{'G':>5}{'cond':>9}   spike recovery (true->rec): 100%epi | 50/50 epi | 100%blood")
rng=np.random.default_rng(0)
for mc in [0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5]:
    Rn,G,gmem,heme,epi,lin=build(mc)
    if epi.sum()==0: continue
    cond=np.linalg.cond(Rn)
    # spike 1: 100% epithelial (spread over epithelial groups)
    w=np.zeros(G); w[epi]=1.0/epi.sum(); yh=solve(Rn,Rn@w); e1=yh[epi].sum()
    # spike 2: 50% epi + 50% blood
    bk=np.where(heme)[0]; w=np.zeros(G); w[epi]=0.5/epi.sum(); w[bk]=0.5/len(bk); yh=solve(Rn,Rn@w); e2=yh[epi].sum(); h2=yh[heme].sum()
    # spike 3: 100% blood
    w=np.zeros(G); w[bk]=1.0/len(bk); yh=solve(Rn,Rn@w); h3=yh[heme].sum()
    print(f"{mc:>6}{G:>5}{cond:>9.0f}   epi 1.00->{e1:.2f} | epi .50->{e2:.2f} heme .50->{h2:.2f} | blood 1.00->{h3:.2f}")
