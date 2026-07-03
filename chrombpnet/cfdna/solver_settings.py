"""Find robust solver settings: pass the spike-in AND survive noise + missing cell types.
Tests unknown-column handling (none / flat / mean-of-R) x l2 x finite-depth noise,
and a 'held-out cell type' case (spike a group then remove its column from R)."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of

CORPUS="/mnt/data/jganbat/scatac_corpus"; MC=0.7
z=np.load(f"{CORPUS}/run/griffin_A_adult346.npz",allow_pickle=True); accA=z["accA"]; cts=list(z["cts"])
col=accA/(accA.sum(0,keepdims=True)+1e-12)
D=1-np.clip(np.corrcoef(np.log1p(col*1e4).T),-1,1); np.fill_diagonal(D,0)
cl=fcluster(linkage(squareform(D,checks=False),method="average"),t=1-MC,criterion="distance")
groups={};
for i,g in enumerate(cl): groups.setdefault(g,[]).append(i)
gids=sorted(groups); G=len(gids); gmem=[[cts[i] for i in groups[g]] for g in gids]
Agrp=np.stack([accA[:,groups[g]].mean(1) for g in gids],1)
cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
Zm=(cg-rm)/rs; mk={k:np.argsort(Zm[:,k])[::-1][:400] for k in range(G)}
R=np.stack([Agrp[mk[k]].mean(0) for k in range(G)],0); Rn=R/(R.sum(0,keepdims=True)+1e-12)
lheme=lambda m: lineage_of(m)=="hematopoietic" or any(s in m for s in("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")) or "K562" in m
heme=np.array([any(lheme(x) for x in gmem[k]) for k in range(G)]); epi=np.array([lineage_of(gmem[k][0])=="epithelial" for k in range(G)])
print(f"MC={MC} G={G} cond={np.linalg.cond(Rn):.0f} heme_g={heme.sum()} epi_g={epi.sum()}")
rng=np.random.default_rng(0)

def solve_R(Rmat, y, unk="none", l2=0.0):
    g=Rmat.shape[1]
    if unk=="none": M=Rmat
    elif unk=="flat": M=np.hstack([Rmat,np.full((g,1),1/g)])
    elif unk=="mean": M=np.hstack([Rmat,Rmat.mean(1,keepdims=True)])
    w=dc.deconvolve(y/(y.sum()+1e-12),M,l2=l2,iters=8000)
    return w[:g]

def spike(desc, wstar, unk, l2, depth=None, dropcol=None):
    y=Rn@wstar
    if depth: p=np.clip(y,0,None); p/=p.sum(); y=rng.multinomial(depth,p)/depth
    if dropcol is not None:
        keep=np.ones(G,bool); keep[dropcol]=False
        Rd=Rn[:,keep]/(Rn[:,keep].sum(0,keepdims=True)+1e-12)
        wk=solve_R(Rd,y,unk,l2); w=np.zeros(G); w[keep]=wk
    else:
        w=solve_R(Rn,y,unk,l2)
    print(f"  {desc:<38} unk={unk:<4} l2={l2:<5} depth={str(depth):>6} | epi {wstar[epi].sum():.2f}->{w[epi].sum():.2f}  heme {wstar[heme].sum():.2f}->{w[heme].sum():.2f}")

bk=np.where(heme)[0]; epk=np.where(epi)[0]
w5050=np.zeros(G); w5050[epk]=0.5/len(epk); w5050[bk]=0.5/len(bk)
w1090=np.zeros(G); w1090[epk]=0.1/len(epk); w1090[bk]=0.9/len(bk)
print("\n=== unknown-column + l2 effect (noiseless 50/50) ===")
for unk in ["none","flat","mean"]:
    for l2 in [0.0,1e-3]:
        spike("50% epi + 50% blood", w5050, unk, l2)
print("\n=== finite depth (unk=mean, l2=0) ===")
for d in [None,500000,100000,20000]:
    spike("10% epi + 90% blood", w1090, "mean", 0.0, depth=d)
print("\n=== missing cell type: spike a group, drop its column (unk=mean) ===")
# simulate neutrophil-like gap: 30% from a dropped blood group + 70% other blood
dg=int(bk[0]); w=np.zeros(G); w[dg]=0.3; w[bk[1:6]]=0.7/5
spike(f"30% dropped-group + 70% blood", w, "mean", 0.0, dropcol=dg)
