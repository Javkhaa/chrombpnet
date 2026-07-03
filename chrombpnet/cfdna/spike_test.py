"""Specificity spike-in test: synthesize y = R w* for KNOWN compositions and check
the simplex solver recovers them. If it collapses everything to blood, the solver is
biased; if it recovers spiked non-blood, the healthy/CRC results are real (and CRC's
100%-blood is a low-tumor-fraction / sensitivity issue, not bias)."""
import numpy as np, os, re
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of

CORPUS="/mnt/data/jganbat/scatac_corpus"
z=np.load(f"{CORPUS}/run/griffin_A_adult346.npz",allow_pickle=True)
accA=z["accA"]; cts=list(z["cts"]); print(f"A={accA.shape}")
# group like griffin_grouped (merge-corr 0.9)
col=accA/(accA.sum(0,keepdims=True)+1e-12)
C=np.clip(np.corrcoef(np.log1p(col*1e4).T),-1,1); D=1-C; np.fill_diagonal(D,0)
cl=fcluster(linkage(squareform(D,checks=False),method="average"),t=0.1,criterion="distance")
groups={};
for i,g in enumerate(cl): groups.setdefault(g,[]).append(i)
gids=sorted(groups); G=len(gids); gmem=[[cts[i] for i in groups[g]] for g in gids]
Agrp=np.stack([accA[:,groups[g]].mean(1) for g in gids],1)
cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
Zm=(cg-rm)/rs; markers={k:np.argsort(Zm[:,k])[::-1][:400] for k in range(G)}
R=np.stack([Agrp[markers[k]].mean(0) for k in range(G)],0)  # R[k,j]
Rn=R/(R.sum(0,keepdims=True)+1e-12)
def is_heme(m):
    return any(s in m for s in ("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")) or "K562" in m or lineage_of(m)=="hematopoietic"
heme_g=np.array([any(is_heme(x) for x in gmem[k]) for k in range(G)])
lin_g=[lineage_of(gmem[k][0]) for k in range(G)]
# representative group indices for the spike
def find(kw):
    for k in range(G):
        if any(kw.lower() in x.lower() for x in gmem[k]): return k
    return None
epi=find("Colon_Epithelial"); blood=[k for k in range(G) if heme_g[k]]
print(f"G={G} groups | heme groups={heme_g.sum()} | epithelial group idx={epi} ({gmem[epi][0] if epi else None})")

def solve(y):
    u=np.full((G,1),1/G); Rw=np.hstack([Rn,u]); w=dc.deconvolve(y/ (y.sum()+1e-12),Rw,l2=1e-3)
    return w[:G]  # drop unknown
rng=np.random.default_rng(0)
def report(name,wstar,depth=None):
    y=Rn@wstar
    if depth:
        p=np.clip(y,0,None); p/=p.sum(); y=rng.multinomial(depth,p)/depth  # finite-depth noise
    wh=solve(y)
    heme=wh[heme_g].sum(); epi_w=wh[epi] if epi is not None else 0
    true_heme=wstar[heme_g].sum(); true_epi=wstar[epi] if epi is not None else 0
    print(f"  {name:<34} depth={str(depth):>7} | heme {true_heme:.2f}->{heme:.2f}  epi {true_epi:.2f}->{epi_w:.2f}  | recon corr={np.corrcoef(wstar,wh)[0,1]:.3f}")

print("\n=== SPIKE-IN RECOVERY (true -> recovered) ===")
# 1. pure epithelial
w=np.zeros(G); w[epi]=1.0; report("100% colonic epithelial",w)
# 2. 50/50 epithelial/blood
w=np.zeros(G); w[epi]=0.5;
bk=rng.choice(blood,5,replace=False); w[bk]=0.5/5; report("50% epithelial + 50% blood",w)
# 3. 10% epithelial under 90% blood (the CRC-like regime)
w=np.zeros(G); w[epi]=0.1; w[bk]=0.9/5; report("10% epithelial + 90% blood",w)
w=np.zeros(G); w[epi]=0.1; w[bk]=0.9/5; report("10% epithelial + 90% blood (noisy)",w,depth=200000)
# 4. pure blood (positive)
w=np.zeros(G); w[bk]=1/5; report("100% blood (5 subsets)",w)
