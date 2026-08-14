"""Granularity-stability check on real samples using cached A + cached profiles.
Sweep merge-corr; report hematopoietic mass + total non-blood for healthy vs cancer.
Trustworthy iff the healthy hematopoietic fraction is stable across granularity."""
import numpy as np, os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from chrombpnet.cfdna import deconvolve as dc
from chrombpnet.cfdna.griffin_lineage import lineage_of

CORPUS="/mnt/data/jganbat/scatac_corpus"
z=np.load(f"{CORPUS}/run/griffin_A_adult346.npz",allow_pickle=True); accA=z["accA"]; cts=list(z["cts"])
col=accA/(accA.sum(0,keepdims=True)+1e-12)
D=1-np.clip(np.corrcoef(np.log1p(col*1e4).T),-1,1); np.fill_diagonal(D,0)
Zc=linkage(squareform(D,checks=False),method="average")
W=1000;BIN=10;nb=2*W//BIN
def dip_of(comp):
    c=comp[nb//2-5:nb//2+5].mean(); fl=np.r_[comp[nb//2-70:nb//2-40],comp[nb//2+40:nb//2+70]].mean(); return (fl-c)/(fl+c+1e-6)
lheme=lambda m: lineage_of(m)=="hematopoietic" or any(s in m for s in("Granja2019","Lareau2019","Mimitou2021","Satpathy2019")) or "K562" in m
profs={S:np.load(f"{CORPUS}/run/prof_{S}.npz")["prof"] for S in ["NC-PKU-10","CRC-PKU-32"]}

def run(mc,prof):
    cl=fcluster(Zc,t=1-mc,criterion="distance"); groups={}
    for i,g in enumerate(cl): groups.setdefault(g,[]).append(i)
    gids=sorted(groups); G=len(gids); gmem=[[cts[i] for i in groups[g]] for g in gids]
    Agrp=np.stack([accA[:,groups[g]].mean(1) for g in gids],1)
    cg=Agrp/(Agrp.sum(0,keepdims=True)+1e-12); rm=cg.mean(1,keepdims=True); rs=cg.std(1,keepdims=True)+1e-12
    Zm=(cg-rm)/rs; mk={k:np.argsort(Zm[:,k])[::-1][:400] for k in range(G)}
    R=np.stack([Agrp[mk[k]].mean(0) for k in range(G)],0); Rn=R/(R.sum(0,keepdims=True)+1e-12)
    heme=np.array([any(lheme(x) for x in gmem[k]) for k in range(G)])
    lin=np.array([lineage_of(gmem[k][0]) for k in range(G)])
    y=np.array([dip_of(prof[mk[k]].sum(0)) for k in range(G)])
    u=Rn.mean(1,keepdims=True); Rw=np.hstack([Rn,u]); yn=np.clip(y,0,None); yn/=yn.sum()+1e-12
    w=dc.deconvolve(yn,Rw,l2=0.0,iters=8000)[:G]
    hemem=w[heme].sum()
    # top non-blood lineage
    nb_mass={}
    for k in range(G):
        if not heme[k]: nb_mass[lin[k]]=nb_mass.get(lin[k],0)+w[k]
    top_nb=sorted(nb_mass.items(),key=lambda x:-x[1])[:2]
    return G,hemem,top_nb

print(f"{'mcorr':>6}{'G':>5} | NC-10 heme  (top non-blood) | CRC-32 heme  (top non-blood)")
for mc in [0.92,0.9,0.87,0.85,0.82,0.8,0.75]:
    g,h1,nb1=run(mc,profs["NC-PKU-10"]); _,h2,nb2=run(mc,profs["CRC-PKU-32"])
    s1=" ".join(f"{k}:{v:.2f}" for k,v in nb1); s2=" ".join(f"{k}:{v:.2f}" for k,v in nb2)
    print(f"{mc:>6}{g:>5} | {h1:.3f}  ({s1}) | {h2:.3f}  ({s2})")
