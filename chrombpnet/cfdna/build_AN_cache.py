"""Cache BOTH accessibility and nucleosome-dyad reference matrices over the panel."""
import numpy as np, argparse, os
from chrombpnet.multitask.train_multicell_torch import parse_manifest
from chrombpnet.cfdna import reference as ref
from chrombpnet.cfdna.run_poc import build_R
ap=argparse.ArgumentParser()
ap.add_argument("-m",required=True); ap.add_argument("--manifest-root",required=True)
ap.add_argument("--chroms",nargs="*",default=["chr1","chr3","chr6"])
ap.add_argument("--n-regions",type=int,default=8000); ap.add_argument("--top-per-cell",type=int,default=400)
ap.add_argument("--half-width",type=int,default=500); ap.add_argument("--jobs",type=int,default=12)
ap.add_argument("-o",required=True)
a=ap.parse_args()
man=parse_manifest(a.m,a.manifest_root); cts=[m['cell_type'] for m in man]
print(f"cells={len(cts)} building panel+A(acc,nuc) ...",flush=True)
regions=ref.build_region_panel(man,a.chroms,top_per_cell=a.top_per_cell,n_regions=a.n_regions,half_width=a.half_width,seed=0)
accA,nucA=build_R(man,regions,a.jobs)
np.savez(a.o,accA=accA,nucA=nucA,rchr=regions["chr"].to_numpy(),rc=regions["center"].to_numpy(),cts=np.array(cts))
print(f"cached acc{accA.shape}+nuc{nucA.shape} -> {a.o}",flush=True)
