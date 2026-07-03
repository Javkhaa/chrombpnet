"""Role-b: use the trained model to predict CLEAN acc+nuc reference tracks for each
cell type at the panel regions (replacing noisy observed bigWig sums). Predicted region
signal = expm1(count head) over the outputlen window. Saves a griffin_AN-style cache with
model-predicted accA/nucA so the deconvolution can be re-run on denoised references."""
import torch, numpy as np, pyfaidx, argparse
from chrombpnet.training.models.multitask_nucleosome_torch import ConditionedMultiCellModel, MultiCellMultiTaskModel
from chrombpnet.training.utils import one_hot

def fetch_seq(genome, chrom, center, width):
    start=int(center-width//2); end=int(center+width//2)
    s=str(genome[chrom][start:end]).upper()
    if len(s)!=width: s=(s+"N"*width)[:width]
    return one_hot.dna_to_one_hot([s])[0].astype(np.float32)

ap=argparse.ArgumentParser()
ap.add_argument("-c","--checkpoint",required=True)
ap.add_argument("--an-cache",required=True,help="observed AN cache: panel regions + adult cell-type list")
ap.add_argument("-g","--genome",required=True)
ap.add_argument("--batch",type=int,default=256); ap.add_argument("-o",required=True)
a=ap.parse_args()

ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
cfg=ck["args"]; model_cts=ck["cell_types"]
if cfg.get("conditioned"):
    model=ConditionedMultiCellModel(len(model_cts),cfg["inputlen"],cfg["outputlen"],cfg["filters"],
                                    cfg["n_dil_layers"],cfg.get("embed_dim",32),cond_mode=cfg.get("cond_mode","film"))
else:
    model=MultiCellMultiTaskModel(len(model_cts),cfg["inputlen"],cfg["outputlen"],cfg["filters"],cfg["n_dil_layers"])
model.load_state_dict(ck["model_state_dict"]); model.to("cuda").eval()
print(f"model: {len(model_cts)} cells, filters={cfg['filters']}, inputlen={cfg['inputlen']}",flush=True)

z=np.load(a.an_cache,allow_pickle=True); rchr=z["rchr"]; rc=z["rc"].astype(int); adult_cts=list(z["cts"])
ci={n:i for i,n in enumerate(model_cts)}
missing=[c for c in adult_cts if c not in ci]
assert not missing, f"{len(missing)} adult cells not in model: {missing[:3]}"
midx=[ci[c] for c in adult_cts]
print(f"panel={len(rc)} regions | adult cells={len(adult_cts)}",flush=True)

genome=pyfaidx.Fasta(a.genome); IL=cfg["inputlen"]
print("fetching sequences ...",flush=True)
seqs=torch.from_numpy(np.stack([fetch_seq(genome,rchr[i],rc[i],IL) for i in range(len(rc))])).float()
accP=np.zeros((len(rc),len(adult_cts)),dtype=np.float32); nucP=np.zeros_like(accP)
print("predicting ...",flush=True)
with torch.no_grad():
    for j,mi in enumerate(midx):
        for b in range(0,len(rc),a.batch):
            sb=seqs[b:b+a.batch].to("cuda"); ctb=torch.full((sb.shape[0],),mi,dtype=torch.long,device="cuda")
            out=model(sb,ctb)  # (acc_prof, acc_cnt, nuc_prof, nuc_cnt)
            accP[b:b+sb.shape[0],j]=torch.expm1(out[1].squeeze(-1)).float().cpu().numpy()
            nucP[b:b+sb.shape[0],j]=torch.expm1(out[3].squeeze(-1)).float().cpu().numpy()
        if (j+1)%50==0: print(f"  {j+1}/{len(midx)} cells",flush=True)
np.savez(a.o,accA=accP,nucA=nucP,rchr=rchr,rc=z["rc"],cts=np.array(adult_cts))
print(f"saved predicted refs acc{accP.shape}+nuc{nucP.shape} -> {a.o}",flush=True)
