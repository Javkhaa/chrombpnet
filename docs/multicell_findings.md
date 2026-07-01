# Multi-cell-type nucleosome model — findings & status

Session log of building and validating the multi-cell-type ChromBPNet nucleosome
model across the 146-cell-type scATAC corpus, plus the (unresolved) cell-type
conditioning experiment. Everything referenced here is committed on `nucleosome-head`.

## TL;DR

- **Working deliverable:** a comprehensive **146-cell-type** model (`model_146.pt`,
  shared trunk + per-cell-type heads, 256 filters) trained end-to-end and evaluated
  on held-out test chromosomes. **Cell-type specificity is strong (~0.89 centered);
  counts correlation is decent (acc median 0.65); but AUROC plateaus (~0.57) and
  profile/positioning is weak.** It's a validated proof-of-concept, not yet a strong
  predictor.
- **Cell-type conditioning (to break the AUROC plateau) is unresolved:** every variant
  tried (multiplicative FiLM ± bound ± GroupNorm, and additive) eventually **diverges**;
  lower LR delays it. Strong evidence it's an **LR/optimization-schedule** problem, not
  purely the conditioning mechanism. Parked here.

## What was built (all committed, entry points registered)

| Component | Entry point / file | Notes |
|---|---|---|
| Sharded peak calling | `call_peaks_sharded.sh` | per-chrom parallel MACS3; whole-genome `-q` stalls on deep pseudobulks |
| Off-node staging | `stage_cell_type.sh` | pull → tracks → peaks → manifest row; Flyte-safe (`EMIT_ROW_ONLY=1`) |
| Multi-task trainer (1 cell type) | `chrombpnet-train-multitask` | acc cut-site + nucleosome dyad heads |
| **Multi-cell trainer** | `chrombpnet-train-multicell` | shared trunk + per-cell heads (vectorized) or `--conditioned` |
| **Multi-cell eval** | `chrombpnet-eval-multicell` | per-cell accuracy + cell-type specificity on TEST chroms |
| Model-investigation notebook | `notebooks/investigate_model.ipynb` | tokenization → CNN I/O → inference |
| Corpus/curation docs | `docs/dataset_curation.md`, `docs/staging_targets.tsv` | 724 GiB / 68-sample GCS inventory |
| cfDNA deconvolution assessment | `docs/cfdna_deconvolution.md` | reference-based NNLS/QP feasibility |

Trainer features added this session: step-based validation (`--val-every-steps`,
`--val-max-batches`), per-step loss logging (`--log-loss-every` → `trainstep/loss`),
bf16 (`--amp`), `torch.compile` (`--compile`), gradient clipping (`--grad-clip`),
cell-type conditioning (`--conditioned`, `--embed-dim`). wandb logs per-head loss
components, held-out metrics, example plots, diagnostics.

## Data / corpus

- GCS: `gs://cfdx-experiments/.../scatac_corpus/` — 724 GiB, 68 samples, 26 studies.
- Staged (pre-built tracks+peaks): `.../scatac_corpus/staged/` → transferred to
  `/mnt/data/jganbat/scatac_corpus/` (183 GiB). **146 cell types, 20 study-tissues**
  (cardiac, brain, hepatic, intestinal, renal, pulmonary, retinal, adrenal,
  ovarian/testicular, pituitary, 3 tumor types).
- **Gotcha:** the staged `manifest.tsv` was a *partial* aggregate (47/146 rows); rebuild
  the full one with `cat manifest.d/*.tsv`.
- Fold split is **by chromosome, shared across all cell types** (fold_0: train=19 /
  valid=chr8,chr20 / test=chr1,chr3,chr6) — prevents cross-cell-type leakage.

## Model results (held-out TEST chromosomes)

**Single-cell GM12878 (baseline sanity):** acc counts r≈0.63, AUROC 0.84 (15 epochs).

**146-cell baseline (`model_146.pt`, shared trunk + per-cell heads, 256f):**

| Metric | min / median / max | mean |
|---|---|---|
| acc counts r | 0.38 / **0.65** / 0.80 | 0.64 |
| nuc counts r | 0.36 / **0.60** / 0.75 | 0.59 |
| peak-vs-nonpeak AUROC | 0.52 / **0.57** / 0.87 | 0.58 |
| acc profile r | — | 0.31 |
| nuc profile r | — | **0.06** |

Cell-type specificity (accessibility log-counts, centered = the key metric):
**overall 0.87, cell-type-specific (centered) 0.89, per-region median 0.90.**

Interpretation: the model learns **cell-type identity + relative depth well**
(specificity ~0.9, counts ~0.65), but **peak/background discrimination (AUROC) and
per-base shape — especially nucleosome positioning — are weak.** It early-stopped at
~13% of epoch 1 (fast plateau) → capacity-limited (256 filters + shallow heads across
146 tasks) and/or optimization-limited.

## Performance-engineering journey (throughput)

The bottleneck was the **model, not data IO** (data loader ≈ 16.8k samp/s; the network
FS is fine). Key findings:

- **bf16 at 512 filters is pathological:** 512-channel dilated Conv1d in bf16 hits a bad
  cuDNN kernel — 885 samp/s (fp32) → **96 samp/s (bf16), 9× slower.** Do NOT use `--amp`
  at 512 filters. (This masqueraded as a "compile hang" earlier — misdiagnosed at first.)
- **256 filters + bf16 + compile + cudnn.benchmark = 3410 samp/s** (4× the 512-fp32
  baseline). This is the chosen config.
- **Per-cell-type head loop scaled badly:** `for ct in unique(ct_idx)` + `int(ct)` sync
  was fine at 20 cells (3300 samp/s) but 1230 at 146. **Vectorized** (stacked params +
  grouped conv1d, groups=B, gathered weights) → **4606 samp/s @146, cell-count-independent**,
  numerically identical (max|diff|~2e-6).
- Net: full 146 went from infeasible (~15 hr/epoch) to **~1.7 hr/epoch**.
- Also fixed: validation metrics were logged at `step=epoch` while trainstep advanced the
  wandb step → wandb silently dropped them; now logged at the global step.

## Cell-type conditioning experiment (UNRESOLVED — negative results)

Goal: break the AUROC plateau by making the shared trunk cell-type-aware (FiLM on a
per-cell-type embedding + shared heads) instead of only per-cell output heads.

Every variant **diverges** ("stable for a while, then loss explodes"):

| Variant | LR | Diverged around |
|---|---|---|
| Multiplicative FiLM `(1+γ)·x`, unbounded | 1e-3 | step ~1.1k (→73k) |
| + real grad-clip 1.0 | 1e-3 | worse (→574k) |
| + tanh-bounded scale, LR↓ | 5e-4 | slower (→13k by ~1.7k) |
| + **GroupNorm before FiLM** (canonical) | 5e-4 | delayed to ~step 6k |
| **Additive** bias on normed features | 1e-3 | ~step 2.2k |
| Additive | 1e-4 | inconclusive (run didn't stay up) |

Key inferences:
- It's **not** specific to multiplicative FiLM (additive diverges too) → not just
  multiplicative compounding.
- **Lower LR consistently delays divergence** (1e-3→1.1k, 5e-4→6k) → this is an
  **LR / optimization-schedule** instability, not (only) an architecture flaw.
- GroupNorm (normalizing features before conditioning — the canonical FiLM setup) helped
  materially but didn't eliminate it.
- The conditioned model also differs from the baseline in using **shared heads** (one
  acc/nuc head for all 146 cell types) — an unexamined possible contributor (the single
  shared count head must span very different per-cell depths).

The conditioned model itself is correct and cheap (1.79M params, ~4.3–6k samp/s with
GroupNorm/additive) and identity-initialized (cell-invariant at init). Only the training
dynamics are the problem.

## Recommended next steps (in priority order)

1. **LR warmup + lower peak LR (e.g. 200–1000 step linear warmup to 1e-4/2e-4).** The
   "stable then explode" signature is textbook for this; grad-clip alone isn't enough.
   Needs a small LR-scheduler addition to `train_loop.fit` (not yet implemented).
2. **Isolate the cause:** run the conditioned architecture with conditioning frozen to
   identity (bias=0, requires_grad=False) — if it *still* diverges, the culprit is
   GroupNorm/shared-heads, not conditioning.
3. **Try per-cell-type heads WITH conditioning** (don't switch to shared heads at the
   same time as adding conditioning — change one variable at a time).
4. If conditioning proves too finicky: the baseline's plateau may instead be **capacity**
   — try **512 filters in fp32** (bf16 is broken at 512), or deeper heads.
5. Independent of the above: **nucleosome positioning** (profile r ~0.06–0.19) is the
   persistent weakest head and the one the cfDNA deconvolution idea most depends on —
   worth a dedicated push (higher `--nucleosome-profile-weight`, longer training).

## Infra notes

- Box is a **preemptible instance**; on restart it **lost the `/mnt/data` mount** (no
  fstab entry — mounted manually). Corpus + checkpoints live there. Remount required
  after every preemption. The corpus is *also* safe in GCS (source of the staged copy);
  code is safe in git.
- Frequent step-based checkpoints (every 5k steps) limit preemption loss.
- wandb key is in `~/.zshrc`; detached runs must inject it
  (`eval "$(grep 'export WANDB_API_KEY' ~/.zshrc | tail -1)"`).

## Current state at stop

- **GPU idle**, no runs in flight. Baseline `model_146.pt` and `model_subset20.pt`
  present under `/mnt/data/jganbat/scatac_corpus/run/` (needs the mount).
- wandb project: `prima-mente/jg_experiments` (baseline run `multicell-146-vec`;
  diverged conditioning runs `multicell-146-cond*`).
- All code committed + pushed on `nucleosome-head` (through `fc455a5`).
