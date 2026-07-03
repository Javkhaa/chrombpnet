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
- **Cell-type conditioning works (RESOLVED).** Conditioning the trunk on a per-cell-type
  embedding improves everything, and more expressive = better: **baseline < additive < FiLM**
  monotonically (acc counts r 0.64→0.70→**0.72**; AUROC 0.58→0.60→**0.635**; specificity
  ~0.90), all with a ~3× smaller model. The blocker was training instability, fixed by
  **warmup → cosine LR decay**. Best model: `model_146_film.pt`. Nucleosome *positioning*
  is the one thing conditioning does NOT fix — but the raw r≈0.07 is a coverage-noise
  artifact: on a σ=20 smoothed occupancy profile the model hits **0.37 vs a 0.71 split-half
  ceiling (52%)**. Smoothing the *training* target doesn't help (a no-op); reweighting the
  profile loss and adding capacity are the live levers. See **Nucleosome positioning push**.

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

## Cell-type conditioning experiment (RESOLVED — conditioning helps; FiLM best)

**Outcome:** conditioning the shared trunk on a per-cell-type embedding *improves* the
model, and the more expressive the conditioning, the better — a clean monotonic gradient,
all with a ~3× smaller model (shared heads + embedding vs 146 head-pairs).

| Metric (test chroms, mean over 146) | Baseline (per-cell heads) | Additive cond. | FiLM cond. |
|---|---|---|---|
| acc counts r | 0.637 | 0.700 | **0.723** |
| nuc counts r | 0.589 | 0.628 | **0.648** |
| peak-vs-nonpeak AUROC | 0.579 | 0.605 | **0.635** |
| cell-type specificity (centered) | 0.89 | 0.887 | **0.896** |
| acc / nuc profile r | 0.31 / 0.06 | 0.32 / 0.065 | 0.32 / 0.068 |
| params | large | 1.8M | 1.8M |

Weakest cell types gained most (acc counts r min 0.38 → 0.52). **Nucleosome positioning
(profile r ~0.068) is unchanged by conditioning** — the isolated remaining weakness.

**The key that unlocked it — LR schedule.** Every conditioned variant (FiLM and additive)
initially **diverged** ("stable then explodes"); the fix was **linear warmup → cosine LR
decay** (`--warmup-steps`, `--lr-decay-steps`). Warmup alone only delayed divergence
(2.2k→8.6k); adding cosine decay (peak drops through training) removed the sustained-high-LR
regime and both additive and FiLM then trained stably to convergence. GroupNorm before the
conditioning + a bounded `1+tanh(gamma)` FiLM scale were also necessary. Recipe that works:
`--conditioned --cond-mode film --grad-clip 1.0 --warmup-steps 1000 --lr-decay-steps 12000
--learning-rate 5e-4`.

Checkpoints: `model_146.pt` (baseline), `model_146_cond.pt` (additive), `model_146_film.pt`
(FiLM, best). Backed up to `gs://.../scatac_corpus/model_runs/`.

## Nucleosome positioning push (the isolated weakness)

Positioning (nuc **profile** r) is the one head conditioning does not fix. But the raw
number (~0.068) is a **measurement artifact**, not the model's true quality:

- The dyad target is one-fragment-wide spikes on sparse coverage; a raw per-base Pearson
  between two such sparse tracks is near zero **even for two halves of the same data**.
- Evaluating on a Gaussian-**smoothed** (σ=20 bp) occupancy profile, and reporting the
  **split-half reproducibility ceiling** (binomial-thin the observed counts into halves,
  correlate — the max any model could achieve at this coverage), reframes it:

  | Head | profile r (raw) | profile r (σ=20) | ceiling (σ=20) | % of ceiling |
  |---|---|---|---|---|
  | accessibility | 0.320 | 0.525 | 0.796 | 66% |
  | nucleosome | 0.068 | **0.373** | 0.712 | **52%** |

  So the model already captures ~half of the *reproducible* nucleosome positioning signal;
  raw r≈0.07 was measuring coverage noise, not the model. (`--smooth-sigma` in
  `chrombpnet-eval-multicell`.)

**Lever 1 — smooth the training target (`--nuc-smooth-sigma 20`): NO-OP.** Trained a full
FiLM run on the σ=20-smoothed dyad target (occupancy instead of spikes), same recipe. It
trained perfectly stably (zero divergence — the LR recipe holds), but eval was statistically
identical to the raw-target FiLM:

| Metric | raw target (`model_146_film`) | σ=20 target (`model_146_film_smooth`) |
|---|---|---|
| nuc profile r (σ=20) | 0.373 | **0.371** |
| acc profile r (σ=20) | 0.525 | 0.526 |
| acc / nuc counts r | 0.723 / 0.648 | 0.718 / 0.646 |
| AUROC / specificity | 0.635 / 0.896 | 0.627 / 0.897 |

**Conclusion: target representation is not the bottleneck.** The conv trunk was already
recovering all the smooth-scale signal from the spiky target; pre-smoothing adds nothing.
The 0.37-vs-0.71 gap is therefore capacity/optimization or a genuine learnability wall —
not something reshaping the target can close.

**Lever 2 — up-weight the profile loss (`--nucleosome-profile-weight 4`): in flight.**
Isolates the reweighting lever against the matched σ=20 weight-1 run above (only the profile
weight changes). If it lifts nuc profile r above 0.371 (even at the cost of acc), the head
was optimization-starved → then invest in **capacity (512 filters, fp32)**. If it just
trades acc down for no nuc gain, positioning is capacity- or coverage-limited → the honest
next move is 512-fp32 as a last capacity test, else accept ~0.37 and pivot to the cfDNA PoC
(what positioning was for). Run: `model_146_film_pw4`.

### (original divergence log, for reference)

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

## Capacity, loss-reweight, and corpus-scaling experiments (2026-07-02)

Three follow-ups to the FiLM result, all conditioned FiLM with the stable warmup→cosine
recipe. Checkpoints + evals backed up to `gs://.../scatac_corpus/model_runs/`.

| Run | Change vs `model_146_film` | nuc profile r (σ20) | AUROC | acc counts r | specificity |
|---|---|---|---|---|---|
| `model_146_film` (ref) | — | 0.373 / ceil 0.712 | 0.635 | 0.723 | 0.896 |
| `model_146_film_smooth` | train on σ20-smoothed nuc target | 0.371 | 0.627 | 0.718 | 0.897 |
| `model_146_film_pw4` | `--nucleosome-profile-weight 4` | 0.377 | 0.625 | 0.718 | 0.896 |
| `model_146_film_512` | 512 filters, **fp32** (+TF32) | **0.398** / ceil 0.713 | **0.682** | **0.753** | 0.899 |
| `model_315_film` | **315 cell types** (corpus 146→315) | 0.324 / ceil 0.772 | 0.628 | 0.721 | 0.837 |

**Nucleosome positioning is CAPACITY-limited (resolved 2026-07-03).** Target-smoothing and
4× profile-loss weight are no-ops at 256 filters (<0.01), but **512 filters (fp32+TF32) is the
lever that moves it: nuc profile r 0.373→0.398** (52%→56% of ceiling) — and it lifts everything
else too: **peak AUROC 0.635→0.682** (the bullet-2 win), acc counts r 0.723→0.753, acc profile
0.525→0.557, specificity 0.896→0.899. So positioning wasn't target-representation- or
optimization-limited; the shared trunk needed more width. Cost: fp32-512 is ~3.6× slower
(~930 vs ~3410 samp/s). `model_146_film_512` is the best 146 model.

**Corpus scaling 146→315 slightly *lowered* per-cell metrics** (specificity 0.896→0.837,
nuc counts 0.648→0.605). Expected: 315 types is a harder discrimination task trained to the
same ~10k-step budget, so each cell type is seen ~half as often (~8k vs ~17.5k examples) —
**undertrained per-cell**, not a regression in method. The retrain's real purpose was to add
**blood/immune reference cell types** (Granja/Lareau/Satpathy/Mimitou/Buenrostro: CD4/CD8/B/NK,
monocyte, macrophage, erythroid, megakaryocyte, HSC/GMP/CLP) for the cfDNA work — achieved.
**No mature neutrophils/granulocytes** (PBMC prep excludes them; scATAC drops them) — the one
remaining reference gap, handled by an "unknown" deconvolution component. To make the 315 model
*match* the 146, train ~2× longer (raise `--lr-decay-steps`). TF32 (`set_float32_matmul_precision`)
was added to the trainer for the fp32-512 path.

## Infra notes

- Box is a **preemptible instance**; on restart it **lost the `/mnt/data` mount** (no
  fstab entry — mounted manually). Corpus + checkpoints live there. Remount required
  after every preemption. The corpus is *also* safe in GCS (source of the staged copy);
  code is safe in git.
- Frequent step-based checkpoints (every 5k steps) limit preemption loss.
- wandb key is in `~/.zshrc`; detached runs must inject it
  (`eval "$(grep 'export WANDB_API_KEY' ~/.zshrc | tail -1)"`).

## Current state at stop (2026-07-02, instance being stopped)

- **GPU idle**, no runs in flight. All checkpoints/evals backed up to
  `gs://.../scatac_corpus/model_runs/` (survive the `/mnt/data` mount loss).
- **Corpus is now 315 cell types**, staged at `/mnt/data/jganbat/scatac_corpus/staged/`
  (source of truth `gs://.../scatac_corpus/staged/`, manifest 315 rows / 638 bigWigs).
- wandb: `prima-mente/jg_experiments` — latest run `multicell-315-film` (`fu64exg3`).
- Code committed + pushed on `nucleosome-head` (cfDNA `ingest.py`/`deconvolve_real.py`,
  TF32). See `docs/cfdna_deconvolution.md §9` and `docs/HANDOFF_2026-07-02.md` for the
  cfDNA real-data investigation and next steps.
- **Pending:** eval of `model_146_film_512` (capacity question); Griffin cfDNA feature
  pipeline (the real deconvolution blocker).
