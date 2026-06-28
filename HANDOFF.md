# Handoff — Multi-Task scATAC Nucleosome Model

**Owner:** Jay (Prima Mente) · **Compute:** 1× H100 (Nebius) · **Last updated:** 2026-06-28

This document is the single source of truth to resume work. It captures what was
decided, what was built and verified, where everything lives, and exactly how to
start training on the H100.

---

## 1. Objective (unchanged)

Train a sequence→signal model with a shared conv trunk and **two heads**:
1. **Accessibility** (head 1) — Tn5 cut-site profile + count. Proven ChromBPNet target.
2. **Nucleosome** (head 2, the novelty) — mono-nucleosomal dyad occupancy profile + count,
   derived from fragment insert lengths normally discarded.

Scoped to train on a single H100 in hours–days. See
[proposal-multitask-scatac-nucleosome.md](proposal-multitask-scatac-nucleosome.md) for the full plan.

---

## 2. Status at a glance

| Workstream | State |
|---|---|
| Gate 0 (insert lengths preserved) — go/no-go for head 2 | ✅ **PASS** on Pierce2021 |
| Gate 1 (hg38 + chr naming) | ✅ confirmed |
| Primary dataset chosen | ✅ **Pierce2021** (GM12878 primary, K562 secondary) |
| Corpus-wide Gate 0 sweep (26 datasets) | ✅ 23/26 viable |
| H100 env (TF version) resolved + CPU-proven | ✅ TF 2.15 |
| Two-head model implemented + smoke-tested | ✅ |
| Dyad label builder + dual-target loader + trainer | ✅ committed & pushed |
| End-to-end CPU pipeline smoke test | ✅ passes |
| **Peaks + nonpeaks for Pierce2021** | ⬜ TODO (needs MACS2 on pseudobulk; H100 or CPU) |
| **Head-1 baseline run** | ⬜ TODO (first H100 run; generates cut-site bigwig) |
| **Multi-task training run** | ⬜ TODO (after baseline) |
| ENCODE MNase/DNase + TSS validation assets | ⬜ TODO (for validation phase) |
| Head-1 bias-model integration into multi-task | ⬜ deferred (see §7) |

**Bottom line:** everything off-GPU is done and validated. The H100 is needed
only for peak calling (fast) + the two training runs.

---

## 3. Key decisions & findings

### 3.1 Dataset: Pierce2021 (droplet 10x, Spear-ATAC)
- Cell lines come from these corpus datasets: **GM12878 ← Pierce2021** (only), **K562 ← Pierce2021 + Liscovitch-Brauer2021**, **MCF7 ← Pierce2021**.
- **Pierce2021 is the golden source**: one droplet protocol covering GM12878 + K562 + MCF7, all Gate-0 PASS, all with ENCODE MNase/DNase for orthogonal validation.
- **Use only Pierce2021 batches for K562** — the K562 tissue tar also contains Liscovitch-Brauer2021 sci-ATAC batches that are collapsed (see below); they would poison the pseudobulk.

### 3.2 Gate 0 results (insert-size verdicts)
| Dataset (cell line) | median | modal (share) | sub<100 | mono | di | verdict |
|---|--:|--|--:|--:|--:|---|
| Pierce2021 GM12878 | 177 | 168 (0.7%) | 27% | 42% | 12% | ✅ PASS |
| Pierce2021 K562 | 185 | 168 (0.6%) | 27% | 41% | 15% | ✅ PASS |
| Liscovitch-Brauer2021 K562 | 176 | 175/176 (≈99.7%) | 0% | 100% | 0% | ❌ collapsed |

### 3.3 Corpus-wide sweep: 23/26 datasets nucleosome-viable
- **FAIL (collapsed to a fixed read length):** Buenrostro2018 (76 bp), **Li2023a (101 bp — 1.1M cells, the big loss)**, Liscovitch-Brauer2021 (176 bp).
- Insert preservation is **per-dataset/per-pipeline, not per-assay-class**: Zhang2021 *catlas* (combinatorial indexing) PASSES; Liscovitch (also combinatorial) FAILS. Always gate empirically.
- Artifacts: [data/gate0_sweep_report.txt](data/gate0_sweep_report.txt), [data/gate0_sweep_results.json](data/gate0_sweep_results.json), [data/gate0_sweep_inputs.tsv](data/gate0_sweep_inputs.tsv).
- **`verify_fragments.py` has a known blind spot** (passed Liscovitch because the spike splits across 175+176). The sweep uses a hardened check (±1 bp adjacent-window mass > 50% → fail; sub-nucleosomal < 10% → fail). Backport into the repo checker before trusting a bare PASS.

### 3.4 Corpus standardization & the Tn5 shift — resolved
- The Tsinghua corpus **does not shift coordinates** — "standardization" is metadata/format only; fragment coordinates are inherited raw from each source pipeline.
- **No manual shift needed.** chrombpnet's `reads_to_bigwig.py` runs `auto_shift_detect`: it estimates the existing shift by matching the insertion-site PWM to the Tn5 reference motif (needs `-g hg38.fa`), then applies only the delta to reach the canonical +4/−4. Just feed `-ifrag` + `-g`, do **not** hardcode `-ps/-ms`, and check `bw_shift_qc.png`.
- Head 2 (dyad = fragment center) is shift-insensitive anyway (±4 bp on both ends barely moves a center).

### 3.5 Environment: TF 2.15 (not the pinned 2.8)
- Stock chrombpnet pins `tensorflow==2.8.0`, which has **no Hopper (sm_90) kernels** → would not accelerate on H100.
- Fix: **TF 2.15** (last release with Keras-2 default → chrombpnet code ports with zero changes; verified). Install with `tensorflow[and-cuda]==2.15.*` — it **bundles CUDA 12 libs**, which run fine under the host's **CUDA 13 driver** (backward compatible). System CUDA toolkit version is irrelevant with this install path.
- tfp 0.23.0, numpy<2. Interpretation deps (deeplift/modisco/shap) deferred — not needed to train.
- The repo was **migrated from `setup.py` to a uv project** (`pyproject.toml` + `uv.lock`, branch `uv-migration`). Deps are platform-marked (CUDA on linux, CPU on mac) so `uv sync` works on both; `uv lock` resolves 190 pkgs universally; `uv run` validated (imports, console scripts, package data all OK).

### 3.6 Note on Camiel2023 (if ever used)
= Mannens et al. 2024 *Nature* (human fetal brain, 10x, ~27.6k frags/cell → 300 GB). Corpus lists ~679k cells (pre-QC) vs paper's 526,094 — apply QC, don't assume the published set.

---

## 4. Where everything lives

### Code — fork: https://github.com/Javkhaa/chrombpnet
Branches: `nucleosome-head` (feature code) and **`uv-migration`** (feature code + uv build; use this one on the VM).
- `pyproject.toml` + `uv.lock` — **uv project**. TF is platform-marked: `tensorflow[and-cuda]` on linux (H100), plain `tensorflow` on mac (dev). Optional extras: `peaks` (macs2), `interpret`, `report`.
- `chrombpnet/training/models/multitask_nucleosome_model.py` — shared trunk + 2 heads.
- `chrombpnet/multitask/build_label_tracks.py` — build dyad (head-2) / cut-site bigwigs (entry point `chrombpnet-build-tracks`).
- `chrombpnet/multitask/data_generator.py` — `MultiTaskBatchGenerator` (two bigwigs, shared crop+revcomp).
- `chrombpnet/multitask/train_multitask.py` — training entry (`chrombpnet-train-multitask`).
- `tests/smoke_test_pipeline.py` — end-to-end CPU smoke test.
- `setup_h100.sh`, `call_peaks.sh` — one-command VM bring-up (uv) + peak calling.
- `origin` = your fork, `upstream` = kundajelab. (Legacy `setup.py`/`requirements*.txt` removed on `uv-migration`.)

### Local project: `/Users/javkhlan-ochirganbat/agent_outputs/ATACModel`
- `verify_fragments.py` — Gate 0 checker (has the known blind spot; see §3.3).
- `smoke_test_model.py`, `smoke_test_multitask.py`, `smoke_test_pipeline.py` — CPU smoke tests.
- Env is defined canonically by the fork's `pyproject.toml`/`uv.lock` (`uv-migration` branch).
- `data/` — `hg38.genome.fa` (2.9 GB), `hg38.chrom.sizes`, `bias_models.zip`, `folds.zip`, gate0 sweep artifacts. (These are re-downloaded on the VM by `setup_h100.sh`; the local copies are for reference.)

### Data — GCS: `gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus/`
- Full corpus, one `{Dataset}/{Dataset}-{tissue}/fragments_standardized.tar.gz` each.
- Inner archive → `fragments_standardized/{Dataset}-{tissue}-{batch}.tsv.gz` (5-col BED: chr,start,end,barcode,count; hg38) + `-metadata.csv` (cell_type, organ, sample, tsse, logUMI, …).
- Pierce2021 GM12878 = `Pierce2021/Pierce2021-GM12878_cells/fragments_standardized.tar.gz`.

### Hosted reference (chrombpnet)
- genome/chrom.sizes: `https://storage.googleapis.com/chrombpnet_data/input_files/{hg38.genome.fa,hg38.chrom.sizes}`
- bias models + folds: `https://zenodo.org/records/7443683/files/{bias_models.zip,folds.zip}`

---

## 5. How to run on the H100

It's a **uv project** (`pyproject.toml` + `uv.lock`). `setup_h100.sh` runs `uv sync`,
so there's no manual venv — run everything with `uv run` from `$HOME/atac/chrombpnet`.

```bash
# 0. SSH to the VM, then:
git clone -b uv-migration https://github.com/Javkhaa/chrombpnet.git
cd chrombpnet
bash setup_h100.sh                 # uv sync (TF2.15+cuda), GPU assert, pull data, build dyad track
```
`setup_h100.sh` (edit `CELL_LINE=GM12878|K562|MCF7` at top): installs uv + the env via
`uv sync --extra peaks`, **fails fast if no GPU**, pulls genome/bias/folds, pulls
Pierce2021 fragments from your bucket, concatenates batches into one pseudobulk
fragment file, builds the nucleosome dyad bigwig. Run the rest from `$HOME/atac/chrombpnet`:

```bash
# 1. PEAKS + NONPEAKS (CPU, minutes) — one command:
uv run bash call_peaks.sh $HOME/atac/data/GM12878.fragments.tsv.gz $HOME/atac/peaks
#    -> peaks.narrowPeak + nonpeaks.narrowPeak (MACS2 + chrombpnet GC-matched nonpeaks)

# 2. HEAD-1 BASELINE (stock chrombpnet) — also writes the +4/-4 cut-site bigwig
uv run chrombpnet pipeline -ifrag $HOME/atac/data/GM12878.fragments.tsv.gz -d ATAC \
  -g $HOME/atac/data/hg38.fa -c $HOME/atac/data/hg38.chrom.sizes \
  -p $HOME/atac/peaks/peaks.narrowPeak -n $HOME/atac/peaks/nonpeaks.narrowPeak \
  -fl $HOME/atac/data/folds/fold_0.json -b $HOME/atac/data/bias_models/<ATAC_bias>.h5 \
  -o $HOME/atac/run_head1

# 3. MULTI-TASK (head-1 + head-2) — reuses the cut-site bigwig + our dyad track
uv run chrombpnet-train-multitask \
  -p $HOME/atac/peaks/peaks.narrowPeak -n $HOME/atac/peaks/nonpeaks.narrowPeak -g $HOME/atac/data/hg38.fa \
  --acc-bw $HOME/atac/run_head1/.../<cutsite>.bw \
  --nuc-bw $HOME/atac/data/GM12878.nucleosome_dyad.bw \
  -fl $HOME/atac/data/folds/fold_0.json -o $HOME/atac/run_multitask
```
Hold out a chromosome (the fold JSON already defines train/valid/test). Default model: filters 512, 8 dilated layers, inputlen 2114, outputlen 1000.

---

## 6. Validation plan (after a clean run)
1. **+1 nucleosome:** predicted dyad occupancy should show the positioned +1 nucleosome just downstream of active TSSs (need GENCODE TSS).
2. **Orthogonal:** predicted accessibility vs ENCODE **DNase**; predicted nucleosome vs ENCODE **MNase-seq** (GM12878/K562).
3. **Held-out chromosome** never seen in training.

---

## 7. Open items / next actions
- [x] **`call_peaks.sh`** helper in the fork (MACS2 + `chrombpnet prep nonpeaks`) — done; verify flag names against your chrombpnet version.
- [ ] Run head-1 baseline, confirm GPU + `bw_shift_qc.png` looks right.
- [ ] Run multi-task; tune `--nucleosome-profile-weight` (start 1.0).
- [ ] **Head-1 bias correction in the multi-task model**: currently bias-free. To match stock chrombpnet, graft the frozen bias model onto the accessibility head only (Add in logit space, logsumexp on counts — same pattern as `chrombpnet_with_bias_model.py`). Head 2 stays bias-free (dyads carry no Tn5 cut-site bias). Decide whether baseline-quality head-1 needs it before investing.
- [ ] Download ENCODE MNase/DNase (GM12878, K562) + GENCODE TSS for validation.
- [ ] Backport the hardened Gate-0 check into `verify_fragments.py`.
- [ ] If pretraining head-2 corpus-wide later: exclude Buenrostro2018, Li2023a, Liscovitch-Brauer2021.

---

## 8. Gotchas
- **Don't double-shift.** Let chrombpnet auto-detect (don't pass `-ps/-ms`); always pass `-g hg38.fa` so detection works.
- **K562 pseudobulk:** Pierce2021 batches only; drop Liscovitch-Brauer.
- **Chromosome filter:** corpus fragments include unplaced scaffolds (GL/KI) + chrM; restrict to chr1–22,X,Y (folds JSON does this; `build_label_tracks.py` defaults to main chroms).
- **Memory:** `build_label_tracks.py` allocates ~12 GB for the whole genome (fine on the VM).
- **Keras version:** stay on TF 2.15 (Keras 2). TF ≥2.16 defaults to Keras 3 and breaks the `tf.keras` code unless `TF_USE_LEGACY_KERAS=1` + `tf-keras`.

---

## 9. References
- Proposal: `proposal-multitask-scatac-nucleosome.md`
- Corpus: https://health.tsinghua.edu.cn/human-scatac-corpus/ · paper NAR 2026 D175 · repo https://github.com/xy-chen16/Human-scATAC-Corpus
- ChromBPNet: https://github.com/kundajelab/chrombpnet · wiki tutorial + preprocessing
- Pierce2021 (Spear-ATAC): Nat Commun 12:2969, doi:10.1038/s41467-021-23213-w
