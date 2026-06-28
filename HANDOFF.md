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
| H100 env / training stack | ✅ PyTorch GPU extra installed |
| Two-head model implemented + smoke-tested | ✅ |
| Dyad label builder + dual-target loader + trainer | ✅ committed & pushed |
| End-to-end CPU pipeline smoke test | ✅ passes |
| **Peaks + nonpeaks for Pierce2021** | ⬜ TODO (needs MACS3 + chrombpnet nonpeaks; H100 or CPU) |
| **Head-1 accessibility cut-site track** | ✅ built for GM12878/K562/MCF7 |
| **Multi-task training run** | ✅ 2-epoch GM12878 pilot complete; full run next |
| ENCODE MNase/DNase + TSS validation assets | ⬜ TODO (for validation phase) |
| Head-1 bias-model integration into multi-task | ⬜ deferred (see §7) |

**Bottom line:** everything off-GPU is done and validated. The H100 is needed
only for peak calling (fast) + full training runs.

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
- Head 1 uses `chrombpnet-build-tracks --mode cutsite --plus-shift 4 --minus-shift -4` to build the accessibility cut-site bigwig directly from fragments.
- Head 2 uses fragment centers from mono-nucleosomal fragments; dyad labels are effectively insensitive to the small Tn5 end shift.

### 3.5 Environment: PyTorch-first
- Prima's internal ML stack is PyTorch + `uv` + Python 3.11. The owned multi-task nucleosome path now follows that convention.
- `chrombpnet-train-multitask` points to the PyTorch trainer; legacy training/evaluation code has been pruned from this fork.
- The repo is a `uv` project (`pyproject.toml` + `uv.lock`). Use `uv sync --extra gpu --extra peaks` for PyTorch GPU training + MACS3 peak calling.

### 3.6 Note on Camiel2023 (if ever used)
= Mannens et al. 2024 *Nature* (human fetal brain, 10x, ~27.6k frags/cell → 300 GB). Corpus lists ~679k cells (pre-QC) vs paper's 526,094 — apply QC, don't assume the published set.

---

## 4. Where everything lives

### Code — fork: https://github.com/Javkhaa/chrombpnet
Branch: `nucleosome-head` (feature code + uv build).
- `pyproject.toml` + `uv.lock` — **uv project**. Optional extras: `gpu`/`cpu` (PyTorch), `peaks` (MACS3).
- `chrombpnet/training/models/multitask_nucleosome_torch.py` — PyTorch shared trunk + 2 heads.
- `chrombpnet/multitask/build_label_tracks.py` — build dyad (head-2) / cut-site bigwigs (entry point `chrombpnet-build-tracks`).
- `chrombpnet/multitask/torch_data.py` — PyTorch dataset (two bigwigs, shared crop+revcomp).
- `chrombpnet/multitask/train_multitask_torch.py` — PyTorch training entry (`chrombpnet-train-multitask`).
- `tests/smoke_test_pipeline_torch.py` and `tests/smoke_test_multitask_torch.py` — PyTorch smoke tests.
- `setup_h100.sh`, `call_peaks.sh` — one-command VM bring-up (uv) + peak calling.
- `origin` = your fork, `upstream` = kundajelab.

### Local project: `/Users/javkhlan-ochirganbat/agent_outputs/ATACModel`
- `verify_fragments.py` — Gate 0 checker (has the known blind spot; see §3.3).
- `smoke_test_model.py`, `smoke_test_multitask.py`, `smoke_test_pipeline.py` — CPU smoke tests.
- Env is defined canonically by the fork's `pyproject.toml`/`uv.lock`.
- `data/` — `hg38.genome.fa` (2.9 GB), `hg38.chrom.sizes`, `folds.zip`, gate0 sweep artifacts. (These are re-downloaded on the VM by `setup_h100.sh`; the local copies are for reference.)

### Data — GCS: `gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus/`
- Full corpus, one `{Dataset}/{Dataset}-{tissue}/fragments_standardized.tar.gz` each.
- Inner archive → `fragments_standardized/{Dataset}-{tissue}-{batch}.tsv.gz` (5-col BED: chr,start,end,barcode,count; hg38) + `-metadata.csv` (cell_type, organ, sample, tsse, logUMI, …).
- Pierce2021 GM12878 = `Pierce2021/Pierce2021-GM12878_cells/fragments_standardized.tar.gz`.

### Hosted reference (chrombpnet)
- genome/chrom.sizes: `https://storage.googleapis.com/chrombpnet_data/input_files/{hg38.genome.fa,hg38.chrom.sizes}`
- folds: `https://zenodo.org/records/7443683/files/folds.zip`

---

## 5. How to run on the H100

It's a **uv project** (`pyproject.toml` + `uv.lock`). `setup_h100.sh` runs `uv sync`,
so there's no manual venv — run everything with `uv run` from `$HOME/atac/chrombpnet`.

```bash
# 0. SSH to the VM, then:
git clone -b nucleosome-head https://github.com/Javkhaa/chrombpnet.git
cd chrombpnet
bash setup_h100.sh                 # uv sync, GPU assert, pull data, build label tracks
```
`setup_h100.sh` (edit `CELL_LINE=GM12878|K562|MCF7` at top): installs uv + the env via
`uv sync --extra peaks`, **fails fast if no GPU**, pulls genome/bias/folds, pulls
Pierce2021 fragments from your bucket, concatenates batches into one pseudobulk
fragment file, builds the nucleosome dyad bigwig. Run the rest from `$HOME/atac/chrombpnet`:

```bash
# 1. PEAKS + NONPEAKS (CPU, minutes) — one command:
uv run bash call_peaks.sh $HOME/atac/data/GM12878.fragments.tsv.gz $HOME/atac/peaks
#    -> peaks.narrowPeak + nonpeaks.narrowPeak (MACS3 + chrombpnet GC-matched nonpeaks)

# 2. ACCESSIBILITY BIGWIG
# setup_h100.sh builds the accessibility cut-site bigwig used as --acc-bw.

# 3. MULTI-TASK PYTORCH (head-1 + head-2) — reuses the cut-site bigwig + our dyad track
uv run chrombpnet-train-multitask \
  -p $HOME/atac/peaks/peaks.narrowPeak -n $HOME/atac/peaks/nonpeaks.narrowPeak -g $HOME/atac/data/hg38.fa \
  --acc-bw $HOME/atac/run_head1/.../<cutsite>.bw \
  --nuc-bw $HOME/atac/data/GM12878.nucleosome_dyad.bw \
  -fl $HOME/atac/data/folds/fold_0.json -o $HOME/atac/run_multitask \
  --num-workers 16
```
Hold out a chromosome (the fold JSON already defines train/valid/test). Default model: filters 512, 8 dilated layers, inputlen 2114, outputlen 1000, num-workers 16.

---

## 6. Validation plan (after a clean run)
1. **+1 nucleosome:** predicted dyad occupancy should show the positioned +1 nucleosome just downstream of active TSSs (need GENCODE TSS).
2. **Orthogonal:** predicted accessibility vs ENCODE **DNase**; predicted nucleosome vs ENCODE **MNase-seq** (GM12878/K562).
3. **Held-out chromosome** never seen in training.

---

## 7. Open items / next actions
- [x] **`call_peaks.sh`** helper in the fork (MACS3 + `chrombpnet prep nonpeaks`) — done; verify flag names against your chrombpnet version.
- [x] Build head-1 accessibility cut-site bigwigs for GM12878/K562/MCF7.
- [ ] Run multi-task; tune `--nucleosome-profile-weight` (start 1.0).
- [ ] **Head-1 bias correction in the multi-task model**: currently bias-free. If baseline-quality head-1 becomes important, add a PyTorch bias branch to the accessibility head only. Head 2 stays bias-free because dyads carry no Tn5 cut-site bias.
- [ ] Download ENCODE MNase/DNase (GM12878, K562) + GENCODE TSS for validation.
- [ ] Backport the hardened Gate-0 check into `verify_fragments.py`.
- [ ] If pretraining head-2 corpus-wide later: exclude Buenrostro2018, Li2023a, Liscovitch-Brauer2021.

---

## 8. Gotchas
- **Don't double-shift.** Let chrombpnet auto-detect (don't pass `-ps/-ms`); always pass `-g hg38.fa` so detection works.
- **K562 pseudobulk:** Pierce2021 batches only; drop Liscovitch-Brauer.
- **Chromosome filter:** corpus fragments include unplaced scaffolds (GL/KI) + chrM; restrict to chr1–22,X,Y (folds JSON does this; `build_label_tracks.py` defaults to main chroms).
- **Memory:** `build_label_tracks.py` now uses Polars sparse aggregation by default; smoothed dyad tracks still materialize dense chromosome arrays.
- **PyTorch-only for owned work:** training now goes through `chrombpnet-train-multitask`; legacy training/evaluation code was removed to keep the fork focused.

---

## 9. References
- Proposal: `proposal-multitask-scatac-nucleosome.md`
- Corpus: https://health.tsinghua.edu.cn/human-scatac-corpus/ · paper NAR 2026 D175 · repo https://github.com/xy-chen16/Human-scATAC-Corpus
- ChromBPNet: https://github.com/kundajelab/chrombpnet · wiki tutorial + preprocessing
- Pierce2021 (Spear-ATAC): Nat Commun 12:2969, doi:10.1038/s41467-021-23213-w
