# From `fragments_standardized.tar.gz` to a trainable dataset

Instruction for agents / Flyte tasks. Goal: turn **one** pseudobulk dataset's
`fragments_standardized.tar.gz` (from the GCS corpus) into the four artifacts the
multi-cell-type trainer needs, plus a manifest row.

## TL;DR (one command)

Everything below is automated by `stage_cell_type.sh`. For a Flyte task, run **one
per cell type**:

```bash
EMIT_ROW_ONLY=1 \
REF_DATA=/refs \                         # dir with hg38.fa, hg38.chrom.sizes, folds/
TMPDIR=/scratch \                        # large ephemeral volume for the per-chrom split
  bash stage_cell_type.sh <CELL_TYPE> <gs://.../fragments_standardized.tar.gz> <OUT_ROOT>
```

It is idempotent (skips any step whose output already exists), needs **no GPU**, and
emits a race-free `OUT_ROOT/manifest.d/<CELL_TYPE>.tsv` plus a `MANIFEST_ROW ...` line
on stdout. The rest of this doc explains each step so you can verify or re-implement it.

## Inputs

| What | Where | Notes |
|---|---|---|
| Fragments | `gs://…/<Study>/<Study>-<tissue>_cells/fragments_standardized.tar.gz` | from `docs/staging_targets.tsv` |
| Reference genome | `$REF_DATA/hg38.fa` (+ `.fai`) | hg38, shared read-only mount |
| Chrom sizes | `$REF_DATA/hg38.chrom.sizes` | |
| Fold | `$REF_DATA/folds/fold_0.json` | one global fold shared across all cell types |

Tools required on PATH (the staging container image): `gcloud`, `bedtools`, `macs3`,
`chrombpnet-build-tracks`, and `call_peaks_sharded.sh`. (No `torch`/GPU needed.)

## Outputs (under `OUT_ROOT`, paths in the manifest are relative to it)

| Artifact | Path | Role |
|---|---|---|
| Concatenated fragments | `data/<CELL>.fragments.tsv.gz` | intermediate |
| Accessibility track | `data/<CELL>.accessibility_cutsite.bw` | **label** (head 1) |
| Nucleosome track | `data/<CELL>.nucleosome_dyad.bw` | **label** (head 2) |
| Peaks | `peaks/<CELL>/peaks.narrowPeak` | **training regions** |
| Nonpeaks | `peaks/<CELL>/nonpeaks.narrowPeak` | GC-matched negatives |
| Manifest row | `manifest.d/<CELL>.tsv` | one TSV line for the trainer |

## Steps

### 1. Pull + decompress + concatenate fragments
```bash
gcloud storage cp <GS_PATH> data/<CELL>.fragments_standardized.tar.gz
tar xzf data/<CELL>.fragments_standardized.tar.gz -C tmp/
cat tmp/fragments_standardized/*.tsv.gz > data/<CELL>.fragments.tsv.gz
```
The tarball holds per-shard `*.tsv.gz` files; concatenating gzip members is valid.
Format is 10x-style 5-column fragments: `chr  start  end  barcode  count`, hg38 with
`chr` prefix. **Non-main contigs** (`chrUn_*`, `*_random`, `*_alt`, `_GL/_KI`) are
present and are dropped downstream (peak calling restricts to chr1–22,X,Y).
*Check:* `zcat … | head` shows 5 tab-separated columns with `chr*` in col 1.

### 2. Build the two label tracks (the prediction targets)
```bash
# Accessibility cut-site signal (Tn5 insertion sites, +4/-4 shift)
chrombpnet-build-tracks --frag data/<CELL>.fragments.tsv.gz \
  --chrom-sizes $REF_DATA/hg38.chrom.sizes \
  --out data/<CELL>.accessibility_cutsite.bw \
  --mode cutsite --plus-shift 4 --minus-shift -4 --smooth-sigma 0

# Nucleosome dyad signal (midpoints of mono-nucleosome-length fragments, 150–250 bp)
chrombpnet-build-tracks --frag data/<CELL>.fragments.tsv.gz \
  --chrom-sizes $REF_DATA/hg38.chrom.sizes \
  --out data/<CELL>.nucleosome_dyad.bw \
  --mode dyad --min-len 150 --max-len 250 --smooth-sigma 0
```
These bigWigs are what the model regresses against. `smooth-sigma 0` keeps integer
counts (the multinomial profile loss expects count-like values).
*Check:* both `.bw` files are non-empty (hundreds of MB for a deep sample).

### 3. Call peaks + GC-matched nonpeaks (the training regions)
```bash
GENOME=$REF_DATA/hg38.fa CHROMSIZES=$REF_DATA/hg38.chrom.sizes FOLD=$REF_DATA/folds/fold_0.json \
  bash call_peaks_sharded.sh data/<CELL>.fragments.tsv.gz peaks/<CELL>
```
Per-chromosome parallel MACS3 (q=0.01, --call-summits), then `chrombpnet prep nonpeaks`
for GC-matched negatives. Use the **sharded** caller — whole-genome `callpeak -q` stalls
on deep pseudobulks. Output is standard 10-column narrowPeak; col 10 (summit offset)
defines the window center the dataset uses.
*Check:* `peaks.narrowPeak` has ~10⁵ lines for a deep sample (GM12878 gave 220k);
`nonpeaks.narrowPeak` is non-empty. A peak count near zero means the sample is too
shallow — flag it.

### 4. Emit the manifest row
A single tab-separated line (paths relative to `OUT_ROOT`):
```
<CELL>	peaks/<CELL>/peaks.narrowPeak	peaks/<CELL>/nonpeaks.narrowPeak	data/<CELL>.accessibility_cutsite.bw	data/<CELL>.nucleosome_dyad.bw
```
Written to `manifest.d/<CELL>.tsv` (race-free) and echoed to stdout as `MANIFEST_ROW …`.

## After all cell types are staged (one reduce step)
```bash
cat $OUT_ROOT/manifest.d/*.tsv > $OUT_ROOT/manifest.tsv
chrombpnet-train-multicell \
  -m $OUT_ROOT/manifest.tsv --manifest-root $OUT_ROOT \
  -g $REF_DATA/hg38.fa -fl $REF_DATA/folds/fold_0.json \
  -o $OUT_ROOT/run/model --num-workers 16 --wandb --wandb-run-name multicell
```

## QC / gotchas
- **Idempotent:** every step skips if its output exists — safe to retry a failed task.
- **Depth gate:** samples <1 GiB tarball (tier `SKIP_LOWDEPTH` in `staging_targets.tsv`)
  usually yield too few peaks; skip or quarantine. Validate peak count before adding to
  the manifest.
- **Scratch space:** the per-chromosome split decompresses the full fragment file; set
  `TMPDIR` to a volume with ≥ ~5× the tarball size.
- **Resources:** ~16 CPU (MACS runs up to 16 chroms in parallel), ~24 GB RAM, no GPU.
- **Shared fold:** every cell type must use the SAME `fold_0.json` so a held-out
  chromosome is held out across all cell types (prevents cross-cell-type leakage).
- **DEFER_HUGE** samples (Camiel2023 296 GiB, Li2023a 76 GiB): give them a fatter
  scratch/RAM allocation or downsample first.
