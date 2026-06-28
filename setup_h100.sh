#!/usr/bin/env bash
# One-shot H100 VM bring-up for the multi-task scATAC nucleosome model (uv project).
# Idempotent-ish: safe to re-run; skips downloads that already exist.
#
# Usage:  bash setup_h100.sh
# Then follow the printed "NEXT STEPS" for the head-1 baseline and multi-task run.
set -euo pipefail

# ---- config (edit if needed) ----
BUCKET="gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus"
FORK="https://github.com/Javkhaa/chrombpnet.git"
BRANCH="uv-migration"                       # branch with the uv project
WORK="${WORK:-$HOME/atac}"
DATA="$WORK/data"
CELL_LINE="${CELL_LINE:-GM12878}"           # GM12878 | K562 | MCF7  (all Pierce2021)
GENOME_GCS="https://storage.googleapis.com/chrombpnet_data/input_files"
ZENODO="https://zenodo.org/records/7443683/files"
mkdir -p "$WORK" "$DATA"

echo "== 1. clone fork ($BRANCH) =="
[ -d "$WORK/chrombpnet" ] || git clone --branch "$BRANCH" "$FORK" "$WORK/chrombpnet"
cd "$WORK/chrombpnet"

echo "== 2. uv + env from pyproject (Hopper-ready TF 2.15 [and-cuda] on linux) =="
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv sync --extra peaks          # creates .venv, installs project + macs2, from uv.lock

echo "== 3. verify GPU is visible =="
uv run python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); \
print('TF', tf.__version__, 'GPUs:', gpus); assert gpus, 'NO GPU VISIBLE — stop here'"

echo "== 4. reference + bias + folds =="
cd "$DATA"
[ -f hg38.fa ]          || curl -sSL -o hg38.fa "$GENOME_GCS/hg38.genome.fa"
[ -f hg38.chrom.sizes ] || curl -sSL -o hg38.chrom.sizes "$GENOME_GCS/hg38.chrom.sizes"
[ -f hg38.fa.fai ]      || (cd "$WORK/chrombpnet" && uv run python -c "import pyfaidx; pyfaidx.Fasta('$DATA/hg38.fa')")
[ -d bias_models ]      || { curl -sSL -o bias.zip "$ZENODO/bias_models.zip?download=1"; unzip -qo bias.zip -d bias_models; }
[ -d folds ]            || { curl -sSL -o folds.zip "$ZENODO/folds.zip?download=1"; unzip -qo folds.zip -d folds; }

echo "== 5. pull Pierce2021 $CELL_LINE fragments from your bucket =="
FRAGTAR="$DATA/${CELL_LINE}.fragments_standardized.tar.gz"
[ -f "$FRAGTAR" ] || gcloud storage cp \
  "$BUCKET/Pierce2021/Pierce2021-${CELL_LINE}_cells/fragments_standardized.tar.gz" "$FRAGTAR"
[ -d "$DATA/${CELL_LINE}_frags" ] || { mkdir -p "$DATA/${CELL_LINE}_frags"; \
  tar xzf "$FRAGTAR" -C "$DATA/${CELL_LINE}_frags"; }
FRAGS="$DATA/${CELL_LINE}.fragments.tsv.gz"
[ -f "$FRAGS" ] || cat "$DATA/${CELL_LINE}_frags"/fragments_standardized/*.tsv.gz > "$FRAGS"

echo "== 6. build nucleosome dyad track (head-2 label) =="
cd "$WORK/chrombpnet"
NUCBW="$DATA/${CELL_LINE}.nucleosome_dyad.bw"
[ -f "$NUCBW" ] || uv run chrombpnet-build-tracks \
  --frag "$FRAGS" --chrom-sizes "$DATA/hg38.chrom.sizes" \
  --out "$NUCBW" --mode dyad --min-len 150 --max-len 250 --smooth-sigma 0

cat <<EOF

=========================  SETUP COMPLETE  =========================
WORK=$WORK   CELL_LINE=$CELL_LINE   (uv project at \$WORK/chrombpnet)
Run all commands with 'uv run' from \$WORK/chrombpnet.  Built: $NUCBW

NEXT STEPS
0) System tools (once): conda install -c bioconda samtools bedtools ucsc-bedgraphtobigwig meme
1) PEAKS + NONPEAKS:
   uv run bash call_peaks.sh $FRAGS $WORK/peaks
2) HEAD-1 BASELINE (also writes the +4/-4 cut-site bigwig):
   uv run chrombpnet pipeline -ifrag $FRAGS -d ATAC \\
     -g $DATA/hg38.fa -c $DATA/hg38.chrom.sizes \\
     -p $WORK/peaks/peaks.narrowPeak -n $WORK/peaks/nonpeaks.narrowPeak \\
     -fl $DATA/folds/fold_0.json -b $DATA/bias_models/<ATAC_bias>.h5 -o $WORK/run_head1
3) MULTI-TASK (head-1 + head-2):
   uv run chrombpnet-train-multitask \\
     -p $WORK/peaks/peaks.narrowPeak -n $WORK/peaks/nonpeaks.narrowPeak -g $DATA/hg38.fa \\
     --acc-bw $WORK/run_head1/.../<cutsite>.bw --nuc-bw $NUCBW \\
     -fl $DATA/folds/fold_0.json -o $WORK/run_multitask
===================================================================
EOF
