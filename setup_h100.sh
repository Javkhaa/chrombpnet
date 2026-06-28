#!/usr/bin/env bash
# One-shot H100 VM bring-up for the multi-task scATAC nucleosome model.
# Idempotent-ish: safe to re-run; skips downloads that already exist.
#
# Usage:  bash setup_h100.sh
# Then follow the printed "NEXT STEPS" to run the head-1 baseline and the
# multi-task training.
set -euo pipefail

# ---- config (edit if needed) ----
BUCKET="gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus"
FORK="https://github.com/Javkhaa/chrombpnet.git"
BRANCH="nucleosome-head"
WORK="${WORK:-$HOME/atac}"
DATA="$WORK/data"
CELL_LINE="${CELL_LINE:-GM12878}"          # GM12878 | K562 | MCF7  (all Pierce2021)
GENOME_GCS="https://storage.googleapis.com/chrombpnet_data/input_files"
ZENODO="https://zenodo.org/records/7443683/files"
mkdir -p "$WORK" "$DATA"

echo "== 1. clone fork ($BRANCH) =="
[ -d "$WORK/chrombpnet" ] || git clone --branch "$BRANCH" "$FORK" "$WORK/chrombpnet"

echo "== 2. python env (3.11) + deps (Hopper-ready TF 2.15) =="
cd "$WORK/chrombpnet"
python3.11 -m venv "$WORK/.venv" 2>/dev/null || python3 -m venv "$WORK/.venv"
# shellcheck disable=SC1091
source "$WORK/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$WORK/chrombpnet/requirements-h100.txt" 2>/dev/null || \
  pip install -q 'tensorflow[and-cuda]==2.15.*' 'tensorflow-probability==0.23.0' \
                 'numpy<2' pandas scipy h5py pyfaidx pyBigWig matplotlib tqdm
pip install -q -e .   # install chrombpnet (incl. our multitask package)

echo "== 3. verify GPU is visible =="
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); \
print('TF', tf.__version__, 'GPUs:', gpus); assert gpus, 'NO GPU VISIBLE — stop here'"

echo "== 4. reference + bias + folds =="
cd "$DATA"
[ -f hg38.fa ]          || { curl -sSL -o hg38.fa "$GENOME_GCS/hg38.genome.fa"; }
[ -f hg38.chrom.sizes ] || curl -sSL -o hg38.chrom.sizes "$GENOME_GCS/hg38.chrom.sizes"
[ -f hg38.fa.fai ]      || python -c "import pyfaidx; pyfaidx.Fasta('hg38.fa')"
[ -d bias_models ]      || { curl -sSL -o bias.zip "$ZENODO/bias_models.zip?download=1"; unzip -qo bias.zip -d bias_models; }
[ -d folds ]            || { curl -sSL -o folds.zip "$ZENODO/folds.zip?download=1"; unzip -qo folds.zip -d folds; }

echo "== 5. pull Pierce2021 $CELL_LINE fragments from your bucket =="
FRAGTAR="$DATA/${CELL_LINE}.fragments_standardized.tar.gz"
[ -f "$FRAGTAR" ] || gcloud storage cp \
  "$BUCKET/Pierce2021/Pierce2021-${CELL_LINE}_cells/fragments_standardized.tar.gz" "$FRAGTAR"
[ -d "$DATA/${CELL_LINE}_frags" ] || { mkdir -p "$DATA/${CELL_LINE}_frags"; \
  tar xzf "$FRAGTAR" -C "$DATA/${CELL_LINE}_frags"; }
# concatenate all batches into one fragment file for pseudobulk
FRAGS="$DATA/${CELL_LINE}.fragments.tsv.gz"
[ -f "$FRAGS" ] || cat "$DATA/${CELL_LINE}_frags"/fragments_standardized/*-*.tsv.gz \
  | grep -v metadata > "$FRAGS" || \
  cat "$DATA/${CELL_LINE}_frags"/fragments_standardized/*.tsv.gz > "$FRAGS"

echo "== 6. build nucleosome dyad track (head-2 label; head-1 cut-site built by chrombpnet) =="
NUCBW="$DATA/${CELL_LINE}.nucleosome_dyad.bw"
[ -f "$NUCBW" ] || python -m chrombpnet.multitask.build_label_tracks \
  --frag "$FRAGS" --chrom-sizes "$DATA/hg38.chrom.sizes" \
  --out "$NUCBW" --mode dyad --min-len 150 --max-len 250 --smooth-sigma 0

cat <<EOF

=========================  SETUP COMPLETE  =========================
WORK=$WORK   CELL_LINE=$CELL_LINE
Built: $NUCBW

NEXT STEPS
1) HEAD-1 BASELINE (also generates the cut-site bigwig + peaks + nonpeaks):
   call MACS2 peaks on \$FRAGS, then run stock chrombpnet:
     chrombpnet pipeline -ifrag $FRAGS -d ATAC \\
       -g $DATA/hg38.fa -c $DATA/hg38.chrom.sizes \\
       -p peaks.narrowPeak -n nonpeaks.narrowPeak \\
       -fl $DATA/folds/fold_0.json -b $DATA/bias_models/<ATAC_bias>.h5 \\
       -o $WORK/run_head1
   (this writes the +4/-4 cut-site bigwig used as --acc-bw below)

2) MULTI-TASK (head-1 + head-2):
   python -m chrombpnet.multitask.train_multitask \\
     -p peaks.narrowPeak -n nonpeaks.narrowPeak -g $DATA/hg38.fa \\
     --acc-bw $WORK/run_head1/.../<cutsite>.bw --nuc-bw $NUCBW \\
     -fl $DATA/folds/fold_0.json -o $WORK/run_multitask
===================================================================
EOF
