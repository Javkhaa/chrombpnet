#!/usr/bin/env bash
# Stage ONE cell type for multi-cell-type training: pull standardized fragments
# from GCS, build accessibility + nucleosome bigWig tracks, call peaks/nonpeaks
# (sharded), and append a row to the training manifest.
#
# Node-agnostic: run wherever you have gcloud + the chrombpnet uv env. Designed
# for off-node bulk staging of many cell types.
#
# Usage:
#   uv run bash stage_cell_type.sh <CELL_TYPE> <gs://.../fragments_standardized.tar.gz> [OUT_ROOT]
#
# Example (paths from docs/dataset_curation.md):
#   uv run bash stage_cell_type.sh Kanemaru2023-cardiomyocyte \
#     gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus/Kanemaru2023/Kanemaru2023-cardiomyocyte_cells/fragments_standardized.tar.gz \
#     ~/atac/corpus
#
# Produces under OUT_ROOT:
#   data/<CELL>.fragments.tsv.gz, data/<CELL>.accessibility_cutsite.bw, data/<CELL>.nucleosome_dyad.bw
#   peaks/<CELL>/peaks.narrowPeak, peaks/<CELL>/nonpeaks.narrowPeak
#   manifest.tsv   (one row per staged cell type; feed to chrombpnet-train-multicell -m)
set -euo pipefail

CELL="${1:?cell_type name required}"
GCS="${2:?gs:// path to fragments_standardized.tar.gz required}"
OUT_ROOT="${3:-$HOME/atac/corpus}"

# Shared reference assets (reuse the GM12878 setup_h100.sh download location by default).
REF_DATA="${REF_DATA:-$HOME/atac/data}"
GENOME="${GENOME:-$REF_DATA/hg38.fa}"
CHROM_SIZES="${CHROMSIZES:-$REF_DATA/hg38.chrom.sizes}"
FOLD="${FOLD:-$REF_DATA/folds/fold_0.json}"

DATA="$OUT_ROOT/data"
PEAKS="$OUT_ROOT/peaks"
MANIFEST="$OUT_ROOT/manifest.tsv"
mkdir -p "$DATA" "$PEAKS"

for t in gcloud chrombpnet-build-tracks macs3 bedtools; do
  command -v "$t" >/dev/null 2>&1 || { echo "missing on PATH: $t (run under 'uv run')" >&2; exit 1; }
done
for f in "$GENOME" "$CHROM_SIZES" "$FOLD"; do
  [ -f "$f" ] || { echo "missing reference: $f (set REF_DATA / GENOME / FOLD)" >&2; exit 1; }
done

# Resolve script dir so we can call the sharded peak caller next to this file.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== [$CELL] 1. pull fragments =="
TAR="$DATA/${CELL}.fragments_standardized.tar.gz"
[ -f "$TAR" ] || gcloud storage cp "$GCS" "$TAR"

FRAGS="$DATA/${CELL}.fragments.tsv.gz"
if [ ! -f "$FRAGS" ]; then
  TMPX="$DATA/${CELL}_frags"; mkdir -p "$TMPX"
  tar xzf "$TAR" -C "$TMPX"
  # concat all per-shard standardized fragment files into one
  cat "$TMPX"/fragments_standardized/*.tsv.gz > "$FRAGS"
  rm -rf "$TMPX"
fi

echo "== [$CELL] 2. build label tracks =="
ACCBW="$DATA/${CELL}.accessibility_cutsite.bw"
NUCBW="$DATA/${CELL}.nucleosome_dyad.bw"
[ -f "$ACCBW" ] || chrombpnet-build-tracks --frag "$FRAGS" --chrom-sizes "$CHROM_SIZES" \
  --out "$ACCBW" --mode cutsite --plus-shift 4 --minus-shift -4 --smooth-sigma 0
[ -f "$NUCBW" ] || chrombpnet-build-tracks --frag "$FRAGS" --chrom-sizes "$CHROM_SIZES" \
  --out "$NUCBW" --mode dyad --min-len 150 --max-len 250 --smooth-sigma 0

echo "== [$CELL] 3. call peaks (sharded) =="
CELL_PEAKDIR="$PEAKS/$CELL"
if [ ! -f "$CELL_PEAKDIR/peaks.narrowPeak" ]; then
  GENOME="$GENOME" CHROMSIZES="$CHROM_SIZES" FOLD="$FOLD" \
    bash "$HERE/call_peaks_sharded.sh" "$FRAGS" "$CELL_PEAKDIR"
fi

echo "== [$CELL] 4. append manifest row =="
# Paths relative to OUT_ROOT so the manifest is portable.
ROW="$(printf '%s\t%s\t%s\t%s\t%s' \
  "$CELL" \
  "peaks/$CELL/peaks.narrowPeak" \
  "peaks/$CELL/nonpeaks.narrowPeak" \
  "data/${CELL}.accessibility_cutsite.bw" \
  "data/${CELL}.nucleosome_dyad.bw")"
[ -f "$MANIFEST" ] || printf '# cell_type\tpeaks\tnonpeaks\tacc_bw\tnuc_bw\n' > "$MANIFEST"
# replace any existing row for this cell type, then append
grep -vP "^${CELL}\t" "$MANIFEST" > "$MANIFEST.tmp" 2>/dev/null || cp "$MANIFEST" "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"
printf '%s\n' "$ROW" >> "$MANIFEST"

cat <<EOF

== [$CELL] STAGED ==
fragments: $FRAGS
acc track: $ACCBW
nuc track: $NUCBW
peaks:     $CELL_PEAKDIR/peaks.narrowPeak
manifest:  $MANIFEST  (train with: chrombpnet-train-multicell -m $MANIFEST --manifest-root $OUT_ROOT -g $GENOME -fl $FOLD -o <out>)
EOF
