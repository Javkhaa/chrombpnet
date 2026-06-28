#!/usr/bin/env bash
# Call peaks + GC-matched nonpeaks for a pseudobulk fragment file, producing the
# peaks.narrowPeak / nonpeaks.narrowPeak inputs that `chrombpnet pipeline` and
# train_multitask need. CPU, ~minutes. Run after setup_h100.sh.
#
# Usage:  bash call_peaks.sh <fragments.tsv.gz> <out_dir>
# Env:    GENOME (hg38.fa), CHROMSIZES (hg38.chrom.sizes), FOLD (fold_0.json),
#         BLACKLIST (optional hg38 blacklist .bed.gz)
set -euo pipefail

FRAGS="${1:?usage: call_peaks.sh <fragments.tsv.gz> <out_dir>}"
OUT="${2:?usage: call_peaks.sh <fragments.tsv.gz> <out_dir>}"
DATA="${DATA:-$HOME/atac/data}"
GENOME="${GENOME:-$DATA/hg38.fa}"
CHROMSIZES="${CHROMSIZES:-$DATA/hg38.chrom.sizes}"
FOLD="${FOLD:-$DATA/folds/fold_0.json}"
BLACKLIST="${BLACKLIST:-}"
mkdir -p "$OUT"; cd "$OUT"

echo "== 1. MACS2 peak calling (Tn5 insertion mode) =="
# -f BEDPE on the fragment file; --shift -75 --extsize 150 is standard ATAC.
macs2 callpeak -t "$FRAGS" -f BEDPE -g hs -n pseudobulk \
  --nomodel --shift -75 --extsize 150 --keep-dup all -q 0.01 --call-summits \
  --outdir "$OUT"

echo "== 2. format -> 10-col narrowPeak centered on summit, main chroms only =="
# MACS2 narrowPeak already has summit offset in col 10; keep chr1-22,X,Y.
awk 'BEGIN{OFS="\t"} $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/' \
  pseudobulk_peaks.narrowPeak > peaks.narrowPeak
echo "   peaks: $(wc -l < peaks.narrowPeak)"

echo "== 3. GC-matched nonpeaks (background) via chrombpnet =="
# chrombpnet's helper builds GC-matched negatives given peaks + genome + folds.
BL_ARG=""; [ -n "$BLACKLIST" ] && BL_ARG="-br $BLACKLIST"
chrombpnet prep nonpeaks \
  -g "$GENOME" -p peaks.narrowPeak -c "$CHROMSIZES" \
  -fl "$FOLD" -o "$OUT/output" $BL_ARG
# chrombpnet writes <prefix>_negatives.bed
[ -f output_negatives.bed ] && cp output_negatives.bed nonpeaks.narrowPeak
echo "   nonpeaks: $(wc -l < nonpeaks.narrowPeak 2>/dev/null || echo '??? check output_negatives.bed')"

cat <<EOF

== DONE ==  peaks.narrowPeak + nonpeaks.narrowPeak in $OUT
Note: verify column counts (both must be 10-col narrowPeak with summit in col 10).
If 'chrombpnet prep nonpeaks' flags differ in your version, see:
  https://github.com/kundajelab/chrombpnet/wiki/Preprocessing#generate-non-peaks-background-regions
EOF
