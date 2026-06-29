#!/usr/bin/env bash
# call_peaks_sharded.sh — parallel per-chromosome MACS3 peak calling + GC-matched
# nonpeaks. A drop-in faster alternative to call_peaks.sh for large pseudobulk
# fragment files where whole-genome `callpeak -q` stalls on the genome-wide
# pvalue-qvalue table.
#
# Strategy:
#   1. ONE decompression pass splits the fragments by chromosome (the only serial step).
#   2. macs3 callpeak runs once per main chromosome, JOBS-way parallel. Each shard is
#      given a per-chromosome effective genome size G_chr = L_chr * (HS_EFF / MAIN_TOTAL)
#      so the genome-wide background lambda density (lambda_BG = N*d/G) matches what a
#      whole-genome run would use.
#   3. Per-chromosome narrowPeaks are concatenated, filtered, sorted.
#   4. GC-matched nonpeaks via `chrombpnet prep nonpeaks` (unchanged from call_peaks.sh).
#
# FDR caveat: q-values are MACS's native position-level Benjamini-Hochberg, computed
# INDEPENDENTLY per chromosome. Versus a single genome-wide FDR pass this is slightly
# more lenient on small chromosomes (smaller test denominator -> marginally more peaks).
# For ChromBPNet training regions this difference is immaterial. If you need a
# bit-identical genome-wide FDR, use the whole-genome call_peaks.sh as the reference.
#
# Usage:
#   bash call_peaks_sharded.sh <fragments.tsv.gz> <out_dir>
#   bash call_peaks_sharded.sh -f FRAG -g GENOME_FA -c CHROM_SIZES -fl FOLD_JSON -o OUTDIR [opts]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash call_peaks_sharded.sh <fragments.tsv.gz> <out_dir>
  bash call_peaks_sharded.sh -f FRAGMENTS -g GENOME_FA -c CHROM_SIZES -fl FOLD_JSON -o OUTDIR [options]

Optional:
  -n,  --name        MACS sample name/prefix [pseudobulk]
       --qvalue      MACS q-value cutoff [0.01]
       --jobs        Parallel chromosome jobs [nproc]
       --blacklist   BED blacklist passed to chrombpnet prep nonpeaks
       --inputlen    ChromBPNet input length for nonpeaks [2114]
       --stride      ChromBPNet nonpeak genome stride [1000]
       --keep-temp   Keep per-chromosome split + MACS temp files
  -h,  --help
EOF
}

DATA="${DATA:-$HOME/atac/data}"
FRAGMENTS=""
GENOME="${GENOME:-$DATA/hg38.fa}"
CHROM_SIZES="${CHROMSIZES:-$DATA/hg38.chrom.sizes}"
FOLD_JSON="${FOLD:-$DATA/folds/fold_0.json}"
OUTDIR=""
NAME="pseudobulk"
QVALUE="0.01"
JOBS="$(nproc)"
BLACKLIST="${BLACKLIST:-}"
INPUTLEN="2114"
STRIDE="1000"
KEEP_TEMP=0

# MACS3 'hs' effective (mappable) genome size. Background density is preserved by
# splitting this proportionally to chromosome length across the main chromosomes.
HS_EFF=2913022398

if [[ $# -eq 2 && "${1#-}" == "$1" ]]; then
  FRAGMENTS="$1"; OUTDIR="$2"; shift 2
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--fragments) FRAGMENTS="$2"; shift 2 ;;
    -g|--genome) GENOME="$2"; shift 2 ;;
    -c|--chrom-sizes) CHROM_SIZES="$2"; shift 2 ;;
    -fl|--chr-fold-path) FOLD_JSON="$2"; shift 2 ;;
    -o|--outdir) OUTDIR="$2"; shift 2 ;;
    -n|--name) NAME="$2"; shift 2 ;;
    --qvalue) QVALUE="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --blacklist) BLACKLIST="$2"; shift 2 ;;
    --inputlen) INPUTLEN="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in FRAGMENTS GENOME CHROM_SIZES FOLD_JSON OUTDIR; do
  [[ -n "${!required}" ]] || { echo "Missing required argument: $required" >&2; usage >&2; exit 2; }
done
for path in "$FRAGMENTS" "$GENOME" "$CHROM_SIZES" "$FOLD_JSON"; do
  [[ -f "$path" ]] || { echo "File not found: $path" >&2; exit 1; }
done

command -v macs3     >/dev/null 2>&1 || { echo "macs3 not on PATH. Use 'uv run'." >&2; exit 1; }
command -v chrombpnet >/dev/null 2>&1 || { echo "chrombpnet not on PATH. Use 'uv run'." >&2; exit 1; }
command -v bedtools  >/dev/null 2>&1 || { echo "bedtools not on PATH (needed by prep nonpeaks)." >&2; exit 1; }

# Prefer pigz for the decompression pass if available (a little faster).
if command -v pigz >/dev/null 2>&1; then DECOMP=(pigz -dc); else DECOMP=(gzip -cd); fi

mkdir -p "$OUTDIR"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/chrombpnet_peaks_sharded.XXXXXX")"
SPLITDIR="$TMPDIR/split"
MACS_ROOT="$TMPDIR/macs"
mkdir -p "$SPLITDIR" "$MACS_ROOT"
if [[ "$KEEP_TEMP" -eq 0 ]]; then trap 'rm -rf "$TMPDIR"' EXIT; else echo "Keeping temp under $TMPDIR"; fi

# Main chromosomes present in chrom.sizes, and total length for the background split.
MAIN_RE='^chr([1-9]|1[0-9]|2[0-2]|X|Y)$'
mapfile -t CHROMS < <(awk -v re="$MAIN_RE" '$1 ~ re {print $1}' "$CHROM_SIZES" | sort -V)
MAIN_TOTAL=$(awk -v re="$MAIN_RE" '$1 ~ re {s+=$2} END{printf "%d", s}' "$CHROM_SIZES")
[[ "${#CHROMS[@]}" -gt 0 && "$MAIN_TOTAL" -gt 0 ]] || { echo "No main chromosomes found in $CHROM_SIZES" >&2; exit 1; }

# Per-chromosome effective genome size manifest: chrom<TAB>Lchr<TAB>G_chr
GSIZE_TSV="$TMPDIR/chrom_gsize.tsv"
awk -v re="$MAIN_RE" -v hs="$HS_EFF" -v tot="$MAIN_TOTAL" \
  '$1 ~ re { printf "%s\t%d\t%d\n", $1, $2, int($2 * hs / tot) }' "$CHROM_SIZES" > "$GSIZE_TSV"

echo "== 1. Split fragments by chromosome (single pass) =="
# Route each main-chrom fragment to its own file. Non-main chroms are dropped (we
# only call peaks on main chromosomes anyway).
"${DECOMP[@]}" "$FRAGMENTS" | awk -v re="$MAIN_RE" -v d="$SPLITDIR" '
  $1 ~ re && $2 >= 0 && $3 > $2 { print > (d "/" $1 ".frags") }
'
echo "   split files: $(ls "$SPLITDIR" | wc -l)"

echo "== 2. Per-chromosome callpeak ($JOBS-way parallel, q=$QVALUE) =="
# Worker: call peaks on one chromosome with its own effective genome size.
call_one() {
  local chrom="$1"
  local frag="$SPLITDIR/$chrom.frags"
  [[ -s "$frag" ]] || { echo "WARN: no fragments for $chrom, skipping" >&2; return 0; }
  local gchr
  gchr=$(awk -v c="$chrom" '$1==c {print $3}' "$GSIZE_TSV")
  [[ -n "$gchr" ]] || { echo "ERROR: no gsize for $chrom" >&2; return 1; }
  local odir="$MACS_ROOT/$chrom"
  mkdir -p "$odir"
  macs3 callpeak -t "$frag" -f FRAG -g "$gchr" -n "$chrom" \
    --outdir "$odir" -q "$QVALUE" --call-summits \
    > "$odir/macs.log" 2>&1 \
    || { echo "ERROR: callpeak failed for $chrom (see $odir/macs.log)" >&2; return 1; }
}
export -f call_one
export SPLITDIR MACS_ROOT GSIZE_TSV QVALUE

printf '%s\n' "${CHROMS[@]}" | xargs -P "$JOBS" -I{} bash -c 'call_one "$@"' _ {}

echo "== 3. Concatenate + filter + sort peaks =="
RAW="$TMPDIR/all_peaks.narrowPeak"
: > "$RAW"
for chrom in "${CHROMS[@]}"; do
  pk="$MACS_ROOT/$chrom/${chrom}_peaks.narrowPeak"
  [[ -s "$pk" ]] && cat "$pk" >> "$RAW"
done
[[ -s "$RAW" ]] || { echo "No peaks produced across any chromosome." >&2; exit 1; }

PEAKS="$OUTDIR/peaks.narrowPeak"
awk -v re="$MAIN_RE" '$1 ~ re' "$RAW" | sort -k1,1 -k2,2n > "$PEAKS"
echo "   peaks: $(wc -l < "$PEAKS")"
[[ -s "$PEAKS" ]] || { echo "No main-chromosome peaks after filtering." >&2; exit 1; }

# Keep the raw concatenated MACS output alongside the final peaks for provenance.
mkdir -p "$OUTDIR/macs"
cp "$RAW" "$OUTDIR/macs/${NAME}_peaks.narrowPeak"

echo "== 4. GC-matched nonpeaks via chrombpnet =="
NONPEAK_PREFIX="$OUTDIR/nonpeaks"
NONPEAKS="$OUTDIR/nonpeaks.narrowPeak"
NONPEAK_ARGS=(prep nonpeaks -g "$GENOME" -c "$CHROM_SIZES" -p "$PEAKS"
  -fl "$FOLD_JSON" -o "$NONPEAK_PREFIX" -il "$INPUTLEN" -st "$STRIDE")
if [[ -n "$BLACKLIST" ]]; then
  [[ -f "$BLACKLIST" ]] || { echo "Blacklist not found: $BLACKLIST" >&2; exit 1; }
  NONPEAK_ARGS+=(-br "$BLACKLIST")
fi
chrombpnet "${NONPEAK_ARGS[@]}"
cp "${NONPEAK_PREFIX}_negatives.bed" "$NONPEAKS"
echo "   nonpeaks: $(wc -l < "$NONPEAKS")"

cat <<EOF

== DONE (sharded) ==
Peaks:    $PEAKS
Nonpeaks: $NONPEAKS
EOF
