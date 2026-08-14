#!/usr/bin/env bash
# Call peaks + GC-matched nonpeaks for a pseudobulk 10x-style fragment file,
# producing the peaks.narrowPeak / nonpeaks.narrowPeak inputs used by
# multi-task training.
#
# Usage, uv setup style:
#   bash call_peaks.sh <fragments.tsv.gz> <out_dir>
#
# Usage, explicit style:
#   bash call_peaks.sh -f FRAGMENTS -g GENOME_FA -c CHROM_SIZES -fl FOLD_JSON -o OUTDIR
#
# Env defaults for positional usage:
#   DATA=$HOME/atac/data, GENOME=$DATA/hg38.fa,
#   CHROMSIZES=$DATA/hg38.chrom.sizes, FOLD=$DATA/folds/fold_0.json,
#   BLACKLIST optional
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash call_peaks.sh <fragments.tsv.gz> <out_dir>
  bash call_peaks.sh -f FRAGMENTS -g GENOME_FA -c CHROM_SIZES -fl FOLD_JSON -o OUTDIR [options]

Required for flag style:
  -f,  --fragments       10x-style fragments.tsv[.gz] (chr, start, end, barcode, count)
  -g,  --genome          Reference genome FASTA
  -c,  --chrom-sizes     Chrom sizes TSV
  -fl, --chr-fold-path   ChromBPNet fold JSON
  -o,  --outdir          Output directory

Optional:
  -n,  --name            MACS sample name/prefix [pseudobulk]
       --qvalue          MACS q-value cutoff [0.01]
       --gsize           MACS effective genome size [hs]
       --blacklist       BED blacklist passed to chrombpnet prep nonpeaks
       --inputlen        ChromBPNet input length for nonpeaks [2114]
       --stride          ChromBPNet nonpeak genome stride [1000]
       --keep-temp       Keep intermediate BEDPE/temp files
  -h,  --help            Show this help

Notes:
  With MACS3, this script passes 5-column fragments directly as -f FRAG so the
  barcode/count columns are interpreted correctly. If only MACS2 is available,
  it converts to MACS-style 3-column BEDPE fragment intervals and expands the
  5th-column count.
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
GSIZE="hs"
BLACKLIST="${BLACKLIST:-}"
INPUTLEN="2114"
STRIDE="1000"
KEEP_TEMP=0

if [[ $# -eq 2 && "${1#-}" == "$1" ]]; then
  FRAGMENTS="$1"
  OUTDIR="$2"
  shift 2
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
    --gsize) GSIZE="$2"; shift 2 ;;
    --blacklist) BLACKLIST="$2"; shift 2 ;;
    --inputlen) INPUTLEN="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in FRAGMENTS GENOME CHROM_SIZES FOLD_JSON OUTDIR; do
  if [[ -z "${!required}" ]]; then
    echo "Missing required argument: $required" >&2
    usage >&2
    exit 2
  fi
done

for path in "$FRAGMENTS" "$GENOME" "$CHROM_SIZES" "$FOLD_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "File not found: $path" >&2
    exit 1
  fi
done

if command -v macs3 >/dev/null 2>&1; then
  MACS=(macs3)
  MACS_FORMAT="FRAG"
elif command -v macs2 >/dev/null 2>&1; then
  MACS=(macs2)
  MACS_FORMAT="BEDPE"
else
  echo "Neither macs3 nor macs2 is on PATH. Install MACS3 before calling peaks." >&2
  exit 1
fi

if ! command -v chrombpnet >/dev/null 2>&1; then
  echo "chrombpnet is not on PATH. Use uv run or activate the chrombpnet environment first." >&2
  exit 1
fi

if ! command -v bedtools >/dev/null 2>&1; then
  echo "bedtools is not on PATH. It is required by chrombpnet prep nonpeaks." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/chrombpnet_peaks.XXXXXX")"
if [[ "$KEEP_TEMP" -eq 0 ]]; then
  trap 'rm -rf "$TMPDIR"' EXIT
else
  echo "Keeping temp files under $TMPDIR"
fi

BEDPE="$TMPDIR/${NAME}.bedpe"
MACS_OUT="$OUTDIR/macs"
mkdir -p "$MACS_OUT"

decompress_fragments() {
  case "$FRAGMENTS" in
    *.gz|*.bgz) gzip -cd "$FRAGMENTS" ;;
    *) cat "$FRAGMENTS" ;;
  esac
}

if [[ "$MACS_FORMAT" == "FRAG" ]]; then
  MACS_INPUT="$FRAGMENTS"
  MACS_CALL=("${MACS[@]}" callpeak
    -t "$MACS_INPUT"
    -f FRAG
    -g "$GSIZE"
    -n "$NAME"
    --outdir "$MACS_OUT"
    -q "$QVALUE" --call-summits)
else
  echo "== 1. Converting fragments to MACS-style BEDPE intervals =="
  decompress_fragments | awk '
    BEGIN { OFS="\t" }
    $1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ && $2 >= 0 && $3 > $2 {
      count = ($5 == "" || $5 < 1) ? 1 : int($5)
      for (i = 0; i < count; i++) {
        print $1, $2, $3
      }
    }
  ' > "$BEDPE"

  if [[ ! -s "$BEDPE" ]]; then
    echo "No main-chromosome fragments survived conversion; refusing to call peaks." >&2
    exit 1
  fi

  MACS_INPUT="$BEDPE"
  MACS_CALL=("${MACS[@]}" callpeak
    -t "$MACS_INPUT"
    -f BEDPE
    -g "$GSIZE"
    -n "$NAME"
    --outdir "$MACS_OUT"
    -q "$QVALUE" --call-summits)
fi

echo "== 1. Calling peaks with ${MACS[*]} (-f $MACS_FORMAT) =="
"${MACS_CALL[@]}"

RAW_PEAKS="$MACS_OUT/${NAME}_peaks.narrowPeak"
if [[ ! -s "$RAW_PEAKS" ]]; then
  echo "MACS did not produce non-empty peaks: $RAW_PEAKS" >&2
  exit 1
fi

PEAKS="$OUTDIR/peaks.narrowPeak"
NONPEAK_PREFIX="$OUTDIR/nonpeaks"
NONPEAKS="$OUTDIR/nonpeaks.narrowPeak"

echo "== 3. Filtering peaks to main chromosomes =="
awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/' "$RAW_PEAKS" \
  | sort -k1,1 -k2,2n > "$PEAKS"
echo "   peaks: $(wc -l < "$PEAKS")"

if [[ ! -s "$PEAKS" ]]; then
  echo "No main-chromosome peaks after filtering: $PEAKS" >&2
  exit 1
fi

echo "== 4. GC-matched nonpeaks via chrombpnet =="
NONPEAK_ARGS=(
  prep nonpeaks
  -g "$GENOME"
  -c "$CHROM_SIZES"
  -p "$PEAKS"
  -fl "$FOLD_JSON"
  -o "$NONPEAK_PREFIX"
  -il "$INPUTLEN"
  -st "$STRIDE"
)

if [[ -n "$BLACKLIST" ]]; then
  if [[ ! -f "$BLACKLIST" ]]; then
    echo "Blacklist file not found: $BLACKLIST" >&2
    exit 1
  fi
  NONPEAK_ARGS+=(-br "$BLACKLIST")
fi

chrombpnet "${NONPEAK_ARGS[@]}"
cp "${NONPEAK_PREFIX}_negatives.bed" "$NONPEAKS"
echo "   nonpeaks: $(wc -l < "$NONPEAKS")"

cat <<EOF

== DONE ==
Peaks:    $PEAKS
Nonpeaks: $NONPEAKS
EOF
