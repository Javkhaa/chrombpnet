# Adult differential nucleosome positioning

This workflow replaces the historical 20-cell differential-positioning pilot with
an explicitly adult, lineage-aware analysis. The pilot included fetal cells and
Pierce cell lines, so its 841 calls are useful candidates but are not an adult atlas.

## Statistical target

The caller tests **positioning shape**, not total nucleosome occupancy. At each
candidate locus it Gaussian-smooths raw dyad counts, normalizes every cell-type
profile to sum to one, and calculates two leave-one-out contrasts:

1. target adult cell type versus other adult cells in the same lineage (primary);
2. target adult cell type versus every other adult cell type (secondary context).

A lineage call must pass minimum depth, total-variation effect, robust-z, and
split-half noise gates. Positive shifts mean the target dyad profile moved right
relative to its lineage consensus.

These are robust atlas-screening calls, not replicate-aware hypothesis tests. The
244 staged tracks are cell-type pseudobulks and do not, by themselves, estimate
between-donor biological variance.

## Required metadata

The five-column model manifest does not encode adulthood or lineage. Supply a TSV
with one row for every manifest cell type. The completed pilot re-score uses the
manifest's tissue-context token as the comparator group:

```text
cell_type	lineage	is_adult	is_cell_line
cardiomyocyte	cardiac	true	false
fetal_neuron	neural	false	false
GM12878	cell_line	true	true
```

`cell_type` must match the manifest exactly. Missing, duplicate, or invalid rows
stop the run. The NFS metadata groups the 244 tracks into 16 explicit tissue
contexts. This grouping is reproducible and useful for screening, but tissue context
is not a molecular cell lineage; audit the table before confirmatory atlas claims.

## Install

```bash
uv sync --extra analysis
```

The caller itself uses core dependencies and does not require a GPU. The `analysis`
extra installs JupyterLab and ipywidgets for the explorer.

## Pass 1: fetal-confound re-score

Recover the original pilot candidate loci from the staged/GCS result bundle and
supply their deduplicated BED as `--regions`. This asks whether the same loci remain
differential after removing fetal cells and cell lines and switching to lineage
comparators.

```bash
uv run chrombpnet-call-diff-nucleosomes \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --regions /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_candidates.bed \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_rescore \
  --window 501 \
  --smooth-sigma 20 \
  --max-shift 100 \
  --min-counts 20 \
  --min-effect 0.05 \
  --min-robust-z 3 \
  --min-lineage-comparators 3 \
  --cache-loci 100
```

## Pass 2: adult discovery

When `--regions` is omitted, the caller takes the strongest peaks from each adult
cell type, merges nearby summits, ranks candidates by cross-cell support, and caps
the analysis for predictable runtime.

Start with a small chromosome/candidate run before the complete pass:

```bash
uv run chrombpnet-call-diff-nucleosomes \
  --manifest /mnt/data/jganbat/scatac_corpus/staged/manifest_adult346.tsv \
  --metadata /mnt/data/jganbat/scatac_corpus/staged/adult_cell_metadata.tsv \
  --out-dir /mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_discovery \
  --chroms chr1 chr3 chr6 \
  --per-cell-peak-limit 50 \
  --max-loci 5000 \
  --merge-bp 100 \
  --window 501 \
  --smooth-sigma 20 \
  --max-shift 100 \
  --min-counts 20 \
  --min-effect 0.05 \
  --min-robust-z 3 \
  --min-lineage-comparators 3 \
  --chunk-size 32 \
  --cache-loci 200
```

The full adult analysis is CPU- and BigWig-I/O-bound. It reads loci in chunks and
keeps only one profile cube in memory, but every locus still requires a profile
from every adult track.

## Outputs

- `differential_nucleosomes.parquet`: every target-by-locus score and call flag;
- `summary.json`: cohort, parameters, candidate count, and call totals;
- `beds/<cell_type>.diffnuc.bed`: primary lineage-aware calls;
- `profiles.npz`: raw and normalized cross-cell profiles for top cached loci.

The NPZ cache is intentionally bounded by `--cache-loci`; the Parquet table retains
scores for all tested loci.

## Notebook

Open `notebooks/adult_differential_nucleosomes.ipynb`, then choose `TARGET_CELL`
and `LOCUS_RANK`. It defaults to the repository's `runs/adult_diffnuc` link; set
`CHROMBPNET_DIFFNUC_RESULT` to use another completed output directory. The notebook
shows:

- adult cohort and lineage QC;
- ranked lineage-aware sites;
- raw dyad and smoothed target/consensus overlays;
- target-minus-lineage and target-minus-global deviations;
- a cross-cell-type positioning heatmap;
- shift and effect distributions;
- coordinates and per-cell BED paths for genome-browser follow-up.

```bash
uv run --extra analysis jupyter lab notebooks/adult_differential_nucleosomes.ipynb
```

## Current execution status

Pass 1 completed on 2026-08-17 against all 244 adult tracks. The cohort contains no
fetal samples or Pierce cell lines. The 841 mixed-pilot target calls collapsed to
553 unique candidate loci for adult-only re-scoring.

The completed output contains 134,932 target-locus tests, 983 lineage-aware calls
across 387 loci and 187 cell types, and 901 global-context calls. The median absolute
lineage-relative shift among primary calls is 25 bp. Artifacts are in
`/mnt/data/jganbat/scatac_corpus/run/adult_diffnuc_pilot_rescore`, with the local
`runs/adult_diffnuc` link used by the notebook.

These results answer whether mixed-pilot loci remain dynamic in the adult cohort.
They are not the unbiased adult discovery atlas; Pass 2 remains the next analysis.
