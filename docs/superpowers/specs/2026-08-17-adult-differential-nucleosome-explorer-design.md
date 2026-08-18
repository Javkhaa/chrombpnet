# Adult Differential Nucleosome Explorer Design

## Goal

Recompute differential nucleosome-positioning sites using only the adult scATAC
reference and provide a notebook for inspecting each call across cell types.

## Scope

The adult cohort is supplied by the staged adult manifest. Cohort metadata must
explicitly assign each manifest cell type a lineage and adult status. The caller
will reject fetal and cell-line rows even when they accidentally appear in the
input. It will not infer age or lineage from display names.

The previous 20-cell result remains a historical pilot. The new result is stored
under a separate output directory and does not overwrite prior BED or summary
files.

## Differential Definition

For each candidate genomic center and target cell type, the caller reads a dyad
count profile, smooths it with a Gaussian kernel, and normalizes the smoothed
profile to a probability distribution. This removes total occupancy and tests
positioning shape.

The primary comparison is the target profile against the leave-one-out consensus
of other adult cell types in the same lineage. The secondary comparison is against
the leave-one-out consensus of all adult cell types. A lineage call requires at
least three comparator cell types; otherwise the target receives only a global
score.

The caller reports:

- lineage and global profile-deviation effects;
- robust z-scores across cell types at the same locus;
- signed positioning shift from bounded cross-correlation;
- target depth and comparator depth;
- a deterministic split-half noise estimate obtained by binomial thinning of raw
  integer dyad counts;
- call flags based on configurable depth, robust-z, effect, and noise thresholds.

The primary `is_differential` flag requires a lineage call. The global flag is
retained for context and for cell types without enough lineage comparators.

## Inputs And Outputs

Inputs:

- five-column multicell manifest: cell type, peaks, nonpeaks, accessibility BigWig,
  nucleosome BigWig;
- metadata TSV with `cell_type`, `lineage`, and `is_adult` columns;
- candidate loci from the deduplicated union of peak summits, optionally capped per
  cell type for bounded runtime;
- standard analysis parameters such as window size, smoothing sigma, and maximum
  shift.

Outputs:

- `differential_nucleosomes.parquet`, one row per target and candidate call;
- `summary.json`, including parameters, cohort composition, and call counts;
- `beds/<cell_type>.diffnuc.bed`, one BED per cell type;
- `profiles.npz`, compact raw and smoothed profiles for called loci so the notebook
  does not repeatedly scan hundreds of BigWigs.

The notebook reads these outputs and provides cohort QC, ranked calls, target and
consensus overlays, target-minus-consensus profiles, cell-type heatmaps, and shift
and effect distributions. A small synthetic result bundle is committed so the
notebook can be opened and exercised without the NFS.

## Implementation Boundaries

`chrombpnet/multitask/differential_nucleosomes.py` owns pure cohort validation,
profile normalization, consensus construction, robust scoring, shift estimation, and
split-half noise estimation. `differential_nucleosomes_runner.py` owns manifest,
BigWig, serialization, and CLI concerns. `notebooks/adult_differential_nucleosomes.ipynb` only reads result
artifacts and visualizes them; it does not contain the expensive caller.

Polars is used for manifests and result tables. NumPy and SciPy handle profile
statistics, pyBigWig handles genomic tracks, and Matplotlib plus ipywidgets power
the notebook.

## Validation

Synthetic tests construct known shifted dyad profiles and tiny BigWigs. They verify
that fetal and cell-line rows are excluded, lineage consensus excludes the target,
known shifts have the correct sign, shallow profiles fail the depth gate, output
schemas are stable, and the notebook references the reusable result bundle rather
than hard-coded production paths.

The real 244-cell analysis cannot run until `/mnt/data/jganbat` is remounted or GCS
authentication is restored. The CLI must fail with actionable missing-file errors in
that state.
