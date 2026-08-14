# Unified Nucleosome Project Report Expansion Design

**Date:** 2026-08-14
**Status:** Approved for implementation
**Artifact:** `docs/unified_project_report.html`

## Purpose

Expand the existing unified HTML report into the comprehensive project record for the
sequence-to-nucleosome program. The report must work at two reading depths:

1. leadership and research partners can understand the objective, evidence, biological
   significance, present status, and next decisions without reading implementation detail;
2. ML and bioinformatics collaborators can reconstruct the data flow, model, training run,
   evaluation, and artifact locations from the same document.

The report is a record of completed work and current evidence. It must distinguish measured
results from interpretation, planned work, and hypotheses.

## Scope

The report will cover the full project arc represented in the repository:

- the original cfDNA sequence-to-nucleosome prior;
- the end-motif and receptive-field ablations;
- the cell-conditioning infrastructure and its data limitation;
- the scATAC corpus inventory and insert-length viability gate;
- selection and preparation of Pierce2021 cell lines;
- construction of cell-type-specific accessibility and nucleosome labels;
- the PyTorch multitask implementation;
- the completed GM12878 training and held-out evaluation;
- the multicell diagnostic evidence;
- the roadmap to K562, MCF7, and manifest-driven multicell training.

The report will not invent missing measurements, claim cell-type generalization before it is
tested, or present planned orthogonal validation as completed work. Existing source documents,
run logs, evaluation JSON, tracked code, and generated artifact paths are the evidence base.

## Report Architecture

### 1. Executive layer

The first viewport and opening sections will state the literal project name and the central
question: whether DNA sequence and scATAC fragment geometry can jointly predict accessibility
and nucleosome positioning in a cell-type-aware model.

The executive layer will contain:

- a one-paragraph outcome summary;
- a compact status board for data, labels, implementation, training, evaluation, and multicell
  readiness;
- headline findings with evidence strength labels;
- the immediate decision: strengthen the nucleosome profile output and establish cross-cell-type
  baselines before scaling to the full corpus;
- a milestone timeline from cfDNA prior through the current GM12878 benchmark.

### 2. Scientific narrative

The narrative will explain why each phase existed and what was learned:

- why nucleosome positioning requires a per-base prior;
- what the cfDNA model established on held-out chromosomes;
- why the end-motif ablation separates nuclease cleavage preference from positioning signal;
- why two healthy plasma samples could validate conditioning machinery but not cell-type biology;
- why scATAC fragment lengths provide a direct route to cell-type-specific dyad labels;
- why labels must be built separately for every cell type;
- how the corpus and Pierce2021 cell lines support the first controlled benchmark.

Each phase will state its question, method, result, conclusion, and limitation. Cross-phase claims
will be linked explicitly so the reader can see how the earlier work motivated the present model.

### 3. Data and label methods

This section will document:

- corpus location, size, study/sample counts, format, and genome build;
- representative lineage coverage and the recommended sample-priority strategy;
- Gate 0 insert-size criteria and pass/fail examples;
- Pierce2021 sample selection and the exclusion of collapsed K562 data from another study;
- storage layout under `/mnt/data/jganbat` and the repository data symlink;
- fragment filtering, chromosome filtering, Tn5 cut-site shifts, mono-nucleosomal selection,
  fragment-center aggregation, and bigWig output;
- the role of Polars in sparse aggregation and the remaining dense smoothing step;
- per-cell-type artifacts for GM12878, K562, and MCF7;
- MACS3 peak calling and GC-matched nonpeak generation.

A data-flow diagram will show fragments flowing into accessibility cutsites, nucleosome dyads,
peaks/nonpeaks, the sequence dataset, and the two model heads.

### 4. Model and training methods

The model section will document the active PyTorch path only, while noting that owned TensorFlow
training code was retired. It will describe:

- one-hot sequence input;
- the shared convolutional trunk and dilated layers;
- accessibility and nucleosome profile/count outputs;
- profile and count loss roles;
- shared cropping and reverse-complement augmentation;
- chromosome-based train, validation, and test partitions;
- worker configuration, GPU execution, checkpoint selection, early stopping, and W&B logging;
- the distinction between the current single-cell-type two-head model and the planned multicell
  shared-trunk design.

Exact command blocks will be included for environment setup, label construction, peak calling,
training, held-out evaluation, and multicell entry points where the current repository supports
them. Commands will use paths known to exist in the project and will be labeled as completed or
planned.

### 5. GM12878 experiment record

The completed run will receive a full experiment card:

- dataset and labels;
- peak/nonpeak inputs;
- fold and held-out chromosomes;
- checkpoint, logs, and W&B links;
- epoch-by-epoch training and validation losses;
- best epoch and step;
- early-stopping behavior;
- held-out evaluation sample counts and device settings.

The learning curve will be represented with a self-contained inline chart plus an accessible data
table. The report will not imply that the final epoch checkpoint was selected; it will identify
epoch 7 as the best validation checkpoint.

### 6. Evaluation and interpretation

Results will be grouped by output and evaluation level:

- accessibility count Pearson and Spearman correlations;
- accessibility profile JSD and profile Pearson;
- nucleosome count Pearson and Spearman correlations;
- nucleosome profile JSD and profile Pearson;
- flat-profile baselines;
- peak-versus-nonpeak AUROC;
- reverse-complement count consistency and mean absolute differences.

For every major metric, the report will explain what it measures, whether higher or lower is
better, the observed value, and the practical interpretation. The conclusions will be explicit:

- accessibility is a healthy first benchmark;
- nucleosome counts are promising;
- nucleosome profile prediction beats a flat baseline but remains the principal weakness;
- reverse-complement count consistency is strong but not exact profile equivariance;
- one GM12878 run does not establish multicell or unseen-cell-type generalization.

Claims will use three evidence labels:

- **Established:** directly supported by completed held-out measurements;
- **Encouraging:** supported by a metric but not yet sufficient for the intended biological claim;
- **Open:** hypothesis or planned validation.

### 7. Multicell evidence and roadmap

The report will summarize the existing multicell diagnostic, including the absolute profile
correlation, split-half ceiling, differential-site count, and deviation correlations. It will
explain why absolute prediction can hide failure to learn cell-type-specific residuals.

The roadmap will be staged:

1. export and inspect GM12878 predicted tracks around TSS, CTCF, and differential loci;
2. train matched K562 and MCF7 baselines using their own labels;
3. compare absolute and cell-type-deviation metrics under shared chromosome holdouts;
4. tune nucleosome profile labels, loss weighting, smoothing, and head capacity;
5. move to manifest-driven shared-trunk multicell training;
6. add orthogonal MNase/DNase and TSS-array validation;
7. expand to lineage-diverse corpus samples after QC.

Each stage will include its success criterion and the decision it unlocks.

### 8. Reproducibility appendix

The appendix will include:

- repository, branch, and relevant commit context;
- software and dependency strategy (`uv`, Python 3.11, PyTorch, MACS3, Polars, W&B);
- exact local data, run, model, log, evaluation, and documentation paths;
- W&B run and artifact links;
- command catalogue;
- known limitations and deferred work;
- a glossary of domain and metric terms;
- a source-document map showing which repository document supports each historical section.

## Presentation Design

The report remains a single, private, self-contained HTML file with inline CSS and no network
dependency. The visual treatment will stay restrained and technical: full-width sections,
compact status and metric cards, readable tables, a sticky table of contents on wide screens,
and responsive single-column behavior on mobile.

Charts and diagrams will be inline and accessible. Color will supplement rather than replace
labels. The report will avoid decorative imagery, marketing composition, nested cards, and
excessively large typography. Print styles will preserve section boundaries, tables, commands,
and source paths for PDF export.

## Evidence and Error Handling

- Missing values will be marked as not measured, not inferred.
- Conflicting historical text will be reconciled against current run artifacts and code.
- Planned commands or artifacts will be visibly labeled as planned.
- Paths will be tested for existence where local artifacts are expected.
- External links will be limited to known project resources and W&B run URLs.
- The report will identify pre-existing historical work separately from the current GM12878 run.

## Verification

Before completion, verify:

1. the HTML parses successfully;
2. all required sections and headline metrics are present;
3. local artifact paths referenced as existing are checked;
4. training-table values match `model.log`;
5. held-out metrics match `eval_test.json`;
6. no `TBD`, `TODO`, or unsupported completion claim remains;
7. the report is responsive and printable at the CSS level;
8. only the report file is changed during implementation, apart from this approved design record.

## Acceptance Criteria

The expanded report is complete when a leadership reader can accurately summarize the project
and its present decision in under ten minutes, while a technical collaborator can locate the
inputs, reproduce the completed GM12878 workflow, understand every reported metric, and identify
the exact experiments required to test multicell generalization.
