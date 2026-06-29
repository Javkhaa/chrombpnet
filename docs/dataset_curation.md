# scATAC Pseudobulk Corpus: Inventory & Multi-Cell-Type Curation Plan

_Generated 2026-06-29. Source bucket: `gs://cfdx-experiments/dna_fm/experiments/jg_experiments/scatac_corpus`._
_Inventory reflects actual bucket contents (sizes are on-disk gzipped tarball bytes, GiB)._

## 0. Executive numbers

- **Total corpus size:** 724.6 GiB across **68 pseudobulk samples** in **26 studies**.
- **Format:** uniform — every sample is exactly one `fragments_standardized.tar.gz`. No exceptions found.
- **Genome build:** hg38, `chr`-prefixed. Verified inside a tarball: standard 10x 5-column fragments `chr  start  end  barcode  count`. Non-main contigs present (`chrUn_*`, `*_random`, `*_alt`, `_GL/_KI` scaffolds) — these must be filtered.
- We have so far trained on **GM12878 only** (1 of 68 samples). The other 67 are untouched training material.

## 1. Full inventory (study → sample → size)

Tarball size is a usable **sequencing-depth proxy** (more fragments → larger gz). It is approximate (barcode-string length and fragment count both contribute), but good for ranking.

| Study | Sample (cell type / tissue) | Size | Notes |
|---|---|---:|---|
| **Ameen2022** | cardiac_invitro_tissue | 12.27 GiB | hPSC-derived cardiomyocytes (in vitro) |
| | cardiac_tissue | 1.57 GiB | primary heart |
| **Buenrostro2018** | bone_marrow_tissue | 969.9 MiB | classic hematopoiesis FACS-sorted scATAC; low depth |
| **Camiel2023** | brain_tissue | **296.33 GiB** | by far the largest single sample; huge brain atlas |
| **Chiou2021** | pancreatic_tissue | 646.5 MiB | islet/pancreas; low depth |
| **Domcke2020** | hepatic_tissue | 16.66 GiB | fetal cell atlas (sci-ATAC) |
| | pulmonary_tissue | 13.98 GiB | fetal |
| | cardiac_tissue | 12.18 GiB | fetal |
| | adrenal_gland | 10.23 GiB | fetal |
| | renal_tissue | 9.75 GiB | fetal |
| | brain_tissue | 8.90 GiB | fetal |
| | intestinal_tissue | 5.29 GiB | fetal |
| | muscle_tissue | 4.18 GiB | fetal |
| | thymic_tissue | 3.40 GiB | fetal |
| | placental_tissue | 2.64 GiB | fetal — unusual lineage |
| | ocular_tissue | 2.41 GiB | fetal eye |
| | pancreatic_tissue | 985.6 MiB | fetal |
| | gastric_tissue | 662.6 MiB | fetal |
| | splenic_tissue | 434.2 MiB | fetal; low depth |
| **Garcia-Alonso2022** | ovarian_tissue | 13.03 GiB | reproductive tract |
| | testicular_tissue | 7.11 GiB | reproductive tract |
| **Granja2019** | bone_marrow_tissue | 4.26 GiB | hematopoiesis + leukemia ref |
| | peripheral_blood_mononuclear_cells | 1.28 GiB | PBMC |
| **Hocker2021** | cardiac_tissue | 3.19 GiB | adult human heart regions |
| **Jin2022** | ovarian_tissue | 5.40 GiB | ovary |
| **Kanemaru2023** | cardiac_tissue | 21.85 GiB | adult heart cell atlas |
| **Lareau2019** | bone_marrow_tissue | 1.65 GiB | mitochondrial/hematopoiesis |
| **Lee2023** | brain_tissue | 14.85 GiB | brain |
| **Li2023a** | brain_tissue | 75.77 GiB | large brain atlas |
| **Li2023b** | brain_tissue | 13.06 GiB | brain |
| **Liang2023** | retinal_tissue | 29.04 GiB | retina |
| **Liscovitch-Brauer2021** | K562_cells | 430.3 MiB | CRISPR screen in K562; low depth |
| **Mimitou2021** | PBMC__ASAPseq_CITEseq_stimulation | 1.39 GiB | multimodal protocol |
| | PBMC__CD4_CRISPR_perturbation | 2.16 GiB | perturbed |
| | PBMC__DOGMAseq_stimulation | 1.52 GiB | multimodal protocol |
| **Muto2021** | kidney_tissue | 3.06 GiB | adult kidney |
| **Pierce2021** | K562_cells | 5.36 GiB | cell line |
| | MCF7_cells | 3.23 GiB | cell line |
| | GM12878_cells | 1.70 GiB | **currently trained** |
| **Satpathy2019** | peripheral_blood_mononuclear_cells | 7.74 GiB | immune atlas |
| | bone_marrow_tissue | 4.06 GiB | immune atlas |
| **Terekhanova2023** | clear_cell_renal_cell_carcinoma | 17.22 GiB | tumor (ccRCC) |
| | glioblastoma | 15.68 GiB | tumor (GBM) |
| | pulmonary_tissue | 10.83 GiB | tumor-adjacent lung |
| | brain_tissue | 5.29 GiB | |
| **Wang2020** | pulmonary_tissue | 982.8 MiB | lung; low depth |
| **Wang2022** | retinal_tissue | 8.79 GiB | retina |
| **Yoshimura2023** | renal_tissue | 16.52 GiB | kidney |
| **Zhang2021** | esophageal_tissue | 4.38 GiB | adult multi-tissue atlas (19 tissues) |
| | vascular_tissue | 1.94 GiB | |
| | muscle_tissue | 1.87 GiB | |
| | colonic_tissue | 1.80 GiB | |
| | cardiac_tissue | 1.34 GiB | |
| | thyroid_tissue | 1.25 GiB | |
| | pancreatic_tissue | 1.02 GiB | |
| | cutaneous_tissue | 877.9 MiB | low depth |
| | intestinal_tissue | 784.9 MiB | low depth |
| | mammary_tissue | 759.1 MiB | low depth |
| | pulmonary_tissue | 748.2 MiB | low depth |
| | gastric_tissue | 653.3 MiB | low depth |
| | adipose_tissue | 602.8 MiB | low depth |
| | nervous_tissue | 432.9 MiB | low depth |
| | adrenal_gland | 332.3 MiB | low depth |
| | uterine_tissue | 266.2 MiB | low depth |
| | ovarian_tissue | 199.1 MiB | low depth |
| | genital_tissue | 165.2 MiB | low depth |
| | hepatic_tissue | 139.4 MiB | lowest depth in corpus |
| **Zhang2022** | pituitary_tissue | 5.37 GiB | unique endocrine lineage |

Per-study totals: Camiel2023 296.3, Domcke2020 91.6, Li2023a 75.8, Terekhanova2023 49.0, Liang2023 29.0, Kanemaru2023 21.9, Garcia-Alonso2022 20.1, Zhang2021 19.4, Yoshimura2023 16.5, Lee2023 14.9, Ameen2022 13.8, Li2023b 13.1, Satpathy2019 11.8, Pierce2021 10.3, Wang2022 8.8, Jin2022 5.4, Zhang2022 5.4, Granja2019 5.5, Mimitou2021 5.1, Hocker2021 3.2, Muto2021 3.1, Buenrostro2018 1.0, Wang2020 1.0, Chiou2021 0.6, Liscovitch-Brauer2021 0.4 GiB.

There is also a sibling file `gs://.../jg_experiments/all_dataset_labels.csv` (ENCODE/ROADMAP bulk labels) and sibling `ENCODE/`, `ROADMAP/`, `human_body_atlas/`, `human_brain_atlas/` dirs — bulk DNase/ATAC references, not part of this scATAC corpus but available if bulk pretraining is ever wanted.

## 2. Diversity characterization (lineage grouping)

The corpus has good breadth. Grouping the 68 samples by biological category:

- **Blood / immune / hematopoietic (8):** Buenrostro2018, Granja2019 (BM + PBMC), Lareau2019, Satpathy2019 (BM + PBMC), Mimitou2021 (×3 PBMC). Plus GM12878 (lymphoblastoid) and K562 (myeloid leukemia line). **This is the best-covered lineage** — and the one we already have via GM12878. Diminishing returns here.
- **Brain / neural (7):** Camiel2023, Domcke2020-brain, Lee2023, Li2023a, Li2023b, Terekhanova2023-brain, Zhang2021-nervous. Heavily represented and high-depth (Camiel 296 GiB, Li2023a 76 GiB).
- **Cardiac (5):** Ameen2022 (×2, incl. in-vitro), Domcke2020-cardiac, Hocker2021, Kanemaru2023, Zhang2021-cardiac. Well covered.
- **Renal / kidney (5):** Domcke2020-renal, Muto2021, Yoshimura2023, Terekhanova-ccRCC (tumor), Zhang? (no). Good coverage incl. a tumor.
- **Epithelial / GI / endodermal (many, low depth):** hepatic, pancreatic, gastric, intestinal/colonic, esophageal, thyroid, lung, mammary across Domcke2020 + Zhang2021 + Chiou2021 + Wang2020. Broad but mostly **low-depth** (Zhang2021 tissues are the bottom of the depth ranking).
- **Reproductive / endocrine (6):** ovarian (Garcia-Alonso, Jin, Zhang), testicular, uterine, genital, adrenal, pituitary (Zhang2022). **Pituitary and placenta are genuinely rare lineages.**
- **Sensory — retina/eye (3):** Liang2023, Wang2022, Domcke2020-ocular. A distinct regulatory program not in blood/heart.
- **Cancer / disease states (5):** Terekhanova2023 (ccRCC, GBM, lung), K562 & MCF7 lines, perturbation experiments (Mimitou CRISPR, Liscovitch K562 CRISPR). Adds dysregulated-chromatin diversity.
- **Cell lines (4):** GM12878, K562 (×2 studies), MCF7. Redundant for generalization but useful as reproducible benchmarks.
- **Structural / mesenchymal:** muscle, vascular, adipose, cutaneous (skin). Present but low-depth.

**Redundant vs. additive:**
- Redundant relative to current GM12878: more blood/immune (PBMC, BM) and more cell lines (K562/MCF7) add little generalization signal.
- Highest *new* diversity per sample: solid epithelial organs (liver, kidney, lung, pancreas, gut), neurons/retina, cardiomyocytes, and rare endocrine/reproductive (pituitary, placenta, ovary/testis).

**Notable gaps (absent from corpus):** no T/B-cell-line-independent skin keratinocyte deep sample, no dedicated osteoblast/chondrocyte, no melanocyte, no clearly-labeled hepatocyte-pure deep sample (only low-depth fetal/atlas liver), limited adult-epithelial depth overall, and only fetal (Domcke) vs adult (Zhang/others) — developmental stage is a confound, not a true gap.

## 3. Recommended curation plan

### 3a. Cell types to add first (priority order, diversity-per-effort)

The goal is to span the maximum number of *distinct regulatory programs* while preferring high-depth, format-clean samples and avoiding redundancy with the blood lineage we already have.

1. **Kanemaru2023 cardiac (21.9 GiB)** — deep adult heart; cardiomyocyte program, completely new vs GM12878.
2. **Yoshimura2023 renal (16.5 GiB)** or **Muto2021 kidney (3.1 GiB)** — epithelial/nephron program; pick the deep one.
3. **Liang2023 retinal (29.0 GiB)** — neuronal/sensory program, very deep, distinct.
4. **Li2023a brain (75.8 GiB)** or **Lee2023 brain (14.9 GiB)** — neuronal; use one deep brain (Li2023a is huge; Lee2023 is a more tractable first pick).
5. **A deep liver/hepatic + lung sample** — best available are Domcke2020-hepatic (16.7 GiB) and Domcke2020/Terekhanova pulmonary (11–14 GiB). Endoderm/epithelium currently missing at depth.
6. **Garcia-Alonso2022 ovarian (13.0 GiB) + testicular (7.1 GiB)** and **Zhang2022 pituitary (5.4 GiB)** — rare endocrine/reproductive lineages; high marginal diversity.
7. **Terekhanova2023 GBM (15.7 GiB) + ccRCC (17.2 GiB)** — tumor/dysregulated chromatin, adds robustness to abnormal states.
8. **One immune top-up only if needed:** Satpathy2019 PBMC (7.7 GiB) — but treat as low priority (redundant with GM12878 lineage).

Defer / use later: all Zhang2021 tissues and other sub-1 GiB samples (low depth, see 3b), the cell lines (benchmark only), and Camiel2023 (296 GiB — process last; storage/compute heavy, redundant brain).

A strong "phase 1" set of ~8 deep, lineage-diverse samples (heart, kidney, retina, brain, liver, lung, ovary/pituitary, one tumor) covers the major missing programs at ~150 GiB of fragments.

### 3b. Quality / QC criteria

Apply per pseudobulk before it enters training:

- **Depth floor:** drop or down-weight samples below a fragment threshold. As a proxy, the **19 sub-1 GiB samples** (all of Zhang2021's smaller tissues, Buenrostro2018, Wang2020, Chiou2021, Domcke splenic/gastric/pancreatic, Liscovitch K562) are at risk; after unpacking, require e.g. ≥ ~20–30M usable fragments and a TSS-enrichment / FRiP sanity check. Below that, peaks become noisy and the dyad signal especially unreliable.
- **Format check:** assert 5 columns, hg38 `chr` prefix, start<end, count≥1. Confirmed consistent across the corpus, but verify per-file after untar.
- **Main-chromosome filter:** keep `chr1–22, X, Y` only; drop `chrUn_*`, `*_random`, `*_alt`, `_GL/_KI` scaffolds (present in the data, confirmed). `build_label_tracks.py` already does this by default (`chr1-22,X,Y only`).
- **Blacklist filtering:** subtract ENCODE hg38 blacklist regions from both peaks and the bias/background before training (standard ChromBPNet practice). Not yet wired into the multitask path — add it.
- **Peak-count sanity:** after the sharded MACS3 caller, expect ~50k–250k peaks for a healthy adult pseudobulk. Flag samples with <~20k peaks (under-powered) or implausibly many.
- **Mito / duplicate check:** ensure chrM is excluded and PCR-duplicate collapsing already applied (10x `count` column handles multiplicity).

### 3c. Multi-cell-type training design — options & recommendation

The current trainer (`train_multitask_torch.py`) takes **one** peaks/nonpeaks set + **one** acc_bw + **one** nuc_bw per run and a chromosome fold JSON; the model (`MultiTaskNucleosomeModel`) has 2 output heads (accessibility, nucleosome dyad) but **no cell-type input**. Options:

1. **Separate model per cell type** — what we do now, replicated N times. Pros: simple, isolates cell types. Cons: no cross-cell-type generalization (the stated goal), N× compute, no shared sequence grammar — defeats the purpose.
2. **One shared-trunk multi-task model, one output head pair per cell type** (ChromBPNet/BPNet-multitask style). Shared convolutional trunk learns general sequence→chromatin grammar; each cell type gets its own (acc, nuc) head pair trained on its own peak set and bigWigs. Pros: shared trunk generalizes, marginal cost per added cell type is small (just heads), handles cell-type-specific peaks naturally. Cons: heads grow with N; need a sample/cell-type index per training region.
3. **One model with cell-type conditioning/embedding** — single head pair, cell type injected as a learned embedding (FiLM/concat into trunk). Pros: scales to arbitrary N cell types with constant head size, enables zero/few-shot to held-out cell types, smallest model. Cons: harder to train, embedding can underfit rare lineages, more architectural change.

**Recommendation: option 2 (shared trunk + per-cell-type head pairs) for the first expansion, architected so it can graduate to option 3.** Rationale: it directly delivers the generalization goal via a shared sequence trunk, is the smallest change to the existing 2-head model (replicate the head, add a cell-type index to the dataset/loss), and is the proven ChromBPNet-multitask recipe; keep the trunk outputs cell-type-agnostic so a conditioning embedding can be swapped in later for zero-shot transfer once N is large.

### 3d. Train / valid / test split strategy

- **Chromosome holdout is the primary axis and must be shared across all cell types.** Use one global fold JSON (same `train`/`valid`/`test` chromosome partition for every cell type). This is what the trainer already consumes via `--fold-json`. A typical split: test = chr1, chr8, chr21 (or the lab's existing fold); valid = chr3, chr16, chr20; train = the rest.
- **Why shared chroms:** if cell type A trains on chr5 and cell type B is tested on chr5, the shared trunk has seen that sequence — label leakage across cell types through the genome. Holding out the *same* chromosomes everywhere prevents it.
- **Cell-type holdout (orthogonal):** to measure generalization to *unseen cell types*, also reserve a few entire cell types (e.g. one tumor and one rare endocrine sample) that appear in **no** split, evaluated only at the end. This is the real test of "comprehensive/generalizable."
- **Don't split a single pseudobulk by cell type internally** — these are already pseudobulks; the unit is the sample.

### 3e. Preprocessing changes to scale 1 → N

- **Manifest-driven pipeline:** replace the single-cell-line setup with a manifest (study, cell type, gs:// tarball, lineage tag, depth/QC metrics) generated from the bucket; iterate the existing pull→concat→bigWig→peak-call pipeline over it. The 26-study/68-sample table above is the seed manifest.
- **Per-cell-type artifacts:** each sample produces its own acc_bw, nuc_bw, peaks, nonpeaks. The dataset must yield `(sequence, acc, nuc, cell_type_id)`; extend `MultiTaskRegionDataset` / `torch_data.py` to carry a cell-type index and the loss to route to the right head pair.
- **Shared negatives/bias:** GC-matched nonpeaks should be drawn per cell type (background differs), but the chromosome fold stays global.
- **Add blacklist subtraction** into `build_label_tracks.py` / peak post-processing (see 3b).
- **Storage/compute:** unpacked fragments are several × the 724 GiB compressed; do not stage the whole corpus at once. Process per-sample, keep only bigWigs + peaks (small) for training, and prefer streaming/lazy reads (`build_label_tracks.py` already uses Polars `scan`). Camiel2023 (296 GiB) alone will dominate disk — handle it last and consider subsampling.

## 4. Risk list

- **Format inconsistency (low risk):** all 68 are `fragments_standardized.tar.gz`, verified 5-col hg38. Residual risk: non-main contigs present in every file (must filter); per-file barcode-prefix scheme should be confirmed after untar for the few multimodal/perturbation samples (Mimitou, Liscovitch).
- **Low-depth samples (high risk for label quality):** 19 samples < 1 GiB, almost all of Zhang2021 and the fetal/atlas tails. These give noisy peaks and especially unreliable nucleosome-dyad tracks. Down-weight or exclude.
- **Label leakage across cell types via shared chromosomes (high risk if mishandled):** the shared trunk sees sequence across all cell types — a per-cell-type chromosome split would leak. Mitigation: single global chromosome fold (3d).
- **Storage / compute scaling (high):** 724 GiB compressed, much larger unpacked; Camiel2023 is 296 GiB by itself. Per-sample processing and discard-after-track-build are mandatory; don't naively download everything.
- **Class / lineage imbalance (medium):** blood/immune and brain are over-represented; rare lineages (pituitary, placenta, retina) are few and some are low-depth. Risk that the model over-fits common lineages. Mitigation: cap per-lineage sample contribution, oversample rare lineages, and use the cell-type-balanced sampler when training the shared trunk.
- **Developmental-stage confound (medium):** Domcke2020 is fetal, others adult; "brain" or "heart" is not one program across stages. Tag stage in the manifest and keep it as a covariate rather than assuming equivalence.
- **Tumor / perturbation samples (low–medium):** Terekhanova tumors, Mimitou/Liscovitch CRISPR samples have dysregulated chromatin and engineered perturbations — valuable for robustness but should be labeled distinctly and probably held for the cell-type-generalization test set, not mixed silently into "normal" training.
- **Cell-line redundancy (low):** K562 appears in 2 studies, plus MCF7/GM12878; useful as benchmarks but add little generalization — don't over-weight.
