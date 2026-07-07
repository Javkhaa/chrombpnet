# Literature review + idea generation — cfDNA cell-type deconvolution

_2026-07-07. Reviews where our scATAC-reference / coverage-dip approach sits in the field, and
generates candidate next directions. Companion to `SUMMARY_cfdna_nucleosome.md`._

## Where our approach sits

We deconvolve cfDNA using **scATAC accessibility tracks** as the reference basis and **GC-corrected
composite coverage-dip** over cell-type marker sites as the feature. The field has three broad
reference/feature families; we're in the first, which is the least crowded:

| Family | Reference | Feature | Representative work |
|---|---|---|---|
| **Nucleosome / fragmentation** (ours) | chromatin (ATAC / scRNA-derived) | coverage-dip, WPS, nucleosome spacing | Snyder 2016; Griffin; Nat Commun 2024 |
| **Methylation** | tissue methylation atlas | fragment-level methylation | cfSort (PNAS 2023); cfDecon 2025 |
| **Multi-modal ML** | learned | methylation + fragmentomics + end-motifs + CNV | MRD-EDGE; GCNN tissue-of-origin |

## Key papers and what they tell us

**Snyder et al. 2016, Cell** — the foundation. cfDNA nucleosome occupancy correlates with the
cell-of-origin's chromatin; healthy cfDNA spacing matches lymphoid/myeloid, consistent with
hematopoietic death. This is exactly the ~90% hematopoietic result we validated — our healthy
composition reproduces the field's ground truth. According to PubMed, [DOI](https://doi.org/10.1016/j.cell.2015.11.050).

**Cell type signatures in cfDNA fragmentation profiles reveal disease biology, Nat Commun 2024**
(PMC10933257) — **the closest competitor, and the most important paper for us.** They:
- Use **scRNA (Tabula Sapiens + fetal atlas)** as reference for ~490 cell types — not scATAC.
- Correlate **FFT of the Windowed Protection Score** (nucleosome-spacing *periodicity*) in ~20k
  gene bodies against per-cell-type expression — a **frequency-domain** feature, not raw coverage.
- Report classifiers: CRC 84.7%, early breast 90.1%, multiple myeloma AUC 95.0%, preeclampsia 88.3%
  — from **744 subjects**, at plasma-WGS depth comparable to ours.

The key methodological gap this exposes: **they don't use per-region coverage-dip magnitude (our
feature) — they use the periodicity of nucleosome spacing (FFT of WPS).** Periodicity is far more
robust to per-region Poisson noise at low depth, which is exactly our failure mode. This is the
single most actionable lever in the review.

**cfSort, PNAS 2023** — first deep-learning tissue deconvolution; a methylation atlas (521 samples,
29 tissues) beats fragmentomics on sensitivity. Tells us the **methylation channel** is where DL
has already won. Relevant because the user's dataset includes **MEDIP-seq cfDNA** we haven't touched.

**MRD-EDGE, 2024** — ML-guided WGS signal enrichment, ~300× SNV signal-to-noise for MRD, plus
fragmentomics + allelic-frequency denoising for ultrasensitive low-tumor-fraction detection. This
is the state of the art for the *minor-fraction* problem we failed on — but it's genotype-informed
(tumor-specific), a different (harder-data) regime than tumor-agnostic deconvolution.

**Reviews (Hu 2022, Qi 2023, Ding 2022)** confirm the mature fragmentomic feature set we should be
stacking: fragment **size**, **end motifs**, **jagged ends**, **preferred ends**, WPS/nucleosome
footprints. According to PubMed: [DOI](https://doi.org/10.20517/evcna.2022.34),
[DOI](https://doi.org/10.3390/ijms24021503), [DOI](https://doi.org/10.3390/diagnostics12040978).

## What's genuinely novel in our approach (worth protecting)

- **scATAC reference instead of scRNA.** The Nat Commun 2024 team infers chromatin state indirectly
  from expression; we have direct per-cell-type chromatin accessibility. That's mechanistically
  closer to what cfDNA fragmentation actually measures. Nobody has published a 346-cell-type
  **scATAC**-referenced cfDNA deconvolution.
- **A learned model over the reference** (the ChromBPNet trunk) — currently a negative (Role-b), but
  the asset is unusual.

## Idea generation — ranked

**1. FFT-of-WPS periodicity feature (highest confidence).** Replace/augment the per-region
coverage-dip magnitude with the **periodicity of nucleosome spacing** (FFT of WPS in gene bodies /
marker regions), following Nat Commun 2024. Directly attacks our low-depth failure mode: a periodic
signal integrated over a region is robust where per-region count magnitude is not. This is the
lever most likely to unlock minor-fraction sensitivity, and it reuses our existing marker sets. Low
effort, model-independent.

**2. Add the MEDIP-seq / methylation channel.** The user's dataset has MEDIP-seq cfDNA we've never
used. Methylation is where DL deconvolution demonstrably works (cfSort). A methylation-based
reference (or a methylation feature block stacked onto coverage) is the most proven path to the
cancer detection we couldn't get from fragmentation alone. Medium effort; needs a methylation atlas
reference (WGBS/450k) — separate from our scATAC corpus.

**3. Stack the standard fragmentomic features.** We only use coverage-dip. The field routinely
stacks fragment **size ratio (DELFI)**, **end motifs**, **preferred ends**. Each is orthogonal and
cheap to compute from the fragment parquet we already parse. Even without cancer detection, these
sharpen the reference conditioning. Low effort.

**4. Genome-wide markers** (the previously-recommended lever). chr1/3/6 → all chroms. More sites →
better minor-fraction sensitivity. Model-independent, cheap. Complementary to #1 (more sites × better
per-site feature).

**5. Benchmark against the field's numbers on the same Tao-2023 data.** We have CRC/STAD/NC. Frame
our result against Nat Commun 2024's CRC 84.7% as the target. Turns the "cancer detection doesn't
work" negative into a quantified gap with a clear bar to clear.

## Recommended sequence

Do **#1 (FFT-of-WPS)** first — it's the direct fix for our diagnosed failure mode and reuses
everything. If it moves the needle on CRC-vs-NC separation, combine with **#3** (fragmentomic
stacking) and **#4** (genome-wide). Hold **#2 (methylation/MEDIP)** as the higher-effort, higher-
ceiling parallel track since it's the field's proven cancer-detection channel.

## Sources

- Snyder 2016, Cell — PubMed [DOI](https://doi.org/10.1016/j.cell.2015.11.050)
- Hu 2022 fragmentomics review — PubMed [DOI](https://doi.org/10.20517/evcna.2022.34)
- Qi 2023 fragmentomics biomarker review — PubMed [DOI](https://doi.org/10.3390/ijms24021503)
- Ding 2022 fragmentomics liquid biopsy review — PubMed [DOI](https://doi.org/10.3390/diagnostics12040978)
- [Cell type signatures in cfDNA fragmentation profiles, Nat Commun 2024](https://www.nature.com/articles/s41467-024-46435-0)
- [cfSort, PNAS 2023](https://www.pnas.org/doi/10.1073/pnas.2305236120)
- [MRD-EDGE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7616143/)
- [cfDecon 2025, bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.11.637663)
- [DECA transformer ATAC deconvolution](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11847511/)
- [Knowledge-informed multimodal cfDNA, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.10.20.683167)
