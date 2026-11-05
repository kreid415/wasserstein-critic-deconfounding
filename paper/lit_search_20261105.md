# Gap search repeated 2026-11-05 (Pass 10)

Purpose: repeat the search behind the gap statement in the Introduction (earlier searches: `paper/evidence.md` lines 60-70, up to 2026-10-15).
Window: works published 2026-10-01 to 2026-11-05 (the 15 earlier days overlap the previous search on purpose).

## Sources run
- OpenAlex `/works?search=` with `from_publication_date:2026-10-01`, 25 results per query, the 10 queries of the 2026-10-15 searches (4 of the first search and the 6 of the rigour review's Introduction track): "wasserstein critic discriminator single-cell batch correction"; "adversarial batch correction single-cell wasserstein jensen-shannon comparison"; "single-cell integration adversarial loss benchmark scVI MMD optimal transport"; "domain adversarial single-cell integration over-correction cell type composition"; "Sinkhorn divergence single-cell integration batch effect autoencoder"; "WGAN-GP batch effect removal scRNA-seq"; "Wasserstein critic versus discriminator single-cell batch correction"; "adversarial MMD optimal transport batch correction variational autoencoder single-cell comparison"; "controlled benchmark batch correction loss functions single-cell integration same backbone"; "latent alignment loss comparison scVI batch effect Jensen-Shannon Wasserstein maximum mean discrepancy". 66 unique works.
- Europe PMC REST search, one query (`("batch correction" OR "batch effect" OR "data integration") AND ("single-cell" OR scRNA) AND (Wasserstein OR adversarial OR Sinkhorn OR "optimal transport" OR "maximum mean discrepancy")`, first publication date 2026-10-01 to 2026-11-05): 16 hits.
- arXiv API: the request timed out (60 s) and was not repeated, so the direct arXiv search was not completed. The OpenAlex results
  may include arXiv preprints that OpenAlex indexes (arXiv DOIs appear among the records below), but this does not show that every
  relevant arXiv preprint was retrieved (corrected in Pass 12; the paper names only OpenAlex and Europe PMC for this repeat). **Not run: bioRxiv/medRxiv search outside OpenAlex and Europe PMC; web search.**

## Screening
Titles screened by the agent that ran the search; abstracts read for the four titles that touch single-cell alignment or adversarial models (Dynamic Neural Optimal Transport for Subset Alignment; Optimal transport reveals unresolved cells in single-cell multi-omics; BASILISC; Generative Adversarial Networks for Omics Data, a livestock review). None compares JS (discriminator), W1 (critic) and MMD or OT objectives on one scVI backbone. The two Europe PMC items closest in topic (scAICT, adversarial RNA-ATAC integration; regulatory-prior attention for unpaired RNA-ATAC integration) are new cross-modality methods, not controlled comparisons of alignment losses.
Verdict: the gap statement stands, as a search-based absence (not exhaustive); the search was repeated on 2026-11-05.

## OpenAlex records (date, type, title, DOI)
- 2026-10-01 | review | Deep learning for diabetic foot ulcer: a systematic review of segmentation, classification and multimodal integration | https://doi.org/10.1007/s10586-026-06619-3
- 2026-10-01 | review | AI-driven quality enhancement of medical datasets: a closed-loop framework of synthesis, annotation, quality control, and integration | https://doi.org/10.3389/fmed.2026.1947584
- 2026-10-01 | article | DermFusion-CBAMNet: A Dual-Backbone Attention-Guided Framework for Robust Multiclass Skin Disease Classification | https://doi.org/10.4108/eetiot.13764
- 2026-10-02 | preprint | Robustness of quantum spectrum estimation: weak Schur sampling under noisy inputs | https://doi.org/10.48550/arxiv.2610.03582
- 2026-10-02 | review | Beyond M1 M2: single-cell insights into how glioblastoma reshapes tumour-associated macrophages | https://doi.org/10.3389/fimmu.2026.1930848
- 2026-10-02 | review | Machine Learning for Screening Bioactive Compounds in Traditional Chinese Medicine: Computational Prediction, Multi-Omics Integration, and Clinical Translation  | https://doi.org/10.3390/ph19101567
- 2026-10-03 | article | Physics-informed cross-domain image synthesis for microalgae growth phase classification in turbid media | https://doi.org/10.1016/j.ecoinf.2026.104069
- 2026-10-03 | preprint | Only Project Once: Projection-Adaptive Loss for Exact Constraint Satisfaction | https://doi.org/10.48550/arxiv.2610.04572
- 2026-10-05 | preprint | AI-assisted super-resolution cosmological simulations V: Cosmology-aware super-resolution | https://doi.org/10.48550/arxiv.2610.06710
- 2026-10-06 | article | Machine learning in renewable-to-hydrogen systems: integrating predictive modelling, feature learning, and optimization for trustworthy green hydrogen productio | https://doi.org/10.1016/j.clce.2026.100234
- 2026-10-06 | preprint | Single-cell entropy as a selectable and inducible marker of stem-cell youth: a multi-track technical architecture for producing young autologous hematopoietic s | https://doi.org/10.21203/rs.3.rs-11245878/v1
- 2026-10-07 | article | What Random Splits Measure in Polymer Glass-Transition Prediction: Chemistry Holdouts, Applicability Domains, and Prior Integration | https://doi.org/10.3390/polym18192445
- 2026-10-07 | preprint | DisParQ: Self-Supervised Part Concepts for Interpretable Vision Foundation Models | https://doi.org/10.48550/arxiv.2610.09802
- 2026-10-08 | preprint | SP-DocReader: Difference-Aware Self-Play for Precise Document OCR | https://doi.org/10.48550/arxiv.2610.11148
- 2026-10-08 | preprint | Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data | https://doi.org/10.48550/arxiv.2610.11407
- 2026-10-08 | article | Near-duplicate contamination inflates brain MRI benchmark performance: a size-matched, multi-seed MedNeXt-B study | https://doi.org/10.3389/fmed.2026.1955863
- 2026-10-09 | article | Dynamic user interest modeling and prediction integrating query box embedding with graph neural networks | https://doi.org/10.1038/s41598-026-74718-7
- 2026-10-09 | book-chapter | Advances in Dental Imaging and the Integration of Artificial Intelligence | https://doi.org/10.58830/ozgur.pub1444.c5834
- 2026-10-09 | preprint | Sparse Dictionary Learning for Measure-Valued Data via Linearized Optimal Transport: Efficient Algorithms and Sample Complexity | https://doi.org/10.48550/arxiv.2610.12924
- 2026-10-11 | preprint | Concurrent de novo inference of cell types and gene regulatory networks with NetMap | https://doi.org/10.64898/2026.10.04.756614
- 2026-10-12 | preprint | Penumbria: Advanced 3D cell segmentation for biomedical imaging | https://doi.org/10.7554/elife.112118
- 2026-10-13 | preprint | Learning Perturbation-Specific Representations through Conditional Energy Prior for Cellular Response Prediction | https://doi.org/10.48550/arxiv.2610.18456
- 2026-10-13 | preprint | BioMod: Biological Module Tokenization for Single-Cell Foundation Modeling | https://doi.org/10.64898/2026.10.12.758490
- 2026-10-14 | article | CSPT-SleepNet: An Automatic Sleep Staging Framework Integrating Center-Driven Sequence Supervision and Multimodal Pyramid Token Encoding | https://doi.org/10.3390/brainsci16101108
- 2026-10-15 | preprint | Panoramic modelling of single-cell transcriptomes at scale with CellNet | https://doi.org/10.48550/arxiv.2610.21943
- 2026-10-15 | preprint | FlashControl: One-Step Controllable Generator via Distillation-Friendly Single-Stream Teachers | https://doi.org/10.48550/arxiv.2610.22537
- 2026-10-16 | review | Single-Cell Whole-Transcriptome Detection Expands the Observable RNA Space for AI Virtual-Cell Models | https://doi.org/10.3390/bios16100581
- 2026-10-16 | preprint | Dynamic Neural Optimal Transport for Subset Alignment | https://doi.org/10.48550/arxiv.2610.24258
- 2026-10-16 | preprint | The Loss or the Architecture? What Matters in Survival Analysis with Competing Risks | https://doi.org/10.48550/arxiv.2610.23322
- 2026-10-16 | preprint | Quantum error correction well below the color code threshold | https://doi.org/10.48550/arxiv.2610.24077
- 2026-10-16 | preprint | Benchmarking and Improvement of Artifact Segmentation Methods in H&E-Stained Whole Slide Imaging | https://doi.org/10.48550/arxiv.2610.24037
- 2026-10-19 | preprint | MOFFieldBench: A Machine Learning Benchmark for Continuous Electronic Fields in Metal-Organic Frameworks | https://doi.org/10.48550/arxiv.2610.27036
- 2026-10-20 | preprint | CrystalAR: Autoregressive Modeling for Symmetric Crystal Generation | https://doi.org/10.48550/arxiv.2610.27762
- 2026-10-20 | preprint | Optimal transport reveals unresolved cells in single-cell multi-omics | https://doi.org/10.64898/2026.10.15.759426
- 2026-10-20 | article | Steering or building? How crisis type shapes organisational change in the EU | https://doi.org/10.1080/01402382.2026.2740495
- 2026-10-21 | preprint | Retrieval Augmented Generation Compositionality | https://doi.org/10.48550/arxiv.2610.29722
- 2026-10-21 | article | Multi-cohort transcriptomics and donor-aware single-cell analyses prioritize S100A8 within an emergency-myeloid, HLA-DR-low state in sepsis | https://doi.org/10.1371/journal.pone.0359578
- 2026-10-22 | preprint | No Perfect Randomized Composition | https://doi.org/10.48550/arxiv.2610.31638
- 2026-10-22 | preprint | The Geometry of Meaning in Neural Manifolds | https://doi.org/10.48550/arxiv.2610.30938
- 2026-10-22 | preprint | IFC-SSL-Bench: Benchmarking Self-Supervised Learning for Annotation-Efficient Multispectral Imaging Flow Cytometry | https://doi.org/10.48550/arxiv.2610.31848
- 2026-10-23 | article | Physics-constrained two-stage reconstruction of contact pressure and real contact support in rough elastic contact | https://doi.org/10.26599/frict.2026.9441327
- 2026-10-23 | preprint | Quantifying punctate morphology in mesenchymal stem cells from single cell images provides a definitive quality assessment pipeline for direct correlation to os | https://doi.org/10.64898/2026.10.22.760947
- 2026-10-23 | article | From seed centres to tray cells: a geometry-aware machine vision framework for full-tray sowing quality assessment in maize plug trays | https://doi.org/10.3389/fpls.2026.1981493
- 2026-10-24 | article | SCLC-SBT: a spectral-bilinear transformer for small cell lung cancer histopathology image classification | https://doi.org/10.1038/s41598-026-76898-8
- 2026-10-25 | preprint | Birth, Death and Diff: Insertion-Based Sequence Generation with Corrections | https://doi.org/10.48550/arxiv.2610.34310
- 2026-10-25 | preprint | Accuracy Is Not Enough: An Efficiency- and Statistics-Aware Benchmark of ResNet50 and Vision Transformer for Explainable Multi-Class Gastrointestinal Diseases C | None
- 2026-10-26 | article | Optimization of Advertising Bidding Model With Generative AI and Deep Reinforcement Learning Collaboration: Design and Validation of Closed‐Loop Simulation Fram | https://doi.org/10.1111/exsy.70444
- 2026-10-26 | review | Systematic review of forensic craniofacial reconstruction. II. Integrating FCR with FSTT: Techniques and applications | https://doi.org/10.1016/j.fsir.2026.100540
- 2026-10-26 | preprint | Background-Robust Static Hand Gesture Recognition with Synthetic Scene Compositing: A Grouped Fivefold Ablation Across Five CNN Backbones | https://doi.org/10.21203/rs.3.rs-10086694/v1
- 2026-10-26 | preprint | Generative AI in Turbulence: Foundations, Applications, Validations and Benchmarks | https://doi.org/10.48550/arxiv.2610.35123
- 2026-10-27 | review | Generative Adversarial Networks for Omics Data: An Overview and Future Directions for Livestock Applications | https://doi.org/10.1002/age.70226
- 2026-10-27 | preprint | Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories | https://doi.org/10.48550/arxiv.2610.37369
- 2026-10-27 | preprint | ProtCell: a foundation model for single-cell proteomics | https://doi.org/10.64898/2026.10.22.760932
- 2026-10-28 | preprint | BASILISC: Bayesian Alignment of Siamese-Integrated Latents for Inferring Single-cell Correspondence | https://doi.org/10.48550/arxiv.2610.39050
- 2026-10-28 | preprint | A Tutorial on Generative Models for Music Audio | https://doi.org/10.48550/arxiv.2610.38252
- 2026-10-30 | preprint | Harness engineering enables a multi-agent LLM system to generate cross-domain hypotheses beyond single-investigator reach | https://doi.org/10.64898/2026.10.25.761203
- 2026-10-30 | preprint | Continual Reinforcement Learning Is an Integration Problem | https://doi.org/10.48550/arxiv.2610.41809
- 2026-10-30 | dissertation | Understanding and improving language models through interpretable decompositions | https://doi.org/10.14288/1.0457061
- 2026-10-30 | preprint | From WSIs to Bulk, Spatial and Single-Cell Transcriptomics via Multi-Scale Graph Neural Networks | https://doi.org/10.48550/arxiv.2610.41692
- 2026-11-01 | article | Unlocking the Potential of Machine Learning for Advanced Single‐Atom Catalysts Design in Promoting Electrochemical CO 2 Reduction | https://doi.org/10.1002/rar2.70688
- 2026-11-02 | preprint | Cellular context shapes the transcriptomic impact of somatic variants | https://doi.org/10.64898/2026.10.28.760887
- 2026-11-02 | preprint | Integrated bulk and single-cell transcriptomic analysis identifies BCHE and FOXM1 as oxidative-stress markers linked to prognosis and immune-microenvironment re | https://doi.org/10.21203/rs.3.rs-11297056/v1
- 2026-11-03 | preprint | What is shared across 33 cancer types is the tissue, not the tumour: a non-malignant axis of the microenvironment | https://doi.org/10.21203/rs.3.rs-11281904/v1
- 2026-11-03 | preprint | ARCADE disentangles cell abundance from continuous transcriptional state in spatial transcriptomics | https://doi.org/10.64898/2026.10.28.762703
- 2026-11-03 | preprint | LOTUS: lineage-aware optimal transport reconstructs continuous clonal fate dynamics | https://doi.org/10.64898/2026.10.28.762359
- 2026-11-04 | article | Single-cell analyses identify persistent neuroimmune remodeling and prioritize SPP1 and LGALS3 in neuropathic pain | https://doi.org/10.3389/fimmu.2026.1981038

## Europe PMC records (first publication date, title, DOI)
- 2026-10-01 | Regulatory-prior-guided attention preserves biological structure during unpaired single-cell RNA-ATAC integration. | 10.1093/bioinformatics/btag696
- 2026-10-09 | scAICT: Adversarial Integration and Contrastive Translation for Unpaired scRNA-Seq and scATAC-Seq. | 10.3390/biology15201797
- 2026-10-16 | GAN-enhanced digital twins for biomarker-guided cancer immunotherapy and adoptive cell therapy: a mini review | None
- 2026-10-09 | AI-driven multi-omics integration in breast cancer: clinical applications, immunotherapy prediction, and translational challenges | None
- 2026-10-01 | Characterizing intra- and intertumor heterogeneity in ovarian high-grade serous carcinoma subtypes using single-cell and spatial transcriptomics. | 10.1101/gr.281433.125
- 2026-10-13 | Artificial intelligence and anti-cancer drugs: advancing discovery, personalization, and precision oncology. | 10.3389/fonc.2026.1885312
- 2026-10-01 | Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework. | 10.1101/gr.281431.125
- 2026-10-07 | Dynamics of human cardiogenesis and its disruption in trisomy 21. | 10.1038/s41586-026-11125-y
- 2026-10-19 | A Cross-Condition Single-Cell Atlas Defines Skeletal Muscle Adaptation and Its Predictive Limits. | 10.3390/cimb48101080
- 2026-10-06 | AI-guided ethnopharmacology for cardiovascular drug discovery: from biomedical data to experimental validation | None
- 2026-10-26 | Artificial Intelligence for Multi-omics Breast Cancer Subtype Prediction and Discovery: A Systematic Review. | 10.34133/csbj.0243
- 2026-10-09 | Artificial intelligence-driven therapeutics for disease modification in type 1 diabetes: a digital public health and clinical translation framework | None
- 2026-10-07 | The impact of deep learning and omics data in transforming precision therapy for brain cancer | None
- 2026-10-27 | Circulating Biomarkers with Artificial Intelligence for Clinical Risk Assessment and Therapeutic Monitoring: Evidence, Limitations, and Translation. | 10.2147/tcrm.s612081
- 2026-10-12 | Intelligent immunotherapy: AI-driven optimization of CAR-T therapy in hematologic malignancies. | 10.1016/j.isci.2026.117255
- 2026-10-09 | ESICM LIVES 2026. | 10.1186/s40635-026-00962-w
