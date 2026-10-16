# coverage_lines.md -- nine works for Related work (2026-10-15)

Full text read for 6 of 9 (Beaconet, AIF, ABC, IMGG, Li et al., GLUE). Abstract only for 3: iDLC (bioRxiv HTTP 429), scStarCorrect (IEEE, closed), Portal (Nature Computational Science, subscription). No block was circumvented. Metadata of all nine checked against Crossref on 2026-10-15; Crossref `updates:` filter: no correction or retraction notice for any (AIF carries a self-referencing 'new_version' record only).

## Summary

| # | Key | Work | Text | Paragraph |
|---|---|---|---|---|
| 1 | jiang2026geometric | iDLC | abstract_only | Kernel and optimal-transport alignment |
| 2 | gu2025scstarcorrect | scStarCorrect | abstract_only | Adversarial (JS-type) integration (provisional: the abstract does not state the adversarial loss form) |
| 3 | xu2024beaconet | Beaconet | full_text | Wasserstein critics |
| 4 | zhao2022adversarial | Portal | abstract_only | Adversarial (JS-type) integration (provisional: the abstract does not state the adversarial loss form) |
| 5 | monnier2024novel | AIF | full_text | Adversarial (JS-type) integration |
| 6 | danino2024batch | ABC | full_text | Adversarial (JS-type) integration |
| 7 | wang2022imgg | IMGG | full_text | Wasserstein critics |
| 8 | li2022trade | Li et al. | full_text | Kernel and optimal-transport alignment |
| 9 | cao2022multi | GLUE | full_text | Adversarial (JS-type) integration |

## 1. iDLC -- `jiang2026geometric` (DOI 10.64898/2026.02.17.706490)

**Text access:** abstract_only  
**Paragraph:** Kernel and optimal-transport alignment

**Outline line** (description part: 25 words):

```latex
\item iDLC: separates biological and technical components in a latent space, then aligns batches with an adversarial term regularised by optimal transport, guided by mutual-nearest-neighbour cell pairs \citep{jiang2026geometric}.
```

**BibTeX:**

```bibtex
@article{jiang2026geometric,
  author = {Jiang, Chenghan and Zheng, Ren and Ji, Yongle and Cao, Shubin and Fang, Yanhua and Wang, Zhe and Wang, Ruoyu and Liang, Shanshan and Tao, Shuai},
  title = {Geometric-aware and interpretable deep learning for single-cell batch correction via explicit disentanglement and optimal transport},
  journal = {bioRxiv},
  note = {Preprint, posted 2026-02-20},
  year = {2026},
  doi = {10.64898/2026.02.17.706490},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- jiang2026geometric -- iDLC: explicit separation of biological and technical components in a latent space; alignment by optimal-transport-regularised adversarial term, guided by high-confidence mutual-nearest-neighbour pairs -- checked: https://doi.org/10.64898/2026.02.17.706490; abstract only (bioRxiv full text not obtained: the DOI resolver and Semantic Scholar returned HTTP 429 on two attempts; no mirror tried); text: abstract returned by the fetch tool, metadata from Crossref (posted 2026-02-20). Not stated in the abstract: the adversary's divergence, any reference batch, the backbone.
```

**Locators:** Abstract only (sentences 3-4: 'dual-level correction through explicit feature disentanglement and optimal transport-regularized adversarial alignment'; 'high-confidence mutual nearest neighbor pairs'). No Methods locator available.

**Notes:** Full text refused (HTTP 429) - not circumvented. Line limited to abstract claims. Adversary type, reference-batch use and backbone are unknown. Placement is provisional: could equally sit in 'Adversarial (JS-type) integration' once the adversary is known. Suggest wrapping in \pending{full text of iDLC}{when bioRxiv answers}.

## 2. scStarCorrect -- `gu2025scstarcorrect` (DOI 10.1109/bibm66473.2025.11356167)

**Text access:** abstract_only  
**Paragraph:** Adversarial (JS-type) integration (provisional: the abstract does not state the adversarial loss form)

**Outline line** (description part: 26 words):

```latex
\item scStarCorrect: a StarGAN-based model combining a batch-conditioned variational autoencoder with a domain-aware discriminator that aligns every batch to one fixed reference batch, so correction runs one way \citep{gu2025scstarcorrect}.
```

**BibTeX:**

```bibtex
@inproceedings{gu2025scstarcorrect,
  author = {Gu, Weizhen and Zhang, Han and Han, Mingjing},
  title = {{scStarCorrect:} A {StarGAN}-Based Adversarial Model for Reference-Guided Batch Correction in Single-Cell {RNA-Seq} Data},
  booktitle = {2025 IEEE International Conference on Bioinformatics and Biomedicine ({BIBM})},
  pages = {921--926},
  year = {2025},
  publisher = {IEEE},
  doi = {10.1109/bibm66473.2025.11356167},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- gu2025scstarcorrect -- scStarCorrect: StarGAN-based; shared variational autoencoder conditioned on batch labels; domain-aware discriminator enforcing alignment to a fixed reference batch (one-way correction); loss terms: reconstruction, adversarial, domain classification, variational regularisation -- checked: https://doi.org/10.1109/bibm66473.2025.11356167; abstract only (IEEE full text closed: Unpaywall no OA location, Semantic Scholar no openAccessPdf, DOI resolver HTTP 202); text: abstract returned by the fetch tool, metadata from Crossref (BIBM 2025, pp. 921-926). Not stated in the abstract: the form of the adversarial loss (cross-entropy or Wasserstein).
```

**Locators:** Abstract only (sentences 3-5: 'shared variational autoencoder conditioned on batch labels'; 'domain-aware discriminator that enforces alignment to a fixed reference batch'; 'one-way batch correction'; 'jointly optimize reconstruction loss, adversarial loss, domain classification loss, and variational regularization').

**Notes:** Paywalled (IEEE). Whether the adversarial loss is JS-type or Wasserstein is not in the abstract, so the paragraph is provisional; if the full text shows a Wasserstein loss it belongs in 'Wasserstein critics'. Reference-anchored design matches the reference critic in our study, but the line makes no comparison. Suggest \pending marker.

## 3. Beaconet -- `xu2024beaconet` (DOI 10.1002/advs.202306770)

**Text access:** full_text  
**Paragraph:** Wasserstein critics

**Outline line** (description part: 26 words):

```latex
\item Beaconet (Batch Effect Adversarial Correction Network): corrects each batch in gene-expression space with a corrector network, trained against one Wasserstein-distance critic per batch with a gradient penalty; no reference batch is chosen \citep{xu2024beaconet}.
```

**BibTeX:**

```bibtex
@article{xu2024beaconet,
  author = {Xu, Han and Ye, Yusen and Duan, Ran and Gao, Yong and Hu, Yuxuan and Gao, Lin},
  title = {{Beaconet:} A Reference-Free Method for Integrating Multiple Batches of Single-Cell Transcriptomic Data in Original Molecular Space},
  journal = {Advanced Science},
  volume = {11},
  number = {26},
  pages = {2306770},
  year = {2024},
  doi = {10.1002/advs.202306770},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- xu2024beaconet -- Beaconet: corrector network in the original gene-expression space on log-scaled data; one auxiliary network W_j per batch estimates the Wasserstein distance (Kantorovich-Rubinstein dual, k-Lipschitz) between batch j and the mixture of the other batches; gradient penalty (lambda = 10, after Gulrajani et al.); reference-free, no cell-type labels or cell-pair matching -- checked: https://doi.org/10.1002/advs.202306770; abstract; Introduction (last paragraph); Experimental Section: 'Problem Formulation', 'Wasserstein distance' (Eqs. 1-2), 'Penalty term for 1-lipschitz condition', 'Algorithm of Beaconet', 'Objective Function' (Eqs. 3-8); text: PMC full text (PMC11234410), no page numbers (section and equation locators)
```

**Locators:** PMC11234410 full text (no page numbers). Abstract ('reference-free'; 'original molecular space'; 'adversarial correction network'). Introduction, last paragraph (no cell-type labels or cell-pair matching). Experimental Section: Problem Formulation (log-scaled expression); 'Wasserstein distance' (Eq. 1 W1 with l1 cost; Eq. 2 dual with 1-Lipschitz W); 'Penalty term for 1-lipschitz condition'; 'Algorithm of Beaconet' (corrector C + encoder-L networks W_i; 'does not impose any prior bias on the importance of different batches'); 'Objective Function' (Eq. 3 two-batch WGAN-GP form, lambda=10; Eqs. 5-8 one critic W_j per batch j against the mixture of the other batches; Eq. 8 final objective).

**Notes:** The abstract-level candidate description said only 'adversarial'; the full text shows a Wasserstein critic with gradient penalty, so the work belongs to 'Wasserstein critics', not the JS-type paragraph. Backbone is not a VAE or autoencoder: a corrector network applied to expression values. Beaconet's text also says iMAP uses a WGAN with reference-based training (Experimental Section).

## 4. Portal -- `zhao2022adversarial` (DOI 10.1038/s43588-022-00251-y)

**Text access:** abstract_only  
**Paragraph:** Adversarial (JS-type) integration (provisional: the abstract does not state the adversarial loss form)

**Outline line** (description part: 27 words):

```latex
\item Portal: a unified framework of adversarial domain translation that learns harmonised representations of datasets, applied across samples, platforms and data types and to integration of millions of cells \citep{zhao2022adversarial}.
```

**BibTeX:**

```bibtex
@article{zhao2022adversarial,
  author = {Zhao, Jia and Wang, Gefei and Ming, Jingsi and Lin, Zhixiang and Wang, Yang and {The Tabula Microcebus Consortium} and Agarwal, Snigdha and Agrawal, Aditi and Al-Moujahed, Ahmad and Alam, Alina and Albertelli, Megan A. and Allegakoen, Paul and Ambrosi, Thomas and Antony, Jane and Artandi, Steven and Aujard, Fabienne and Awayan, Kyle and Baghel, Ankit and Bakerman, Isaac and Bakken, Trygve E. and Baruni, Jalal and Beachy, Philip and Bilen, Biter and Botvinnik, Olga and Boyd, Scott D. and Burhan, Deviana and Casey, Kerriann M. and Chan, Charles and Chang, Charles A. and Chang, Stephen and Chen, Ming and Clarke, Michael F. and Crasta, Sheela and Culver, Rebecca and D'Addabbo, Jessica and Darmanis, Spyros and Dehghannasiri, Roozbeh and Ding, Song-Lin and Duffy, Connor V. and Epelbaum, Jacques and Espinoza, F. Hern{\'a}n and Ezran, Camille and Farup, Jean and Ferrell Jr, James E. and Frank, Hannah K. and Fuller, Margaret and Gillich, Astrid and Godoy, Elias and Gratzinger, Dita and Guethlein, Lisbeth A. and Hang, Yan and Hasegawa, Kazuteru and Hodge, Rebecca D. and Hoover, Malachia and Huang, Franklin W. and Huang, Kerwyn Casey and Huynh, Shelly and Isobe, Taichi and Israel, Carly and Jang, SoRi and Jing, Qiuyu and Jones, Robert C. and Kang, Jengmin and Karanewsky, Caitlin J. and Karkanias, Jim and Kebschull, Justus and Kershner, Aaron and Kim, Lily and Kim, Seung K. and Kirk, E. Christopher and Koh, Winston and Konermann, Silvana and Kong, William and Krasnow, Mark A. and Kuo, Christin and Lautier, Corinne and Lee, Song Eun and Lein, Ed S. and Lewis, Rebecca and Li, Peng and Lin, Shengda and Liu, Shixuan and Liu, Yin and Loeb, Gabriel and Long, Jonathan Z. and Lu, Wan-Jin and Lucot, Katherine and Luo, Liqun and McGeever, Aaron and Metzger, Ross and Ming, Jingsi and Montine, Tom and de Morree, Antoine and Morri, Maurizio and Mrouj, Karim and Mukherjee, Shravani and Nabhan, Ahmad and Nafees, Saba and Neff, Norma and Neuh{\"o}fer, Patrick and Nguyen, Patricia and Okamoto, Jennifer and Olivieri, Julia and Ouadah, Youcef and Paine, Honor and Parham, Peter and Pendleton, Jozeph L. and Penland, Lolita and Perret, Martine and Pisco, Angela Oliveira and Qi, Zhen and Quake, Stephen R. and Radespiel, Ute and Rando, Thomas A. and Ravelonjanahary, Hajanirina No{\"e}line and Razafindrakoto, Andriamahery and Salzman, Julia and Schaum, Nicholas and Schopler, Robert and Scott, Bronwyn and Shapiro, Liza and Sin, Hosu and Sinha, Rahul and Sit, Rene and Stanley, Geoff and Stryer, Lubert and Subramaniam, Varun Ramanan and Swarup, Aditi and Tan, Weilun and Tarashansky, Alexander and Taychameekiatchai, Aris and Terrien, J{\'e}r{\'e}my and Travaglini, Kyle J. and Urtasun, Andoni and {Sivakamasundari} and Veerakumar, Avin and Vemuri, Venkata Naga Pranathi and Verdier, Jean-Michel and De Vlaminck, Iwijn and Vollrath, Douglas and Wang, Bo and Wang, Bruce and Wang, Gefei and Wang, Michael F. Z. and Wang, Sheng and Webber, James and Weinstein, Hannah and Weissman, Irving L. and Wiggenhorn, Amanda L. and Williams, Cathy V. and Wright, Patricia and Wu, Albert Y. and Wu, Angela Ruohao and Wyss-Coray, Tony and Xiang, Bao and Yan, Jia and Yang, Can and Yang, Jinxurong and Yoder, Anne D. and Yu, Brian and Yung, Andrea R. and Zhang, Yue and Zhao, Jia and Zhao, Zicheng and Wu, Angela Ruohao and Yang, Can},
  title = {Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets},
  journal = {Nature Computational Science},
  volume = {2},
  number = {5},
  pages = {317--330},
  year = {2022},
  doi = {10.1038/s43588-022-00251-y},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- zhao2022adversarial -- Portal: unified framework of adversarial domain translation learning harmonised representations; integrates datasets across samples, platforms and data types; scales to millions of cells -- checked: https://doi.org/10.1038/s43588-022-00251-y; abstract only (publisher full text is subscription: Europe PMC marks 'Subscription required', Unpaywall returned no OA location, fetch tool found no readable copy); text: abstract from Europe PMC record, metadata from Crossref (Nat Comput Sci 2:317-330). Not stated in the abstract: the adversarial loss, any reference batch, the backbone.
```

**Locators:** Abstract only ('unified framework of adversarial domain translation to learn harmonized representations of datasets'; 'integrating datasets across different samples, platforms and data types'; 'integration of millions of cells, in minutes').

**Notes:** Paywalled. Adversary type, reference-batch use and backbone are unknown, so placement is provisional. Crossref lists 166 authors, including the collective author 'The Tabula Microcebus Consortium' and one single-name author (Sivakamasundari); the BibTeX keeps Crossref order. A bioRxiv preprint exists (Crossref relation has-preprint) but is a different version and was not read. Suggest \pending marker.

## 5. AIF -- `monnier2024novel` (DOI 10.1371/journal.pcbi.1011880)

**Text access:** full_text  
**Paragraph:** Adversarial (JS-type) integration

**Outline line** (description part: 31 words):

```latex
\item AIF (Adversarial Information Factorization): a conditional variational autoencoder on gene expression whose encoder is trained against an auxiliary cross-entropy classifier of batch from the latent code; cells are then decoded under one chosen batch label \citep{monnier2024novel}.
```

**BibTeX:**

```bibtex
@article{monnier2024novel,
  author = {Monnier, Lily and Courn{\`e}de, Paul-Henry},
  title = {A novel batch-effect correction method for {scRNA-seq} data based on {Adversarial Information Factorization}},
  journal = {PLOS Computational Biology},
  volume = {20},
  number = {2},
  pages = {e1011880},
  year = {2024},
  doi = {10.1371/journal.pcbi.1011880},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- monnier2024novel -- AIF: conditional VAE on gene expression with batch-label conditioning (MSE reconstruction, KL, cross-entropy batch classification); real/fake GAN discriminator (cross-entropy); auxiliary network predicts batch from the latent code (cross-entropy L_aux) and enters the encoder objective adversarially; at correction all cells are decoded under one batch label, the study using the most represented batch -- checked: https://doi.org/10.1371/journal.pcbi.1011880; abstract; p3 (Methods, 'Model presentation', Fig 2), p4 (loss list; discriminator C_chi; auxiliary network A_psi), p5 ('Batch effect correction'); text: PDF in PLOS Comput Biol layout (page 3 footer 'February 22, 2024 3 / 22'), retrieved through Unpaywall (HAL-hosted copy)
```

**Locators:** PDF p3 (Methods 'Model presentation': CVAE backbone on gene expression x with batch label y; Gaussian latent; Fig 2). PDF p4 (loss list: reconstruction MSE, KL, cross-entropy batch classification; real/fake discriminator with cross-entropy; auxiliary network A_psi with cross-entropy L_aux 'adversarially incorporated into the encoder's objective'). PDF p5 ('Batch effect correction': decode all cells under one batch label, 'any batch label or a combination'; study used the most represented batch).

**Notes:** Alignment in the latent space needs no reference batch; the decoding step does choose one batch label (most represented batch in the study). Crossref shows one 'new_version' record dated 2024-03-05 that points to the article's own DOI; no correction or retraction notice (updates: filter returns only that self-record).

## 6. ABC -- `danino2024batch` (DOI 10.1093/bioadv/vbad186)

**Text access:** full_text  
**Paragraph:** Adversarial (JS-type) integration

**Outline line** (description part: 27 words):

```latex
\item ABC (Autoencoder-based Batch Correction): an autoencoder trained against a multi-class batch discriminator with categorical cross-entropy, plus cell-type classifier branches that use labels to retain biological signal; no reference batch is designated \citep{danino2024batch}.
```

**BibTeX:**

```bibtex
@article{danino2024batch,
  author = {Danino, Reut and Nachman, Iftach and Sharan, Roded},
  title = {Batch correction of single-cell sequencing data via an autoencoder architecture},
  journal = {Bioinformatics Advances},
  volume = {4},
  number = {1},
  pages = {vbad186},
  year = {2024},
  doi = {10.1093/bioadv/vbad186},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- danino2024batch -- ABC: autoencoder (encoder E, decoder DC) on normalised gene counts, latent and output cell-type classifiers, batch discriminator DI = multi-class batch classifier with categorical cross-entropy (Eq. 1), autoencoder trained against it with L_ADV = -1/n sum log(1 - DI(.)_c) (Eq. 2), MSE reconstruction (Eq. 3); the discriminator predicts one of C batches, with no reference-batch designation (the word 'reference' occurs only in the 'References' heading); no variational term -- checked: https://doi.org/10.1093/bioadv/vbad186; abstract; Introduction (last paragraphs); Sec. 2.1 'The model and training process' (Eqs. 1-3, Fig. 1); text: PMC full text, no page numbers
```

**Locators:** PMC full text (no page numbers). Sec. 1 Introduction, last paragraphs ('autoencoder architecture trained in an adversarial manner alongside a batch label discriminator'; 'multi-class batch classifier as a discriminator ... informed by cell type information'). Sec. 2.1 'The model and training process': Eq. 1 (discriminator categorical cross-entropy over C batches), Eq. 2 (adversarial loss), Eq. 3 (MSE + two cell-type classifier terms), Fig. 1 caption (three training stages).

**Notes:** 'No reference batch is designated' rests on the Methods text naming none (the word 'reference' occurs only in the 'References' heading); it is a statement about the description, not a test. Year: Crossref online 2023-12-28, volume 4 issue 1 print 2024-01-05; the BibTeX uses 2024 (print volume year, as luecken2022benchmarking does).

## 7. IMGG -- `wang2022imgg` (DOI 10.3390/ijms23042082)

**Text access:** full_text  
**Paragraph:** Wasserstein critics

**Outline line** (description part: 32 words):

```latex
\item IMGG (integrating multiple single-cell datasets through connected graphs and generative adversarial networks): a generator on highly variable genes trained with a Wasserstein GAN with gradient penalty (WGAN-GP) critic towards a synthetic intermediate target built from similar-cell cohorts matched across batches by mutual nearest neighbours \citep{wang2022imgg}.
```

**BibTeX:**

```bibtex
@article{wang2022imgg,
  author = {Wang, Xun and Zhang, Chaogang and Zhang, Ying and Meng, Xiangyu and Zhang, Zhiyuan and Shi, Xin and Song, Tao},
  title = {{IMGG:} Integrating Multiple Single-Cell Datasets through Connected Graphs and Generative Adversarial Networks},
  journal = {International Journal of Molecular Sciences},
  volume = {23},
  number = {4},
  pages = {2082},
  year = {2022},
  doi = {10.3390/ijms23042082},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- wang2022imgg -- IMGG: HVG selection, PCA, mutual-nearest-neighbour search to build cross-batch connected graphs of similar cells; residual fully connected generator in the HVG space; adversarial loss 'optimized via the WGAN-GP' (critic loss E[D(fake)] - E[D(real)] + gradient penalty, Eq. 1; generator loss -E[D(fake)], Eq. 2) plus an L2 reconstruction loss (Eq. 3); target = intermediate batch synthesised per cohort (Mean, Max or Min of the cohort) instead of an anchor batch -- checked: https://doi.org/10.3390/ijms23042082; abstract; Introduction; Sec. 4 (opening paragraph), Sec. 4.3, Sec. 4.4 (Eqs. 1-3); text: PMC full text, no page numbers
```

**Locators:** PMC full text (no page numbers). Abstract and Sec. 1 (anchor-batch vs low-dimensional-space methods). Sec. 4 opening paragraph (three stages: HVGs + PCA; MNN graphs; GAN). Sec. 4.3 'Correcting Batch Effects by GAN' (intermediate batch from the Mean, Max or Min pattern over a cohort of similar cells; all batches transformed to it simultaneously). Sec. 4.4 'Model Details' (residual fully connected generator; 'optimized via the WGAN-GP'; Eqs. 1-3; reconstruction loss added because WGAN-GP alone was hard to fit, Appendix A.4).

**Notes:** The abstract-level candidate description said 'GAN'; the full text shows a WGAN-GP critic, so it moves from the JS-type paragraph to 'Wasserstein critics'. The word 'Wasserstein' occurs only as WGAN-GP in Sec. 4.4 and in reference 27's title. Target is not a fixed reference batch: it is a synthesised intermediate target per cohort. Sibling of IMAAE (wang2023integrating, already in refs.bib; shares two authors).

## 8. Li et al. -- `li2022trade` (DOI 10.1186/s12859-022-05003-3)

**Text access:** full_text  
**Paragraph:** Kernel and optimal-transport alignment

**Outline line** (description part: 34 words):

```latex
\item Li et al.: learns the whole trade-off between scVI's loss (negative evidence lower bound) and a latent-space batch measure, either a neural-network mutual-information estimate (MINE, Mutual Information Neural Estimation) or maximum mean discrepancy, by Pareto multi-task learning \citep{li2022trade}.
```

**BibTeX:**

```bibtex
@article{li2022trade,
  author = {Li, Hui and McCarthy, Davis J. and Shim, Heejung and Wei, Susan},
  title = {Trade-off between conservation of biological variation and batch effect removal in deep generative modeling for single-cell transcriptomics},
  journal = {BMC Bioinformatics},
  volume = {23},
  number = {1},
  pages = {460},
  year = {2022},
  doi = {10.1186/s12859-022-05003-3},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- li2022trade -- scVI (preliminary version 0.3.0) with bi-objective loss (U_n = negative ELBO, V_n = batch-effect measure on the latent z); Pareto front by Pareto multi-task learning versus scalarisation (K = 10 subproblems + 2 extreme points = 12 candidates); V_n = MINE estimate of mutual information between z and batch (neural network trained adversarially) or MMD between z|s=0 and z|s=1; surrogate scores for the candidates: batch entropy, ASW, ARI, NMI, UCA -- checked: https://doi.org/10.1186/s12859-022-05003-3; abstract; p3 (Controlling batch effect; footnote 3, scVI 0.3.0), p4 (Eqs. 2-3), p5-6 (Overview of methods; U_n = -ELBO; standardisation; 12 candidates), p6 and p19 (MINE needs adversarial training), p9 (surrogate metrics), p16-17 (Methods: MINE Eq. 8, MMD Eq. 9); text: publisher PDF (gold OA, Unpaywall)
```

**Locators:** PDF p1 (abstract). p3 footnote 3 (scVI version 0.3.0). p3-4 'Controlling batch effect' (Eq. 2 bi-objective, Eq. 3 penalised form). p5-6 'Overview of methods' ('Throughout this paper we always employ U_n = -ELBO', p6; standardisation of U_n and V_n, p6; K = 10 subproblems, 12 candidates, p6). p6 and p19 (MINE has its own parameters trained adversarially; Alg. 3). p9 (surrogate metrics: batch entropy, ASW, ARI, NMI, UCA). p16 Methods 'MINE for measuring batch effect V_n' (I(z, s); Eq. 8 on p17). p17 (MMD between z|s=0 and z|s=1, Eq. 9).

**Notes:** Batch removal is measured on the latent z of the scVI encoder, used both as the training penalty V_n (MINE or MMD) and as the plotted axis of the Pareto front (Fig. 2A, p7); BE, ASW, ARI, NMI and UCA are reported only as surrogate scores of selected candidates (p9). Placement is a judgement: MMD is one of the two batch measures; 'Benchmarks' would fit if that paragraph covers trade-off analysis.

## 9. GLUE -- `cao2022multi` (DOI 10.1038/s41587-022-01284-4)

**Text access:** full_text  
**Paragraph:** Adversarial (JS-type) integration

**Outline line** (description part: 26 words):

```latex
\item GLUE (graph-linked unified embedding): omics-specific variational autoencoders linked through a guidance graph of regulatory interactions, with a discriminator that predicts each cell's omics layer and is opposed by the encoders \citep{cao2022multi}.
```

**BibTeX:**

```bibtex
@article{cao2022multi,
  author = {Cao, Zhi-Jie and Gao, Ge},
  title = {Multi-omics single-cell data integration and regulatory inference with graph-linked embedding},
  journal = {Nature Biotechnology},
  volume = {40},
  number = {10},
  pages = {1458--1466},
  year = {2022},
  doi = {10.1038/s41587-022-01284-4},
  verified = {2026-10-15 via doi}
}
```

**evidence.md line:**

```
- cao2022multi -- GLUE: omics-specific variational autoencoders with a shared embedding dimension; guidance graph of regulatory interactions embedded by a graph variational autoencoder and used in the decoders; omics discriminator D (K-way softmax) trained with multiclass cross-entropy (Eq. 18) and opposed by the encoders; canonical adversarial alignment 'amounts to minimizing a generalized form of Jensen-Shannon divergence' (Eq. 21); within-omics batch effect handled by batch as a decoder covariate; gloss 'graph-linked embedding method' matches the title ('graph-linked embedding') and abstract ('graph-linked unified embedding') but omits the adversarial alignment -- checked: https://doi.org/10.1038/s41587-022-01284-4; abstract; PDF p2 (Fig. 1 caption and text), p3 (batch correction, Results), p11 (Methods: Eqs. 18-21, 'Batch effect correction'); text: publisher PDF (hybrid OA, Unpaywall)
```

**Locators:** PDF p1 (abstract: 'graph-linked unified embedding'; title). PDF p2 (Fig. 1 caption: 'an omics discriminator D is used to align the cell embeddings of different omics layers via adversarial learning'; text: omics-specific variational autoencoders, guidance graph, graph variational autoencoder). PDF p3 (Results: batch correction by including batch as a decoder covariate). PDF p11 (Methods: Eq. 18 multiclass cross-entropy discriminator; Eqs. 19-20 training objective; 'Weighted adversarial alignment', Eq. 21 generalised JS divergence; 'Batch effect correction', Eqs. 25-27).

**Notes:** Gloss check: the existing Introduction phrase 'a graph-linked embedding method' is accurate against the title and abstract but does not say it is a multi-omics method or that it aligns omics layers adversarially (JS-type, stated by the authors on PDF p11). Suggested replacement gloss: 'a graph-linked embedding method for multi-omics integration'. Key cao2022multi does not collide with cao2022unified. PDF p10-11 are Methods pages without journal page numbers; the main text spans journal pp. 1458-1466 (Crossref).

## Deviations

- iDLC, scStarCorrect, Portal: abstract only (HTTP 429; IEEE closed; Nature Comput Sci subscription). Lines restricted to abstract statements; adversary type, reference-batch use and backbone are unknown for these three.
- Paragraph assignment for scStarCorrect and Portal is provisional because the abstracts do not name the adversarial loss.
