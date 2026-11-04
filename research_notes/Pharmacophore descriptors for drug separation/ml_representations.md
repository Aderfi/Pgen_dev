# Pharmacophore information as ML / GNN input: representations, design choices, and reported gains

Scope note: this covers (a) pharmacophore-aware GNN architectures and their encodings, (b) fixed
pharmacophore fingerprint + GNN fusion and what ablations showed, (c) representation-level design
choices (binary vs count vs fuzzy, distance binning, scaling, dimensionality, folding), and (d)
evidence from drug–drug interaction / drug response / ADR prediction, including cold-start splits.
Where the literature does not answer a question, it is listed under Gaps rather than guessed.

---

## Q1: Pharmacophore-aware GNN architectures and how they encode pharmacophores

### Takeaway

Three encoding families dominate: (1) **heterogeneous graphs** that keep the atom graph and add
fragment/pharmacophore nodes plus atom↔fragment "junction" edges (PharmHGT, HiGNN); (2) **reduced
graphs** where atoms are pooled into typed pharmacophore nodes and a second message-passing stage
runs on that coarse graph (RG-MPNN); (3) **fully-connected pharmacophore hypothesis graphs** where
nodes are features and edge labels are shortest-path or spatial distances (PGMG). All of them keep
the atom graph in the loop — none replaces it — and the reported gains over atom-graph-only
baselines are real but small (roughly +1 to +4 ROC-AUC points, 0.01–0.3 RMSE).

### Cited Findings

**PharmHGT (Jiang et al., Communications Chemistry 6:60, 2023)**

- Builds a heterogeneous molecular graph with **2 node types (atoms, pharmacophores) and 3 edge types (bonds, BRICS reaction types, atom–pharmacophore junctions)**; molecules are fragmented with BRICS and the inter-fragment reaction type is retained as the edge label — [Nature Comm Chem](https://www.nature.com/articles/s42004-023-00857-x); [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Three explicit *views*: **atom-level** (atoms + directed bonds), **pharm-level** (pharmacophore nodes + BRICS reaction edges), **junction-level** (both node types + junction edges) — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Each view runs multi-view attention message passing with residual updates; a **GRU readout** gives one vector per view (Z_α, Z_β, Z_γ), fused by sequential read-out attention: `Z_γβ = Attn(Z_γ, Z_β)`, then `Z = Attn(Z_α, Z_γβ)` — i.e. the atom view is the *query* and the pharmacophore/junction views are the *context*, not a plain concatenation — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Classification ROC-AUC % (PharmHGT vs best baseline, 0.8/0.1/0.1 split, 5-fold CV, 5 repeats): BBBP 95.4±1.15 vs GROVER-large 94.0; BACE 86.5±2.21 vs GROVER-base 82.6; ClinTox 94.5±0.42 vs GROVER-large 94.4; Tox21 83.9±0.56 vs 83.1; SIDER 66.9±1.63 vs 65.8; HIV 80.6±0.21 vs DMPNN 78.6 — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Regression RMSE (random split): ESOL 0.680 vs CoMPT 0.774; FreeSolv 1.266 vs FragGAT 1.538; Lipophilicity 0.583 vs CoMPT 0.592. Scaffold split: ESOL 0.839 vs CMPNN 0.845; FreeSolv 1.689±0.516 vs CMPNN 1.833; Lipo 0.638 vs CoMPT 0.646 — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- **Ablation (7 variants on BBBP + ESOL):** atom-only (α), pharm-only (β), junction-only (γ), and four fusions. The full model is best; the paper states the **pharm-only variant "performs the worst"**; among the reduced variants γ→α (junction then atom) is best. Numbers appear only in Fig. 3, not in text — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Internal inconsistency to flag: the abstract claims "up to 1.55% higher ROC-AUC", but Table 1 differences run up to +3.9 (BACE); dataset sizes and one standard deviation also disagree between text and tables — [PMC10070395](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070395/)
- Peer review published with the paper notes the atom↔pharmacophore heterogeneous graph "seems new" but that the architecture description was confusing — [Nature Comm Chem](https://www.nature.com/articles/s42004-023-00857-x)
- Code: [github.com/stardj/PharmHGT](https://github.com/stardj/PharmHGT) (also listed under mindrank-ai/PharmHGT)
- A later AAAI paper (PharmaQA) characterizes PharmHGT as "introducing pharmacophore constraints but relying on static representations" — [search summary of AAAI listing](https://ojs.aaai.org/index.php/AAAI/article/view/39037/42999)

**RG-MPNN (Kong et al., "Integrating concept of pharmacophore with graph neural networks for chemical property prediction and interpretation", J. Cheminformatics 14:52, 2022)**

- Uses **Harper's graph reduction scheme: 18 pharmacophore node types** = 3 ring classes (aromatic, aliphatic, acyclic) × 6 feature classes (positive ionizable, negative ionizable, joint H-bond donor/acceptor, donor, acceptor, no feature). Ring and feature classes are prioritized; atoms outside any pharmacophore need extra handling because classical rules leave them unlabeled — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/); DOI [10.1186/s13321-022-00634-3](https://doi.org/10.1186/s13321-022-00634-3)
- **RG pooling:** each RG node's initial state is the **sum of its child atom vectors**; it then receives **attention-weighted messages from its child atoms** and is updated by a **GRU**. Molecule-level readout treats the whole graph as one pharmacophore node (again summed). **RG edge weight = number of bonds shared between adjacent pharmacophore nodes + 1** — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)
- Two-stage architecture: atom-level GAT-style attention message passing with residual skips (k = 2–5 steps) → graph reduction → RG-level attention + GRU message passing (t = 1–3 steps) → MLP head — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)
- Documented limitation: molecules that collapse into a **single pharmacophore node cannot pass any RG-level information** — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)
- Benchmark results (AttentiveFP / MPNN / RG-MPNN): ESOL RMSE 0.650 / 0.853 / **0.605**; FreeSolv 1.162 / 1.255 / **0.939**; Lipo 0.627 / 0.662 / **0.579**; MUV AUC 0.772 / 0.740 / **0.819**; HIV 0.815 / 0.803 / **0.824**; BACE 0.868 / 0.846 / **0.889**; BBBP **0.888** / 0.824 / 0.879; Tox21 0.852 / 0.836 / **0.873**; ToxCast 0.860 / 0.848 / **0.866**; SIDER **0.827** / 0.812 / 0.825; ClinTox 0.940 / 0.941 / **0.965** — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)
- Kinase datasets (10 ChEMBL sets, 807–8,800 molecules, 1000 nM threshold): RG-MPNN best AUC on **7 of 10 random splits**; e.g. EGFR 0.942 vs MPNN 0.923 / AttentiveFP 0.933; CK1 0.800 vs 0.673 / 0.751 — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)
- **RG-pooling ablation (the directly relevant number):** adding RG pooling improved **all ResMPNNs, 80% of AttentiveFP models, and only 55% of MPNNs**, with **AUC changes spanning −0.010 to +0.046**. ResMPNN gained *more on scaffold splits than on random splits*, but the authors say this trend was inconsistent for the other two backbones — [PMC9351086](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351086/)

**HiGNN (Zhu et al., J. Chem. Inf. Model. 2023; arXiv 2208.13994)**

- Learns jointly from the molecular graph and **BRICS fragments** ("exploiting both molecular graphs and fragments information simultaneously"), motivated by the claim that graph methods "neglect the hierarchical information of molecules and the relationships between feature channels" — [arXiv 2208.13994](https://arxiv.org/abs/2208.13994)
- Adds a **plug-in feature-wise attention block** that "adaptively recalibrates atomic features after the message passing phase" — i.e. channel attention, the same mechanism family as SE blocks — [arXiv 2208.13994](https://arxiv.org/abs/2208.13994)
- Proposes a molecule–fragment similarity mechanism for subgraph-level interpretability; code at [github.com/idruglab/hignn](https://github.com/idruglab/hignn) — [arXiv 2208.13994](https://arxiv.org/abs/2208.13994)
- Reported scaffold-split figures surfaced in aggregator listings (BACE ≈ 0.8454 classification, Lipo ≈ 1.68) could not be verified against the journal text — treat as unconfirmed — [aggregator listing](https://www.emergentmind.com/papers/2208.13994)

**PGMG (Zhu et al., Nature Communications 14, 6234, 2023)**

- Represents the pharmacophore as a **fully connected graph whose nodes are pharmacophore features and whose node-pair edge labels are shortest-path distances on the molecular graph**; a GNN encodes that graph and a transformer decoder generates SMILES — [PMC10558534](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558534/); [arXiv 2207.00821](https://arxiv.org/abs/2207.00821); DOI [10.1038/s41467-023-41454-9](https://doi.org/10.1038/s41467-023-41454-9)
- Design intent directly relevant here: pharmacophore hypotheses act "as a bridge to connect different types of activity data", enabling generation for **newly discovered targets with little activity data, without fine-tuning** — [PMC10558534](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558534/)
- Code: [github.com/CSUBioGroup/PGMG](https://github.com/CSUBioGroup/PGMG)

### Inferences

- The consistent architectural pattern is **not** "concatenate a pharmacophore vector onto the graph embedding". It is either (a) a second message-passing stage over typed pharmacophore nodes that *pool the atom states learned by the first stage* (RG-MPNN), or (b) attention fusion where the atom view queries the pharmacophore view (PharmHGT). Both make the pharmacophore channel a *function of the learned atom representation*, so it cannot be redundant-but-noisy in the way a fixed external vector can be.
- The atom↔pharmacophore **junction edges carry much of the value**: PharmHGT's best reduced variant is junction→atom, and its pharm-only view is the worst single view. A pharmacophore channel that is detached from the atom graph (a molecule-level vector) discards exactly this information.
- RG-MPNN's ablation range (−0.010 to +0.046 AUC) is the most honest effect-size estimate in this literature: adding a pharmacophore channel is sometimes neutral-to-harmful, and the benefit depends strongly on the backbone (55% of plain MPNNs improved vs 100% of residual MPNNs).

### Gaps

- **"PhIK" could not be located.** No pharmacophore-ML method under that name appeared in searches; the only "PhIK" in the literature is an unrelated physics-informed Kriging method. Treat the name as unverified.
- PharmHGT and HiGNN ablation magnitudes are published only as figures; exact per-variant deltas are not extractable from the text.
- DiffPhore, PharmacoNet, MolCLR, HiMol, MGSSL and FG-BERT were not researched within budget. The first three in that list (DiffPhore, PharmacoNet) are 3D structure-based docking/pharmacophore-matching methods and are a different problem from a molecule-level descriptor for outcome prediction; MGSSL and FG-BERT are motif-based self-supervised pretraining rather than pharmacophore encodings.

---

## Q2: Fusing fixed pharmacophore fingerprints with GNN embeddings — which fingerprints, and what the ablations showed

### Takeaway

FP-GNN is the main reference point: it concatenates **MACCS (167) + PubChem (881) + pharmacophore ErG (441) = 1,489 dims**, pushes that through a one-layer linear FPN, and fuses with a GAT embedding under a tunable GNN/FPN mixing ratio. The ablations support "graphs and fingerprints are complementary" but **never isolate ErG**, so there is no published number for the marginal value of a pharmacophore fingerprint on top of an atom GNN.

### Cited Findings

**FP-GNN (Cai et al., Briefings in Bioinformatics 23(6):bbac408, 2022; arXiv 2205.03834)**

- Fingerprints used and their exact widths: **MACCS "short variant of 1+166 bits"** (= 167), **PubChem 881 bits**, **Pharmacophore ErG** ("a 2D pharmacophore fingerprint using an extended reduced graph method"). The interpretability section states the mixed fingerprint "had **1489 bits in total**" → ErG contributes **441 dims** (1489 − 881 − 167) — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834); DOI [10.1093/bib/bbac408](https://doi.org/10.1093/bib/bbac408)
- Fusion is deliberately minimal: `FP = (FP_PubChem ‖ FP_MACCS ‖ FP_ErG)`, then **a single affine layer `V' = W2·FP + b`**; the GNN path is a multi-head attention GNN with mean-pooled node readout; the two outputs are "fitted together" into fully connected layers — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- The **mixing ratio between GNN and FPN is a tuned hyperparameter** (one of six optimized by Hyperopt Bayesian search, alongside GNN dropout, number of attention heads, attention hidden size, FPN hidden size, FPN dropout) — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- Across the 13 public datasets, **54.3% of tuned GNN-ratios fell between 0.4 and 0.6**, and **pure-GNN or pure-FPN solutions accounted for only ~4.3%** of models — the search itself never wanted to discard either channel — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- **LIT-PCBA ablation** (15 targets, 7,844 actives / 407,381 inactives; GNN ratio pinned at 0.5): **FP-GNN beat both FPN-only and GNN-only on 10 of 15 targets**; on the other five (ESR1_ago, FEN1, KAT2A, MTORC1, OPRK1) it sat between them — slightly below GNN-only but "evidently higher than FPN" — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- **Mixed-FP vs single Morgan/ECFP-4 (1024 bits) inside the same FP-GNN architecture:** mixed fingerprints won on both classification and regression datasets (Fig. 5). But in the classical-ML comparison on LIT-PCBA the mixed set "did not exhibit absolute superiority" — **42 best models vs 30** for Morgan, with 3 ties — and there was a trend that **simple models (NB, SVM) extracted more from Morgan, while RF/XGBoost/DNN extracted more from the mixed set** — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- **Per-bit importance on FreeSolv:** of the 10 most important of the 1,489 bits, **4 came from MACCS, 3 from ErG, 3 from PubChem** — ErG bits are used, roughly in proportion to a uniform draw. Pearson correlations with hydration free energy exceeded 0.7 for bits 3, 6 and 10 — [arXiv 2205.03834](https://arxiv.org/abs/2205.03834)
- Code: [github.com/idrugLab/FP-GNN](https://github.com/idrugLab/FP-GNN)

**Follow-ups confirming the ErG-in-the-mix recipe**

- **Multi-Level Fusion GNN (arXiv 2507.03430)** ran a fingerprint-set ablation against FP-GNN's exact trio (MACCS + PubChem + ErG) and reports its own **Morgan + PubChem + ErG** combination performed best across datasets, with performance degrading when the fingerprint component was removed entirely — [arXiv 2507.03430](https://arxiv.org/pdf/2507.03430)
- **FP-GNN++** replaces plain concatenation with a graph-attention module plus **cross-attention fusion** between the fingerprint and graph branches, ablating bond features, cross-attention, self-attention, and a GraphSAGE swap. The numeric table could not be retrieved; the document is a non-peer-reviewed PDF and should be treated as weak evidence — [FP-GNN++ PDF](https://bearthesilly.github.io/fpgnn++.pdf)

### Inferences

- The strongest published evidence for pharmacophore fingerprints in a fusion model is **collective, not isolated**: ErG only ever appears bundled with two substructure-key fingerprints. The FreeSolv bit-importance split (4/3/3) is the closest thing to an ErG-specific signal and it is weak — it says ErG bits are not ignored, not that they add predictive power.
- FP-GNN's two most transferable design decisions for a two-tower setup are (i) **make the graph/fingerprint mixing ratio a tuned hyperparameter rather than a fixed 50/50 concat**, and (ii) **keep the fingerprint head shallow** (one affine layer) so the branch cannot overfit on its own. Pinning the ratio at 0.5 is exactly what the authors blame for the 5 targets where FP-GNN underperformed GNN-only.
- The "42 vs 30" classical-ML result is a caution: a pharmacophore-enriched descriptor set helps *high-capacity* models and can hurt low-capacity ones. In a two-tower GNN the fused head is high-capacity, which is favourable — but it also means the pharmacophore vector will be absorbed only if it carries signal the atom tower lacks.

### Gaps

- **No published ablation removes only ErG (or only a pharmacophore fingerprint) from a GNN+FP fusion model.** This is the single biggest evidence gap for the team's exact question. The marginal value of a molecule-level pharmacophore vector on top of an atom GNN is, as of this search, unquantified.
- **CATS** (Chemically Advanced Template Search) descriptors did not surface in any GNN-fusion ablation. No evidence found for CATS-in-GNN.

---

## Q3: Representation choices — binary vs count vs fuzzy, distance-bin resolution, scaling, dimensionality, folding

### Takeaway

Counts beat binary where it has been tested head-to-head (9 of 10 QSAR datasets). For 2D pharmacophore pair/triplet descriptors the two production reference implementations use **far finer distance binning than 4 bins**: RDKit's Gobbi/Pharm2D uses **7 topological-distance bins**, and ErG uses **15 bins by default (21 pairs × 15 = 315 dims)** with **fuzzy spill-over into adjacent bins (default increment 0.3, explicitly tuned for scaffold hopping — i.e. tuned to *increase* similarity)**. That default is a direct candidate explanation for high cosine similarity between unrelated drugs in a coarsely-binned, log-scaled, fuzzy-ish descriptor. No literature was found that prescribes z-scoring or log scaling for pharmacophore vectors specifically.

### Cited Findings

**Counts vs binary**

- Zhong & Guan (*Environ. Sci. Technol.*, 2023), "Count-Based Morgan Fingerprint: A More Efficient and Interpretable Molecular Representation…": across **10 datasets and several ML algorithms, the count fingerprint (C-MF) outperformed the binary (B-MF) in 9 of 10 datasets**. They add that SHAP interpretation of the count model "can elucidate the effect of atom group counts on the target and have a wider range of SHAP values", and that the applicability domain is comparable to the binary model's — [publication record](https://pure.ecnu.edu.cn/en/publications/count-based-morgan-fingerprint-a-more-efficient-and-interpretable/); [data collection](https://figshare.com/collections/Count-Based_Morgan_Fingerprint_A_More_Efficient_and_Interpretable_Molecular_Representation_in_Developing_Machine_Learning-Based_Predictive_Regression_Models_for_Water_Contaminants_Activities_and_Properties/6728035)
- "Count your bits: more subtle similarity measures using larger radius count vectors" (Huber & Pollmann, 2025) reports that **sparse and count-based fingerprints consistently outperform fixed-size binary vectors in preserving structural distinctions**, evaluated on large heterogeneous metabolomics data for virtual screening and similarity — [record](https://opus4.kobv.de/opus4-hs-duesseldorf/frontdoor/index/index/year/2025/docId/5463); secondary summary: [This Week in Cheminformatics #028](https://thisweekincheminformatics.substack.com/p/twenty-eight)
- A NeurIPS 2025 workshop preprint on multi-objective BO with Tanimoto-kernel GPs uses **count-based ECFPs specifically to test whether hash collisions in counted fingerprints hurt performance** ("Count fingerprints store the frequency of each substructure, as opposed to binary fingerprints which only indicate the presence/absence of each substructure") — its conclusion on collision impact was not confirmed — [arXiv 2508.14072](https://arxiv.org/pdf/2508.14072)
- Counter-point on cost: binary and count fingerprints are essentially equal in compute, so the difference is representational, not efficiency — [This Week in Cheminformatics #028](https://thisweekincheminformatics.substack.com/p/twenty-eight)

**Distance-bin resolution for 2D pharmacophore pairs/triplets**

- **RDKit Gobbi_Pharm2D / DefaultSigFactory default bins: (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,100) — 7 bins**, with the last an open-ended catch-all. Feature definitions and bins are user-definable via `SigFactory(bins=...)`, and `Init()` must be re-run after changing them — [rdkit.Chem.Pharm2D.SigFactory](https://www.rdkit.org/docs/source/rdkit.Chem.Pharm2D.SigFactory.html); [RDKit UGM 2012 fingerprints deck](https://mail.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf)
- RDKit's `Gobbi2d` uses Gobbi & Poppinger's own acidic-group pattern and the same 7 bins, but **the "fuzzing" of the original 1998 paper is not implemented**; `Pharm2D` instead uses RDKit `BaseFeatures.fdef` definitions — [RDKit UGM 2012 deck](https://mail.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf); [rdkit.Chem.Pharm2D package](https://rdkit.org/docs/source/rdkit.Chem.Pharm2D.html)
- **ErG (Stiefl, Watson, Baumann, Zaliani — "ErG: 2D pharmacophore descriptions for scaffold hopping", J. Chem. Inf. Model.):** described by the authors as "a hybrid approach of reduced graphs and binding property pairs" using "pharmacophore-type node descriptions to encode the relevant molecular properties" — [record](https://data.zhenzhi.com/details/1/861823400)
- **ErG dimensionality is exactly `21 × (max_distance − min_distance + 1)`**, i.e. **315 for the defaults (min_path 1, max_path 15)**. The 21 comes from **6 property node types** (H-bond donor, H-bond acceptor, aromatic ring system, hydrophobic/aliphatic ring system, positive formal charge, negative formal charge) forming 21 unordered type pairs including same-type pairs (6×7/2). Each distance in range adds one block of 21, so widening the range grows the vector linearly — [skfp.fingerprints.ERGFingerprint](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.fingerprints.ERGFingerprint.html); RDKit: [rdkit.Chem.rdReducedGraphs](https://rdkit.org/docs/source/rdkit.Chem.rdReducedGraphs.html)
  - (The docs' `n_features_out` string says `21 * (max_path - min_path)` = 294, which is inconsistent with the stated default of 315; the inclusive formula is the correct one.)
- **ErG value semantics — "fuzzy incrementation instead of counts":** each occurrence increments its own bin by 1 **and the adjacent dist−1 and dist+1 bins by `fuzz_increment`**. Default `fuzz_increment = 0.3`, **documented as optimized for scaffold-hopping applications**; the docs explicitly say **values around 0.1–0.2 give "more crisp" similarity** (less weight on neighbouring distances). RDKit's `GetErGFingerprint` uses the same defaults (fuzzIncrement 0.3, minPath 1, maxPath 15) — [skfp ERGFingerprint](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.fingerprints.ERGFingerprint.html); [rdkit.Chem.rdReducedGraphs](https://rdkit.org/docs/source/rdkit.Chem.rdReducedGraphs.html)
- **ErG offers three variants: `fuzzy` (default, float), `bit`, and `count` (both integer, and both ignore `fuzz_increment`)** — so binary / count / fuzzy is a one-parameter switch in an off-the-shelf implementation — [skfp ERGFingerprint](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.fingerprints.ERGFingerprint.html)
- **ErG output is not normalized** by the implementation — [skfp ERGFingerprint](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.fingerprints.ERGFingerprint.html)
- **Coarser binning does appear in recent work:** TransPharmer ("Accelerating Discovery of Novel and Bioactive Ligands With Pharmacophore-Informed Generative Models") built its own RDKit-based 2D pharmacophore fingerprints and **discretized topological distances into 2-bin or 3-bin signals with a maximum distance of 8**. This is from search-snippet level only — the PDF exceeded fetch limits and the claim is unverified against the paper text — [arXiv 2401.01059](https://arxiv.org/abs/2401.01059)

### Inferences

- **The team's 4 distance bins is coarse relative to every reference implementation** (ErG 15, Gobbi/Pharm2D 7). With 36 pairs × 4 bins, the number of distinguishable "slots" is 144 vs ErG's 315 at the same 21-pair granularity — and crucially, 4 bins over a realistic topological-distance range (1–15+) means most pairs in a drug-sized molecule land in the top, open-ended bin. That compresses molecules towards a common profile and is a plausible mechanical cause of the observed high cosine similarity.
- **`log1p` on counts plus cosine similarity compounds the problem.** `log1p` squashes the large counts that differentiate big molecules, so after normalization most entries sit in a narrow band; cosine then measures mostly the *shared support pattern* (which families are present at all), and almost every drug-like molecule has donors, acceptors, aromatics and hydrophobes. Two independent levers follow from the literature: (i) **use counts rather than log counts** (counts beat binary in 9/10 datasets — but note nobody tested log-counts head-to-head, so this is an argument against *squashing*, not direct evidence for raw counts), and (ii) **z-score each dimension across the training drugs** so that high-variance, discriminating dimensions dominate the geometry instead of the always-on ones. Cosine on z-scored features is effectively a correlation, which removes the "everything is similar because everything has an H-bond acceptor" floor.
- **Fuzzy spill-over should be treated as a tunable, and probably reduced.** ErG's default 0.3 was chosen to *make dissimilar scaffolds look similar* (scaffold hopping). If the goal is drug *separation* in a pharmacogenomic model, the documented "crisp" setting (0.1–0.2, or 0 / the `count` variant) is the aligned choice. If the current 152-dim vector involves any adjacency smoothing, that is working against separation by design.
- **Folding/collisions are not the team's problem.** 152 dims of *enumerated* family-pair × bin features are a dense, exactly-indexed representation with no hashing, so collision literature (which concerns hashed Morgan/ECFP) does not apply. The relevant failure mode here is *saturation*, not collision.
- **Featureless / degenerate molecules need an explicit path.** RG-MPNN documents the analogous failure (a molecule that collapses to one pharmacophore node can pass no RG-level information), and ErG cannot form a pair in a molecule with fewer than two property points. An all-zero 152-dim vector is indistinguishable from "no information" in a cosine geometry and is a second contributor to clustering; an explicit presence/validity indicator dimension is the natural fix.

### Gaps

- **No source found that prescribes or benchmarks normalization (log vs sqrt vs z-score) for pharmacophore descriptor vectors.** The count-vs-binary evidence is for Morgan counts in QSAR, not for pharmacophore pair counts, and no study in this search applied log1p to pharmacophore counts and compared it against alternatives. Treat the scaling recommendation as reasoned, not cited.
- **No study found that systematically varies distance-bin resolution for 2D pharmacophore fingerprints and reports downstream ML accuracy.** The 7-bin and 15-bin choices are implementation defaults with no published sensitivity analysis found; TransPharmer's 2–3 bin choice is stated without an accompanying ablation (and is itself unverified here).
- **Pair vs triplet features: no clean comparison found.** ErG's documentation calls its units "triplets" while its dimensionality is organized by *pairs* of property types at a distance — the terminology in that doc is internally confusing and was not resolved. RDKit's Pharm2D supports 3-point and (per a mailing-list thread) 4-point pharmacophores, but no benchmark of 2-point vs 3-point ML performance was located — [Pharm2D 4-point thread](https://sourceforge.net/p/rdkit/mailman/message/34837629)
- Gobbi-specific bit layout in current RDKit was not verified; `Gobbi_Pharm2D` + `GetBitDescriptionAsText` in the installed version is the authoritative check.

---

## Q4: Evidence from DDI / drug response / ADR / PGx models — does a pharmacophore channel help cold-start generalization to unseen drugs?

### Takeaway

Pharmacophore-as-modeling-unit is now being published for DDI (PC-DDI at AAAI-26, CMCL-DDI 2026), with small transductive gains (+1.84% accuracy). **The one paper that actually ran an unseen-drug cold-start split found its pharmacophore-based model did *not* beat the best atom-graph baseline on AUROC/AUPRC** — it won on accuracy and F1 only. There is essentially **no evidence** for pharmacophore descriptors in drug-response or pharmacogenomic outcome prediction, and the cold-start hypothesis ("pharmacophores generalize across scaffolds, so they should help unseen drugs") remains untested rather than supported.

### Cited Findings

**PC-DDI (Dou, Wen, Xie, Tang, Ma, Guo — AAAI 2026, 40(25):20887–20895), "Closer to Biological Mechanism: Drug-Drug Interaction Prediction from the Perspective of Pharmacophore"**

- Treats **pharmacophores as the basic modeling unit rather than whole-molecule structure**: extracts pharmacophore features from drug molecular graphs and organizes them into a **bipartite pharmacophore graph** connecting one drug's pharmacophores to the other drug's — [AAAI listing](https://ojs.aaai.org/index.php/AAAI/article/view/39229)
- Uses a spatial attention weight graph convolution module (SAWGCN); **causal inference identifies key pharmacophores** in the bipartite graph for interpretability — [AAAI listing](https://ojs.aaai.org/index.php/AAAI/article/view/39229)
- Reported gain: **"accuracy improvement of 1.84% under the transductive setting"** and consistently outperforming others in other experiments — [AAAI listing](https://ojs.aaai.org/index.php/AAAI/article/view/39229); talk: [Underline](https://underline.io/lecture/139881-closer-to-biological-mechanism-drug-drug-interaction-prediction-from-the-perspective-of-pharmacophore)
- Caveat: the OJS full text is behind a bot-protection wall and could not be fetched, so **graph-construction details, the pharmacophore type vocabulary, baselines list, and whether a cold-start split exists are unverified**.

**CMCL-DDI (PLOS ONE, 2026) — "Pharmacophore-aware cross-view contrastive learning for DDI prediction"**

- Decomposes molecules into **pharmacophore subgraphs via BRICS**; atoms are nodes (features: atom type H/C/O/N), bonds are edges (single/double/triple); features are projected to a shared dim and concatenated; **a Transformer encodes each pharmacophore** and a readout pools them to a molecule vector — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- Important architectural detail: **there is no separate atom-level view.** The "graph view" is pharmacophore subgraphs only; the second view is **SMILES encoded by MolBERT** with mean pooling. Cross-view contrastive loss uses cosine similarity with temperature τ; positives are the same drug's graph/SMILES embeddings — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- Datasets: DrugBank (1,706 drugs, 191,808 triplets, 86 interaction types) and TWOSIDES (645 drugs, 963 categories, 4,576,287 triplets) — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- **Warm-start** (Table 3): DrugBank CMCL-DDI ACC 98.26 / AUROC 99.23 / AUPRC 99.25 / F1 97.69 vs best baseline SSI-DDI 96.33 / 98.95 / 98.57 / 96.38. TWOSIDES 90.25 / 94.27 / 91.63 / 88.93 vs best baseline GMPNN-CS 86.96 / 92.94 / 90.38 / 87.85 — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- **Cold-start / inductive (20% of drugs held out, train and test drug sets disjoint), Table 4 — the key negative result:** DrugBank CMCL-DDI ACC 88.15 / AUROC 87.69 / AUPRC 87.31 / F1 83.39, but **GMPNN-CS reaches AUROC 89.34 and AUPRC 89.25 — i.e. the pharmacophore model loses by ~1.65 AUROC points to an atom-level graph baseline on unseen drugs**, winning only on F1 (+6.17). On TWOSIDES CMCL-DDI does lead on AUROC: 88.63 vs DGNN-DDI 85.12 (+3.51), with ACC 80.21 / AUPRC 85.92 / F1 75.81 (+8.60 F1) — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- The paper's text claims a "1.34% AUROC gain on DrugBank" that **contradicts its own Table 4**; it also states cold-start runs were repeated "three times" in one place and "5 runs" in another. Reliability of this paper's narrative claims is therefore low, though the tables are usable — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)
- **Ablation (cold-start, DrugBank, Table 5):** removing the pharmacophore graph view → 82.92 / 81.95 / 80.95 / 78.67 (full model 88.15 / 87.69 / 87.31 / 83.39, so −5.23 ACC / −5.74 AUROC); removing the sequence view → 80.33 / 80.25 / 80.18 / 76.12. **But no ablation isolates pharmacophore decomposition from a plain atom-level graph** — removing the "graph view" removes the whole structural branch, not the pharmacophore grouping — [PMC12928573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12928573/)

**Cold-start DDI literature without pharmacophores (for calibration)**

- A language-grounded cold-start DDI paper (NeurIPS 2026) reports that **a structure-only fingerprint baseline is the strongest transductive model on both datasets but degrades the most under cold start** — the canonical warning that fixed structural descriptors memorize rather than generalize — [NeurIPS listing](https://neurips.cc/virtual/2026/161194)
- **ColdDDI** benchmarks three levels of drug novelty and finds that in the hardest both-drugs-unseen split, the dominant factor is **mediator availability** (external knowledge), not the molecular encoder — [NeurIPS poster](https://neurips.cc/virtual/2026/poster/138874)
- LAMFP extends AMFP to unseen drugs by **looking up structurally similar known drugs** — nearest-neighbour transfer rather than a better descriptor — [search summary](https://www.researchsquare.com/article/rs-10334947)
- A simplified similarity-based DDI approach is a further baseline in this space — [PMC10635435](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10635435/)

**ADR / toxicity prediction with pharmacophore-type descriptors**

- The closest direct precedent is old and small: Bhavani et al. (2006) combined **toxicophore-type and toxicophore-position descriptors** in an SVM — "thus considering both the pharmacophore and the three-dimensional shape information of the molecule" — for torsade de pointes, reporting **up to 97% accuracy on positives and 90% on negatives** — [IISc ePrints 9174](https://eprints.iisc.ac.in/9174)
- A 2024 comparison across RF / gradient-boosted trees / FFN / CNN for ADR prediction found **tree models ahead of neural networks, "significant differences of even 20% in accuracy in certain cases"**, e.g. with molecular descriptors for hepatobiliary disorders — [Zenodo 14475718](https://zenodo.org/records/14475718)
- CrossADR (arXiv 2026) for combination pharmacotherapy uses RDKit physicochemical descriptors (MW, PSA) + MACCS with attention weighting over features — descriptors, not pharmacophores — [arXiv 2603.15047](https://arxiv.org/pdf/2603.15047)
- IBM's interpretable ADR framework (IJCAI 2018) reports **neural fingerprints outperformed all other methods among ten fingerprint models** for ADR prediction — i.e. a *learned* representation beat all fixed fingerprints on exactly this task family — [IBM Research publication](https://research.ibm.com/publications/predicting-adverse-drug-reactions-through-interpretable-deep-learning-framework) (note: search results listed two different first authors, Morrone and Gao; attribution unconfirmed)

### Inferences

- The only cold-start number available for a pharmacophore-centric model is **mixed-to-negative on the ranking metrics that matter** (AUROC/AUPRC on DrugBank), and its F1 advantage is as easily explained by threshold/calibration behaviour as by better representation. Combined with RG-MPNN's inconsistent scaffold-split trend, the honest reading is: **the "pharmacophores transfer across scaffolds, so they help unseen drugs" claim is a plausible hypothesis with no clean supporting measurement in the literature surveyed.**
- The cold-start DDI finding that fixed structural fingerprints are the *best transductive* and the *worst-degrading* model is a direct warning for the team's design: a 152-dim fixed vector concatenated at the fused head is likely to be exploited for memorization in a random split and to contribute little or negatively under `drug_cold`. The project's existing `drug_cold` scenario is exactly the right instrument to detect this, and the pharmacophore vector should be judged on `drug_cold` deltas rather than random-split deltas.
- The high-cosine-similarity symptom is itself diagnostic independent of any downstream metric: if unrelated drugs are near-collinear in the 152-dim space, the vector cannot be supplying drug-discriminating information to the fused head, so it can only act as a weak prior or as noise. Fixing the representation geometry (binning resolution, z-scoring, crisp rather than fuzzy counting, a validity indicator) should precede any further architectural work.
- Everything in this literature that works keeps pharmacophores *attached* to the atom graph (junction edges, child-atom pooling) rather than as a detached molecule-level vector. If the fixed vector cannot be made discriminative, the better-supported alternative for this codebase is a **second pooling stage over typed pharmacophore nodes inside the drug tower** (the RG-MPNN pattern), which reuses the already-learned atom states and needs no external descriptor.

### Gaps

- **No evidence found for pharmacophore descriptors in drug response (GDSC/CCLE-style) or pharmacogenomic outcome prediction.** DeepDDS, DrugBAN and GraphDTA variants were not found to use pharmacophore features; the searches surfaced no PGx model with a pharmacophore channel. This is a genuinely empty cell in the literature, not a search failure to report as thin.
- **PC-DDI's full text could not be retrieved** (bot protection on the AAAI OJS server). Its pharmacophore vocabulary, graph construction, baseline set, and whether it evaluates unseen drugs are all unverified; only the abstract-level +1.84% transductive accuracy claim is citable.
- No paper found that reports pharmacophore-feature contribution specifically under a *drug cold-start* ablation (i.e. same model, same split, with and without the pharmacophore channel). CMCL-DDI's Table 5 ablation is the nearest, and it confounds the pharmacophore grouping with the entire structural branch.
- TransPharmer's bin design and DiffPhore / PharmacoNet were not verified against their full texts within the tool budget.
