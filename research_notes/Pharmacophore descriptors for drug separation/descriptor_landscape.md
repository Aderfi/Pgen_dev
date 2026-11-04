# Pharmacophore and functional-class descriptors for small-molecule drugs: catalog, software, and head-to-head benchmarks

Scope note: all "measured locally" numbers in this file were produced in this session with **RDKit 2025.09.6** on the project machine (200-molecule drug-like set, single core). They are reproducible and are flagged as such; everything else carries a literature citation.

## Q1. Catalog: what does each pharmacophore / functional-class descriptor encode, and at what length?

### Takeaway

The family splits into (a) **topological (2D) pharmacophore pair/triplet descriptors** — CATS (15 pair types x 10 distances = 150 counts), Similog DABE triplets (8,031 codes observed in MDDR), Gobbi-Poppinger 2-/3-point keys (39,972 bits unfolded in RDKit); (b) **reduced-graph / property-profile descriptors** — ErG (315 floats in RDKit), FEPOPS (7 x 4 feature points x 22 values); (c) **3D 3-/4-point pharmacophore keys** — PharmPrint (10,549 three-point basis) and Chem-X/ChemDiverse 4-point keys (up to ~350 million potential 4-point pharmacophores per molecule); and (d) **functional-class circular fingerprints** — FCFP/FCFC, which are ECFP with atoms re-typed to 6 pharmacophoric classes. Only CATS, ErG, Gobbi Pharm2D, FCFP, atom pairs, topological torsions, MAP4 and MHFP are available in open-source Python (RDKit / scikit-fingerprints / map4).

### Cited Findings

**Gobbi-Poppinger 2D pharmacophore fingerprint (RDKit `Chem.Pharm2D` + `Gobbi_Pharm2D`)**
- Original source cited by RDKit itself: A. Gobbi and D. Poppinger, *Biotechnology and Bioengineering* **61**(1), 47-54 (1998), "Genetic optimization of combinatorial libraries". The RDKit module docstring reads verbatim: `Definitions for 2D Pharmacophores from: Gobbi and Poppinger, Biotech. Bioeng. _61_ 47-54 (1998)` (measured locally via `rdkit.Chem.Pharm2D.Gobbi_Pharm2D.__doc__`). Citation independently confirmed by a Springer chapter reference list — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-540-36213-5_1); [Scilit record](https://scilit.com/publications/5cd8aadf6496849ecedb0101b458d9ed). Caveat: I could not access the 1998 full text, so I cannot independently verify from the primary source that the pharmacophore definitions appear there rather than in a companion paper.
- Exact RDKit feature set (read from `Gobbi_Pharm2D` source, measured locally): **8 feature families** — `LH` Hydrophobic, `HD` Donor, `HA` Acceptor, `AR` AromaticAttachment, `RR` AliphaticAttachment, `X` UnusualAtom, `BG` BasicGroup, `AG` AcidicGroup — each defined by an explicit SMARTS pattern.
- Encoding: **2-point and 3-point** pharmacophores (`minPointCount=2, maxPointCount=3`), **7 topological distance bins** `[(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,100)]` (measured locally).
- Dimensionality: `factory.GetSigSize()` = **39,972 bits, binary** (measured locally). For paracetamol only 21 bits are set (density 0.05%). Bits are self-describing: `GetBitDescription(57)` returns `AR AR |0 1|1 0|` (measured locally).
- RDKit generic API: `Pharm2D.DefaultSigFactory(fdefFile, minPointCount=2, maxPointCount=3, bins=...)` with the same default bins; `Generate.Gen2DFingerprint(mol, factory)`; `Matcher.GetAtomsMatchingBit()` maps a bit back to atoms (atom-mapping added to RDKit in 2015) — [RDKit Pharm2D docs](https://rdkit.org/docs/source/rdkit.Chem.Pharm2D.html); [RDKit mailing list, Landrum on GetBitDescription/bitInfo](https://sourceforge.net/p/rdkit/mailman/message/34256384).
- RDKit also supports 4-point Pharm2D in principle via `maxPointCount=4`, discussed on the dev list — [Rdkit-devel: Pharm2D 4-point pharmacophores](https://sourceforge.net/p/rdkit/mailman/message/34836736).
- RDKit's own docs define the vocabulary: a 2D pharmacophore is an N-point combination of features plus inter-feature distances; "proto-pharmacophores" are features without distances and "scaffolds" are distances without feature identities — [RDKit Pharm2D docs](https://rdkit.org/docs/source/rdkit.Chem.Pharm2D.html). The feature definitions and bins are user-definable — [RDKit Pharm2D Generate](https://rdkit.org/docs/source/rdkit.Chem.Pharm2D.Generate.html).
- An R wrapper exposes the same Gobbi feature set as `fp_pharm2d(type = "gobbi")` — [tidychem fp_pharm2d](https://rdrr.io/github/nanxstats/tidychem/man/fp_pharm2d.html).

**ErG — Extended Reduced Graphs (RDKit `rdReducedGraphs`)**
- Primary paper: N. Stiefl, I. A. Watson, K. Baumann, A. Zaliani, "ErG: 2D Pharmacophore Descriptions for Scaffold Hopping", *J. Chem. Inf. Comput. Sci.* **46**(1), 208-220 (2006), DOI 10.1021/ci050457y — [dblp record](https://dblp1.uni-trier.de/rec/journals/jcisd/StieflWBZ06.html); [abstract summary](https://data.zhenzhi.com/details/1/861823400).
- Design: explicitly "a hybrid approach of reduced graphs (Gillet et al.) and binding property pairs (Kearsley et al.)", extended to represent pharmacophoric properties, size and shape — [abstract](https://data.zhenzhi.com/details/1/861823400).
- RDKit API: `rdReducedGraphs.GetErGFingerprint(mol, fuzzIncrement=0.3, minPath=1, maxPath=15)`; also `GenerateMolExtendedReducedGraph` — [RDKit rdReducedGraphs docs](https://rdkit.org/docs/source/rdkit.Chem.rdReducedGraphs.html); [RDKit C++ ReducedGraphs.h](https://rdkit.org/docs/cppapi/ReducedGraphs_8h.html).
- Dimensionality: **315 floating-point values** at RDKit defaults (measured locally: `len(GetErGFingerprint(paracetamol)) == 315`, 21 non-zero entries). This is `21 property-pair types x 15 distance steps`. scikit-fingerprints documents the length as `21 * (max_distance - min_distance + 1)` because it "always uses 6 features", and notes RDKit **omits two features from the original paper** — [skfp ERGFingerprint](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.fingerprints.ERGFingerprint.html). The two formulas quoted on that page (`21*(max-min+1)` and `21*(max-min)`) disagree; my local measurement (315 = 21 x 15 with minPath=1, maxPath=15) supports `21*(maxPath-minPath+1)`.
- Continuous, not binary: ErG spreads weight across neighbouring distance bins via `fuzzIncrement` (RDKit default 0.3), i.e. it is a *fuzzy count/continuous* descriptor — [RDKit rdReducedGraphs docs](https://rdkit.org/docs/source/rdkit.Chem.rdReducedGraphs.html).
- Also wrapped by Ersilia as a deployable model (`eos5guo`), described there as "well suited for scaffold hopping approaches" — [Ersilia eos5guo](https://hub.docker.com/r/ersiliaos/eos5guo).

**CATS / CATS2D / CATS3D**
- Primary paper: G. Schneider, W. Neidhart, T. Giller, G. Schmid, "'Scaffold-Hopping' by Topological Pharmacophore Search: A Contribution to Virtual Screening", *Angewandte Chemie Int. Ed.* **38**(19), 2894-2896 (1999) — [record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_miscellaneous_1859313525).
- Exact 1999/2004-era encoding, as re-implemented and described by Hert et al.: generalised atom types **lipophilic, positive, negative, H-bond donor, H-bond acceptor**; "the occurrences of the 15 possible pairs of pharmacophores are determined for distances up to 10 bonds to give a **150-element (i.e. 15 x 10) vector**" — [Hert et al. 2004 author version, p.9](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Count-based by construction; Hert et al. had to bin counts into 8-bit strings to get a 1,200-bit binary form for kernel methods (see Q4) — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Requires no 3D/target information; atomic properties are H-donor, H-acceptor, partial charge, lipophilicity — [Chemical Descriptors Library: Topological Pharmacophores](https://cdelib.sourceforge.net/doc/cats.html).
- Variants: **CATS2** separates lipophilic from aromatic atoms; **CATS3D** is the 3D (Euclidean-distance) variant — [Reutlinger et al. 2013, CATS paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/).
- Open-source implementation exists in the Chemical Descriptors Library (C++) — [CDL CATS](https://cdelib.sourceforge.net/doc/cats.html). **No native RDKit CATS implementation was found.**

**Similog keys**
- "DABE" atom typing on four properties: **hydrogen-bond donor (D), acceptor (A), bulkiness (B), electropositivity (E)**, encoded as a 4-bit string per atom; each **atom triplet** is represented by three DABE strings plus the associated topological distances; **8,031 distinct codes were observed across MDDR** — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Occurrence-based (counts), not presence/absence; binned to 8-bit strings, giving a **64,248-bit** representation in Hert et al.'s study (= 8,031 x 8) — [Hert et al. 2004, Table 7](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Proprietary to Novartis at the time of that study (the acknowledgements thank Novartis for "access to the Avalon and Similog descriptors") — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf). **No open-source implementation found.**

**PharmPrint (3-point 3D pharmacophore fingerprint)**
- Primary paper: M. J. McGregor and S. M. Muskal, "Pharmacophore Fingerprinting. 1. Application to QSAR and Focused Library Design", *J. Chem. Inf. Comput. Sci.* **39**(3), 569-574 (1999) — [Scinapse record](https://scinapse.io/papers/2153609134).
- Encoding: a **basis set of 10,549 three-point pharmacophores** enumerated over distance ranges and pharmacophoric features; software assigns pharmacophore types to atoms, generates multiple conformations, and sets the bit for every three-point pharmacophore realised; used as the X-block of a PLS QSAR (estrogen-receptor ligands as the worked example) — [Scinapse record](https://scinapse.io/papers/2153609134); a later review refers to "a 10549-bit signature to describe the three-point pharmacophores".
- Follow-up: "Pharmacophore Fingerprinting. 2. Application to Primary Library Design", *J. Chem. Inf. Comput. Sci.* **40**, 117-125 (2000); PharmPrint reportedly "outperformed other descriptors on a statistic measuring discriminating power across target classes" — [record](https://data.zhenzhi.com/details/1/901748912). I could not retrieve the exact statistic or its value (see Gaps).
- Binary, conformer-ensemble-derived; needs 3D conformer generation, so cost is dominated by conformer enumeration. **No open-source implementation found.**

**3-point and 4-point 3D pharmacophore keys (Chem-X / ChemDiverse, Mason & Pickett)**
- Pickett, Mason, McLay (1996) introduced **pharmacophore-derived queries (PDQ)** using three-point feature combinations for diversity analysis — [Pickett 1996 PDF](https://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Pickett1996Diversity.pdf).
- Mason, Morize, Menard, Cheney, Hulme, Labaudiniere, "New 4-Point Pharmacophore Method for Molecular Similarity and Diversity Applications", *J. Med. Chem.* **42**(17), 3251-3264 (1999) — [PubMed 10464012](https://pubmed.ncbi.nlm.nih.gov/10464012/); [Univ. of Arizona record](https://experts.arizona.edu/en/publications/new-4-point-pharmacophore-method-for-molecular-similarity-and-div/).
- Scale: up to **7 features and 15 distance ranges, giving up to ~350 million potential 4-point 3D pharmacophores per molecule**; implemented as an extension to ChemDiverse/Chem-X; uses "privileged" substructures as a special reference feature, applied to 7TM GPCR library design — [Arizona record](https://experts.arizona.edu/en/publications/new-4-point-pharmacophore-method-for-molecular-similarity-and-div/).
- Conference precursor: Mason & Cheney, *Pacific Symposium on Biocomputing* 1999 — [PSB99 Mason PDF](https://d2ygbh7lwzeoju.cloudfront.net/psb-online/proceedings/psb99/Mason.pdf); follow-up PSB 2000 — [PSB00 Mason PDF](https://d2ygbh7lwzeoju.cloudfront.net/psb-online/proceedings/psb00/mason.pdf).
- These are the historical ancestors of MOE's and Schrödinger Phase's 3D pharmacophore fingerprints. Fully proprietary.

**FEPOPS (FEature POint PharmacophoreS)**
- Primary paper: Jenkins, Glick, Davies, "A 3D Similarity Method for Scaffold Hopping from Known Drugs or Natural Ligands to New Chemotypes", *J. Med. Chem.* **47**(25), 6144-6159 (2004) — [Scilit record](https://scilit.com/publications/57e4c4fa98b7b443d9c8095ed7ff51e2); [abstract summary](https://data.zhenzhi.com/details/1/913435993).
- Encoding in the open-source OpenFEPOPS reimplementation: a molecule is described by typically **7 FEPOPS, each with 4 feature points and 22 values per feature point**; feature points capture charge, lipophilicity, and H-bond donor/acceptor status. Conformer ensembles are k-means-clustered into "fuzzy" 4-point descriptions — [OpenFEPOPS, JOSS paper](https://joss.theoj.org/papers/10.21105/joss.05763.pdf).
- Open-source Python: **OpenFEPOPS** (published in Journal of Open Source Software) — [JOSS 10.21105/joss.05763](https://joss.theoj.org/papers/10.21105/joss.05763.pdf).
- Extension: Nettles et al., *J. Mol. Graph. Model.* **26**(3), 622-633 (2007), a combinatorial comparison algorithm with better retrospective VS on WOMBAT than earlier shape/charge methods — [search record](https://oak.novartis.com/709).

**FCFP / FCFC (functional-class circular fingerprints)**
- Definition: identical algorithm to ECFP, but the initial atom code is built from **six generalised atom types - H-bond donor, H-bond acceptor, positively ionisable, negatively ionisable, aromatic and halogen** - instead of element/charge/mass — [Hert et al. 2004, Experimental section](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Open-source: RDKit Morgan generator with `invariants=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()`; both bit (FCFP) and count (FCFC) forms are used in the Riniker-Landrum platform — [Riniker & Landrum 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Observed sparsity (Hert 2004, MDDR): FCFP_4 unfolded = **30,267 bits**, mean 40.63 bits set, density 0.13%; folded to 1,024 bits, mean 39.44 set, density 3.85%. FCFP_2 unfolded = 600 bits, mean 20.88 set — [Hert et al. 2004, Table 7](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).

**Atom pairs and topological torsions (the "proto-pharmacophore" baselines)**
- Topological torsions date to 1987 and are described by Riniker & Landrum as "one of the oldest and simplest descriptors" — [Riniker & Landrum 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Atom pairs encode (atom type i, atom type j, topological distance) counts; O'Boyle & Sayle attribute AP's strength on their single-assay benchmark to atom pairs "being insensitive to small changes such as C->N substitution" — [O'Boyle & Sayle 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Capecchi et al. note that "atom-pair fingerprints encode molecular shape and are often used for scaffold-hopping" — [Capecchi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).

**MAP4 / MHFP6**
- Primary paper: A. Capecchi, D. Probst, J.-L. Reymond, "One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome", *J. Cheminformatics* **12**, 43 (2020), DOI 10.1186/s13321-020-00445-4 — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/); [Springer PDF](https://link.springer.com/content/pdf/10.1186/s13321-020-00445-4.pdf).
- Encoding: circular substructures of radius 1 and 2 around each atom of an atom pair are written as SMILES "shingles", concatenated with the topological distance between the two atoms, hashed, and the hash set MinHashed — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).
- Dimensionality: **1024- and 2048-dimension versions** (main analyses use 1024); TMAP/similarity-search databases use 512; MinHash parameters m = 2^32 - 1, p = 2^61 - 1; default radius r = 2 (diameter 4 bonds) — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).
- Cost: the authors state the implementation is "implemented in Python and therefore it is relatively slow" and suggest C/C++ would speed it up; **no timings reported** — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/). Code: [github.com/reymond-group/map4](https://gdb.unibe.ch/one-molecular-fingerprint-to-rule-them-all-drugs-biomolecules-and-the-metabolome/).

**MOE TGD/TGT and vendor 3D pharmacophore fingerprints**
- No peer-reviewed primary description of MOE's TGD (typed graph distance) / TGT (typed graph triangle) fingerprints was retrievable in this session; they are documented only in CCG's proprietary manuals. Treated as a gap (see Gaps).

### Computational cost (measured locally, RDKit 2025.09.6, 200 drug-like molecules, single core)

| Descriptor | Length | Type | Throughput | Cost vs ECFP4 |
|---|---|---|---|---|
| ECFP4 (Morgan r=2, 2048 bit) | 2,048 | binary | ~67,900 mol/s | 1.0x |
| ErG (`GetErGFingerprint`, defaults) | 315 | continuous (fuzzy counts) | ~11,600 mol/s | 5.9x slower |
| MACCS (167) | 167 | binary | ~3,100 mol/s | 22x slower |
| Gobbi Pharm2D (2-3 point, 7 bins) | 39,972 | binary | ~685 mol/s | **99x slower** |

- Independent corroboration of the pharmacophore-FP cost problem: in the scikit-fingerprints MoleculeNet benchmark (OGB scaffold split, Random Forest), "the pharmacophore fingerprint was omitted due to excessive computation time for some molecules" — [scikit-fingerprints paper](https://arxiv.org/pdf/2407.13291) (abstract-level confirmation of the library's scope: "over 30 molecular fingerprints", parallel computation, no published timings — [arXiv:2407.13291 abstract](https://arxiv.org/abs/2407.13291)).

### Inferences

- The Gobbi fingerprint's 99x cost penalty and 39,972-bit width mean it is almost always *folded* in practice (to 1,024-4,096 bits) — which is exactly the regime where O'Boyle & Sayle measured bit-collision damage for other high-cardinality fingerprints (see Q4).
- ErG is the only pharmacophore-type descriptor in this catalog that is simultaneously (a) short (315 dims), (b) dense-continuous, and (c) cheap enough (~12k mol/s) for routine ML featurisation. That combination, not its benchmark ranking, explains most of its adoption.
- The pair-vs-triplet distinction is the main driver of dimensionality: pair descriptors land at 150-1,200 dims (CATS) or 315 (ErG), triplet descriptors at 10^4-10^5 (Similog 64,248; Gobbi 39,972; PharmPrint 10,549), and 4-point keys at 10^8 potential features (Chem-X).

### Gaps

- The Gobbi & Poppinger (1998) full text was not retrievable; the feature-definition provenance rests on RDKit's own citation plus third-party reference lists.
- No primary, peer-reviewed description of MOE's TGD/TGT fingerprints, Schrödinger Phase/Canvas pharmacophore fingerprints, or OpenEye's pharmacophore tooling was found. Vendor docs were not accessible in this session.
- PharmPrint paper 2's "statistic measuring discriminating power across target classes" was only described qualitatively in the records I could reach; no number.
- The original CATS 1999 Angewandte paper full text was not accessible; the 150-element encoding is cited from Hert et al. 2004's re-implementation (and from Fechner et al., which Hert et al. followed).

---

## Q2. Which published benchmarks compare these descriptors, and what were the numbers?

### Takeaway

The single most quantitative head-to-head that actually includes 2D pharmacophore descriptors is **Hert et al. 2004** (*Org. Biomol. Chem.* 2, 3256-3266), and it is damning for them: averaged over 11 MDDR activity classes, CATS reached **19.43% recall@1%** and Similog **24.93-26.01%**, versus **42.58% for ECFP_4A** — i.e. pharmacophore descriptors recovered roughly *half* as many actives as circular fingerprints. The two modern reference benchmarks (Riniker & Landrum 2013; O'Boyle & Sayle 2016) deliberately **excluded** pharmacophore fingerprints, so there is no modern, large-scale, peer-reviewed number for Gobbi Pharm2D or CATS against ECFP4.

### Cited Findings

**Hert, Willett, Wilton, Acklin, Azzaoui, Jacoby, Schuffenhauer (2004), *Org. Biomol. Chem.* 2, 3256-3266, DOI 10.1039/B409865J** — the only benchmark here with pharmacophore descriptors and published per-class tables. Design: 15 fingerprint types x 2 search algorithms (data fusion, binary kernel discrimination), 11 MDDR activity classes, 10 active reference structures per search, 10 searches per class — [author version, White Rose](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf); [publisher](https://pubs.rsc.org/en/Content/ArticleLanding/2004/OB/B409865J).

Average recall at **1%**, averaged over all 11 classes (Table 1 = BKD, Table 2 = data fusion):

| Fingerprint | Class | Length | Recall@1% (BKD) | Recall@1% (data fusion) |
|---|---|---|---|---|
| ECFP_4A | circular | 142,864 | **42.58** | **43.69** |
| ECFP_4B | circular | 1,024 | 42.26 | 43.45 |
| ECFP_2A | circular | 7,445 | 41.24 | 40.19 |
| ECFP_2B | circular | 1,024 | 40.93 | 40.08 |
| FCFP_4A | circular, functional class | 30,267 | 40.04 | 39.24 |
| FCFP_4B | circular, functional class | 1,024 | 39.73 | 38.99 |
| BCI | dictionary | 1,052 | 36.01 | 32.84 |
| Daylight | hashed path | 2,048 | 34.70 | 31.97 |
| Unity | hashed path | 988 | 33.95 | 30.61 |
| Avalon | hashed path | 2,048 | 33.54 | 31.59 |
| FCFP_2A | circular, functional class | 600 | 32.41 | 28.53 |
| FCFP_2B | circular, functional class | 1,024 | 31.88 | 28.01 |
| **SimilogB** | **pharmacophore triplet** | 64,248 | **26.01** | 30.67 |
| **SimilogA** | **pharmacophore triplet** | 64,248 | **24.93** | 29.84 |
| **CATS** | **pharmacophore pair** | 1,200 | **19.43** | 20.26 |

(All values are percent recall; Hert et al. 2004, Tables 1-2 — [White Rose PDF](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).)

- Corresponding enrichment factors for ECFP_4B: **42.3 (BKD) and 43.5 (data fusion) at 1%**; **13.1 and 13.5 at 5%** — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Authors' verdict, quoted: "Perhaps our most surprising finding is the performance of the pharmacophore descriptors, with both the CATS and Similog fingerprints yielding consistently poorer recall values. Previous studies of these descriptors, for chemogenomics and scaffold-hopping applications, have demonstrated that they can be highly effective in operation, but this was certainly not the case for the present application." — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Hert et al. also report: "there was only a single case where one of these [circular] fingerprints did not provide the best result, viz the average recall at 1% using BKD for the set of cyclooxygenase inhibitors" — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Class-dependence: CATS was *competitive or best* on exactly two classes — **renin inhibitors (60.61% recall@1% with data fusion, beating Daylight's 51.29 and Unity's 50.23, though below ECFP_4A's 75.10)** and **angiotensin II AT1 antagonists (40.15 vs ECFP_4A 49.99)**, and also thrombin (30.44 with BKD). It collapsed on 5HT3 antagonists (11.79 BKD / 8.71 fusion vs ECFP_4A 45.93 / 52.18), substance P (7.98 / 9.27) and cyclooxygenase (10.45 / 7.83) — [Hert et al. 2004, Tables 1-2](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Companion paper: Hert et al., *J. Chem. Inf. Model.* **44**(3), 1177-1185 (2004), "Comparison of Fingerprint-Based Methods for Virtual Screening Using Multiple Bioactive Reference Structures" — fusing similarity scores is the most effective general approach, best individual results from binary kernel discrimination — [record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edo,ejs9527027).

**Riniker & Landrum (2013), *J. Cheminformatics* 5:26, DOI 10.1186/1758-2946-5-26** — the de facto open-source LBVS benchmark platform — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/); [PDF](https://jcheminf.biomedcentral.com/track/pdf/10.1186/1758-2946-5-26).
- Design: **88 targets total** — 21 DUD targets (those with >30 actives), 17 MUV targets (30 actives + 15,000 decoys each), 50 ChEMBL targets (100 diverse actives + 10,000 decoys each, actives chosen with the RDKit diversity picker); 50 repetitions of 5-query searches — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Fingerprints: **14 total** (12 standard + 2 baselines): ECFC0 and MACCS (baselines), AP, TT, RDK5, Avalon (1,024), lAvalon (16,384), ECFP4 (1,024), lECFP4 (16,384), ECFP6, lECFP6, FCFP4, ECFC4, FCFC4 — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- **Pharmacophore fingerprints were explicitly NOT benchmarked**: the paper names pharmacophores as one of four fingerprint classes and benchmarks "three of the four"; FCFP4/FCFC4 are the only pharmacophoric-feature descriptors included — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Metrics: AUC, EF(1%), EF(5%), RIE(20), RIE(100), BEDROC(20), BEDROC(100). Friedman global test p < 2.2e-16 for every evaluation method — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Key findings: ECFC0 and MACCS performed worst and were significantly lower under early-recognition metrics; under **AUC**, path-based fingerprints (TT, AP, and to a degree RDK5/Avalon) ranked *above* circular fingerprints except ECFC4; under **BEDROC/EF/RIE**, circular fingerprints (especially lECFP4/lECFP6) ranked significantly higher. **TT ranked among the top fingerprints under every evaluation method** — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Between-fingerprint correlations (r^2): ECFP4 vs ECFP6 = 0.999; circular variants 0.97-1.0; ECFC0/MACCS vs others 0.88-0.90; AP vs others 0.91-0.97; ECFP4 1,024 vs 16,384 bits = 0.998; ECFP6 = 0.996; Avalon = 0.993 — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Between-metric correlations (ECFP4): AUC vs EF(5%) r^2 = 0.908; AUC vs BEDROC(20) = 0.894; AUC vs RIE(20) = 0.899; BEDROC(20) vs EF(5%) = 0.995. RIE maxima for ChEMBL: 18.1 (RIE20), 63.2 (RIE100) — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Overall: apart from the baselines, fingerprints performed similarly; **differences between targets were larger than differences between fingerprints**; MUV was hardest — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/). Per-fingerprint numeric tables live in the Additional Files (CSV/PDF); the article text does not carry them, and the supplementary PDF (Additional file 2) contains plotted figures rather than tables (verified by extracting its text locally).
- The paper cites Bender et al. as having found that pharmacophores "performed reasonably well", while reporting no pharmacophore results of its own — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).

**O'Boyle & Sayle (2016), *J. Cheminformatics* 8:36, DOI 10.1186/s13321-016-0148-0** — literature-based similarity benchmark — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683); [PDF](https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-016-0148-0); [NextMove blog](https://nextmovesoftware.com/blog/?p=1833).
- Design: **28 fingerprints**, ChEMBL 20 IC50/Ki/EC50 assays restricted to 8-25 unique molecules; single-assay benchmark = 4,563 five-molecule series from one assay with >=0.38 log-unit activity gaps; multi-assay benchmark = 3,629 five-molecule series linked through shared compounds across four papers; 1,000 repetitions, two-sided t-tests at 1%, Holm-Bonferroni over 378 pairwise comparisons — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Fingerprint list: AP, HashAP, HashTT, RDK5/6/7, TT; Avalon, LAvalon, MACCS; ECFP0/2/4/6, ECFC0/2/4/6, FCFP2/4/6, FCFC2/4/6, LECFP4, LECFP6, LFCFP4, LFCFP6. **No Gobbi pharm2D, no ErG, no CATS** — the only feature-based fingerprints are the FCFP/FCFC family — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Quantitative results: multi-assay — **LECFP4 reproduced or nearly reproduced the series order in 2,201 of 3,629 cases (61%)**, ECFP0 in 1,498 (41%). Single-assay — **HashAP was close to correct in 1,744 of 4,563 cases (38%)**, ECFP0 in 942 (20%) — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Bit-length effect (net pairwise differences, multi-assay): LAvalon vs Avalon **+132**; LECFP6 vs ECFP6 **+172**; LECFP4 vs ECFP4 **+99**; LFCFP6 vs FCFP6 **+96**; LFCFP4 vs FCFP4 **+57**. Single-assay showed no clear trend (Avalon beat LAvalon by 28; FCFP6 beat LFCFP6 by 9; LECFP6 beat ECFP6 by 21; LECFP4 beat ECFP4 by 14). Authors attribute the larger diameter-6 gains to **more bit collisions at 1,024 bits** — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Cross-benchmark agreement (Kendall tau): multi-assay vs Riniker-Landrum virtual screen **0.68** (0.76 restricted to significant values); single-assay vs multi-assay **0.39**; single-assay vs virtual screen **0.35** — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Verdict: ECFP diameter 4 and 6 (long versions) consistently best; TT the top non-circular fingerprint; ECFP0, RDK7, ECFC0 and MACCS poorest; **path-based RDKit fingerprints and MACCS should be avoided** — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).

**Bender, Jenkins, Scheiber, Sukuru, Glick, Davies (2009), *J. Chem. Inf. Model.* 49(1), 108-119, DOI 10.1021/ci800249s** — "How Similar Are Similarity Searching Methods? A Principal Component Analysis of Molecular Descriptor Space"
- Design: **37 descriptors** compared by repeatedly selecting query compounds, ranking the database, computing Euclidean distances between the rank-orderings, then PCA — [dblp](https://dblp.dagstuhl.de/rec/journals/jcisd/BenderJSSGD09.html); [record](https://data.zhenzhi.com/details/1/867744030).
- Quantitative result relevant here: **four-point pharmacophores (piDAPH4) gave much higher retrieval rates than three-point pharmacophores (piDAPH3): 28.21% vs 19.15%** — [record](https://data.zhenzhi.com/details/1/867744030). (This is the single most concrete 3-point-vs-4-point pharmacophore number I could source.)

**Capecchi, Probst, Reymond (2020), MAP4, *J. Cheminformatics* 12:43** — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/)
- On the Riniker-Landrum small-molecule benchmark, **MAP4 is slightly better than ECFP4, MHFP6 and TT but not significantly so**; MAP4 significantly outperforms AP and MXFP. On the peptide benchmark, atom-pair fingerprints significantly outperform substructure fingerprints, MAP4 is the best atom-pair fingerprint (not significantly), and **MAP4 is the only fingerprint that performs well on both benchmarks** — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/). AUC/BEDROC values appear only in figures and the supplement, so exact numbers are not extractable from the text.
- Resolution result (Human Metabolome Database, 96,456 molecules): **72.9% of molecules have an indistinguishable nearest neighbour under ECFP4-1024** (70,329 molecules), **72.5% under MHFP6-1024** (69,972), **71.1% under TT** (68,623), **only 1,677 (1.7%) under AP**, and **none under MAP4** — [PMC7291580](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).

**Sastry, Lowrie, Dixon, Sherman (2010), *J. Chem. Inf. Model.* 50, 771-784** — "Large-Scale Systematic Analysis of 2D Fingerprint Methods and Parameters to Improve Virtual Screening Enrichments" (Schrödinger Canvas)
- Design: 11 pharmaceutically relevant targets, 8 2D fingerprint methods, 13 atom-typing schemes, 13 bit-scaling rules, 12 similarity metrics = **157,872 virtual screens** — [record](https://data.zhenzhi.com/details/1/913419974).
- Findings: fingerprints capturing local atomic environments (**MOLPRINT2D, Radial, Dendritic**) outperformed the others; **more specific atom typings generally beat generic (i.e. pharmacophore-class) typings**; **small bit spaces such as 1,024 caused many collisions and significantly degraded enrichments** vs larger bit spaces; no single parameter set was optimal across all targets — [record](https://data.zhenzhi.com/details/1/913419974).
- Companion: Duan, Dixon, Lowrie, Sherman, *J. Mol. Graph. Model.* **29** (2010) 157-170 and a *J. Cheminformatics* 3(S1):P1 conference abstract reporting that **MOLPRINT2D gives the highest average retrieval of actives when little is known about the active ligands** — [PMC3083563](https://pmc.ncbi.nlm.nih.gov/articles/PMC3083563). Note: source counts conflict (the conference abstract says 11 atom typings, 7 bit-scaling rules, 4 metrics, 24,068 screens; the JCIM abstract says 13/13/12 and 157,872). Treat the JCIM numbers as authoritative for the 2010 paper.

**Recent (2018-2026) work**
- **Grisoni et al. (2018), WHALES** (*Communications Chemistry* / PMC6220272): WHALES descriptors compared against **seven state-of-the-art representations on ~30,000 compounds and 182 biological targets**, reported to outperform them **in 89% of tested receptors** for scaffold hopping — [PMC6220272](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6220272/).
- **Reutlinger et al. (2013), CATS reprise** (*Mol. Inform.* / PMC3743170): used **Murcko scaffold diversity among the top 1% of retrieved actives** as the scaffold-hopping metric; found **CATS1 to be the most suitable descriptor for scaffold hopping** among those compared; CATS1 clustered with MOE2D and CATS2 clustered with MACCS on that metric — [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/).
- **Vogt/Bajorath-group (2012)** systematically computed scaffold distances in ChEMBL and published **50 activity classes as scaffold-hopping test cases** — [unpaywall DOI record 10.1007/s10822-012-9603-9](https://unpaywall.org/10.1007%2Fs10822-012-9603-9).
- **ErG vs ECFP in a 2025 ML setting** (honey-bee pesticide toxicity, arXiv:2503.24305): ECFP ~**78% AUROC** on a time split vs ErG ~**65%**; on a MaxMin split ECFP ~**72%** vs ErG ~**68%** — ErG trailed ECFP on both; appendix tables give MCC/AUROC/precision/recall per fingerprint — [arXiv:2503.24305](https://arxiv.org/pdf/2503.24305). Caveat: this is a small, single-endpoint ecotoxicology dataset, not a VS benchmark, and the splits are not Bemis-Murcko scaffold splits.
- **scikit-fingerprints (2024)**: tuned 2D fingerprints on MoleculeNet with OGB scaffold splits + Random Forest; **pharmacophore fingerprint omitted due to excessive computation time**; ECFP reached ~**84.6% AUROC on BACE** and ~**70.2% on BBBP** — [arXiv:2407.13291](https://arxiv.org/pdf/2407.13291).
- **Sort & Slice (2024, arXiv:2403.17954)** proposes an unhashed alternative to bit-folding for ECFP, reporting it superior to hash-based folding — relevant because folding is the mechanism that degrades high-cardinality pharmacophore keys — [arXiv:2403.17954](https://arxiv.org/pdf/2403.17954).
- **Metric caveat (2023, *Nat. Commun.* 14)**: AUROC "cannot well capture the true positive rate", a more relevant metric in virtual screening — [Nature Communications paper](https://www.nature.com/articles/s41467-023-41948-6.pdf).

### Inferences

- The field's two most-used modern benchmarks both omit true pharmacophore fingerprints. Any 2013-2026 claim that "pharmacophore fingerprints beat/lose to ECFP4" is therefore extrapolated from FCFP results or from the 2004-2009 literature, not from a current large-scale benchmark.
- The Hert 2004 CATS numbers and the Reutlinger 2013 CATS conclusion are not actually in conflict: Hert measured *recall of actives*, Reutlinger measured *scaffold diversity among retrieved actives*. CATS can retrieve fewer actives while the actives it does retrieve span more scaffolds. Any synthesis must keep these two metrics apart.
- Bender's piDAPH4 > piDAPH3 result (28.21% vs 19.15%) suggests 4-point pharmacophores recover much of the discriminating power that 3-point keys lose, which is consistent with the information-content argument in Q4: adding a point raises feature cardinality and lowers background similarity.

### Gaps

- **No published benchmark compares RDKit's Gobbi Pharm2D fingerprint head-to-head with ECFP4 on DUD/MUV/ChEMBL with reported AUC/BEDROC.** This is the single largest hole in the literature for this question.
- Per-fingerprint numeric tables from Riniker & Landrum 2013 are in supplementary CSVs that I could not parse in this session (the supplementary PDF holds figures only).
- MAP4's exact AUC/BEDROC values are figure-only in the paper text.
- Stiefl et al. 2006's ErG benchmark detail is abstract-level only: ErG "matched or beat" Daylight fingerprints on **10 of 11 MDDR activity classes**, per the abstract — [abstract](https://data.zhenzhi.com/details/1/861823400). I could not retrieve the actual enrichment table, nor the paper's scaffold-hopping figures.
- Hert 2004's scaffold/framework-retrieval results exist only as Figures 4-7; no tabulated percentages are in the author version.

---

## Q3. Do pharmacophore fingerprints outperform ECFP4 for scaffold hopping, and when do they underperform?

### Takeaway

The evidence is split by metric and by dataset composition. On **raw active retrieval**, ECFP4 wins decisively and consistently (Hert 2004: 42.6% vs CATS 19.4% recall@1%). On **scaffold diversity of what is retrieved**, pharmacophore pair descriptors win on activity classes that are themselves scaffold-diverse (Renner & Schneider 2006; Reutlinger 2013) and lose on classes dominated by one chemotype. Crucially, Riniker & Landrum found that scaffold enrichment factor **correlates strongly with general retrieval performance**, and concluded that assessing scaffold-hopping potential this way is "largely uninformative" — which undercuts the standard scaffold-hopping argument for pharmacophore descriptors.

### Cited Findings

- **Renner & Schneider (2006), "Scaffold-Hopping Potential of Ligand-Based Similarity Concepts", *ChemMedChem*, DOI 10.1002/cmdc.200500005**: the **CATS pharmacophore pair descriptor performed best for ligand classes with a high fraction of different scaffolds, while MACCS fingerprints did better on classes with low scaffold diversity**; the two methods were **complementary** in scaffold retrieval — [Zendy record for DOI 10.1002/cmdc.200500005](https://zendy.io/title/10.1002/cmdc.200500005); corroborated in [CDL CATS docs summary](https://cdelib.sourceforge.net/doc/cats.html) and [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/).
- **Reutlinger et al. (2013)**: with Murcko scaffold diversity in the top 1% as the metric, **CATS1 was the most suitable descriptor for scaffold hopping** among the representations compared — [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/).
- **Stiefl et al. (2006), ErG**: against Daylight fingerprints on 11 MDDR activity classes, **ErG matched or beat them in 10 cases**; the paper demonstrates ErG's ability to move between chemotypes and discusses the pitfalls of reduced-graph methods along with solutions — [abstract](https://data.zhenzhi.com/details/1/861823400); [dblp](https://dblp1.uni-trier.de/rec/journals/jcisd/StieflWBZ06.html).
- **Riniker & Landrum (2013), against the scaffold-hopping framing**: 2D fingerprints "have a significant potential to retrieve structurally diverse molecules", *but* **scaffold enrichment factor correlated strongly with general performance**, so the authors "conclude that assessing scaffold-hopping potential this way is largely uninformative" — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- **Hert et al. (2004), same conclusion from the diversity side**: "the relative performance of the various procedures in terms of retrieving chemotypes (and hence in their suitability for scaffold-hopping applications) **mirrors closely the relative performance based on numbers of actives**" — i.e. the fingerprints that retrieved the most actives also retrieved the most distinct molecular frameworks; pharmacophore descriptors did not get a scaffold-hopping bonus — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- **FCFP as the measured middle ground**: Hert et al. report that FCFP_4 "seem to perform relatively better with the more heterogeneous (i.e., low self-similarity) classes, such as the cyclooxygenase and protein kinase C inhibitors", while ECFP_2 "yield better results with the more homogeneous (i.e., high self-similarity) classes, such as the renin inhibitors" — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf). Concretely, FCFP_4A beat ECFP_2A on cyclooxygenase (24.46 vs 23.07 recall@1%, BKD) and on protein kinase C (40.34 vs 37.45), while losing badly on renin (71.14 vs 76.59) — [Hert et al. 2004, Table 1](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- **The condition under which pharmacophore pairs win outright**: CATS's one clear win in Hert 2004 was **renin inhibitors with data fusion, 60.61% recall@1% vs Daylight 51.29, Unity 50.23, Avalon 56.71, BCI 52.96** — a class of large, peptidomimetic, highly self-similar molecules where 1-D property-pair profiles are highly informative. CATS was below ECFP_4A (75.10) even there — [Hert et al. 2004, Table 2](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- **3D-pharmacophore vs 2D-fingerprint precedent**: Hert et al. note "previous comparisons of 2D fingerprints with 3D pharmacophore descriptors have often shown the former to be superior, despite the claimed effectiveness of the latter methods for diversity analysis and similarity searching" — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- **A 3D pharmacophore method that did beat 2D FPs on novel-scaffold retrieval**: FEPOPS "outperformed 2D fingerprints (DAYLIGHT, MACCS, Pipeline Pilot) and a commercial 3D method (Pharmacophore Distance Triplets) at finding novel scaffolds from a single query", and recovered dopamine D2 and retinoic-acid-receptor ligands by scaffold hopping from their endogenous ligands — [Jenkins et al. 2004 abstract records](https://oak.novartis.com/709); [Scilit](https://scilit.com/publications/57e4c4fa98b7b443d9c8095ed7ff51e2). Note the baseline set here excludes ECFP4.
- **WHALES (2018)** claims scaffold-hopping outperformance of 7 state-of-the-art representations in **89% of 182 receptors** across ~30,000 compounds — [PMC6220272](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6220272/).

### Inferences

- The strongest defensible statement for the report: **pharmacophore descriptors do not beat ECFP4 at retrieving actives, and the published evidence that they beat it at scaffold hopping rests on diversity-of-retrieved-actives metrics that two large benchmarks (Hert 2004, Riniker 2013) found to be redundant with retrieval performance.** The honest framing is complementarity (Renner & Schneider's explicit word), not superiority.
- The dataset condition that predicts a pharmacophore win is **high scaffold diversity among the actives combined with a conserved property profile** (renin/peptidomimetics, AT1 antagonists). The condition that predicts a loss is **a tight chemotype series with distinguishing substructures** (5HT3, substance P, cyclooxygenase in Hert 2004).
- FCFP is the pragmatic compromise and is the only pharmacophore-flavoured descriptor with modern large-scale benchmark numbers: it is within ~2-3 recall points of ECFP at diameter 4 (Hert 2004) and sits mid-table in both Riniker-Landrum and O'Boyle-Sayle. For a project that wants pharmacophoric abstraction without abandoning benchmarked performance, FCFP4/FCFC4 is the lowest-risk choice.

### Gaps

- No benchmark I found evaluates Gobbi Pharm2D or ErG on a formal Bemis-Murcko scaffold-split or scaffold-hopping test set (e.g. the Vogt/Bajorath 50-class set) with reported numbers.
- The WHALES "89%" claim's baseline list and metric were not retrievable at the level of detail needed to judge it.
- I found no reliable source comparing pharmacophore descriptors specifically for **separating drugs by pharmacological/ATC class** (as opposed to single-target activity). That is a distinct task from LBVS and is not covered by any benchmark located here.

---

## Q4. Known weaknesses: baseline similarity, size sensitivity, bit collisions, and encoding artefacts

### Takeaway

Four failure modes are documented: (1) **low information content / high background similarity** for low-resolution pair descriptors — CATS at 8.00% bit density vs ECFP_4A at 0.04%; (2) **bit collisions** when high-cardinality keys are folded, measured directly by O'Boyle & Sayle and by Sastry et al.; (3) **sensitivity to molecular size and to count-vs-binary encoding** — Similog's mean bits-set ranges from 1 to 14,740 across MDDR; and (4) **cost**, which in practice forces folding and thereby triggers (2).

### Cited Findings

**Information content / background similarity (Hert et al. 2004, Table 7 — bits set and density across MDDR):**

| Fingerprint | Length | Mean bits set | SD | Max | Min | Density (%) |
|---|---|---|---|---|---|---|
| ECFP_4A | 142,864 | 53.97 | 16.95 | 191 | 8 | **0.04** |
| FCFP_4A | 30,267 | 40.63 | 11.14 | 122 | 7 | 0.13 |
| ECFP_2A | 7,445 | 32.36 | 9.40 | 103 | 5 | 0.44 |
| SimilogB | 64,248 | 863.52 | 900.10 | 10,101 | 1 | 1.34 |
| FCFP_2B | 1,024 | 20.41 | 4.83 | 45 | 5 | 1.99 |
| SimilogA | 64,248 | 1,308.14 | 1,437.17 | 14,740 | 1 | **2.04** |
| ECFP_2B | 1,024 | 31.82 | 9.12 | 98 | 5 | 3.11 |
| FCFP_4B | 1,024 | 39.44 | 10.55 | 113 | 7 | 3.85 |
| ECFP_4B | 1,024 | 52.43 | 15.95 | 177 | 8 | 5.12 |
| **CATS** | **1,200** | **95.99** | **36.18** | **453** | **1** | **8.00** |
| BCI | 1,052 | 96.67 | 30.91 | 264 | 8 | 9.19 |
| Avalon | 2,048 | 285.06 | 149.31 | 1,076 | 16 | 13.92 |
| Daylight | 2,048 | 289.45 | 111.21 | 1,046 | 24 | 14.13 |
| Unity | 988 | 219.65 | 69.16 | 558 | 27 | 22.23 |

Source: [Hert et al. 2004, Table 7](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf). The authors comment that these densities "show a very wide range of levels of molecular description."

- Hert et al. tested whether the poor pharmacophore results were an *encoding* artefact and ruled it out: they re-ran the searches with "the original, occurrence-based vectors" using "the non-binary form of the Tanimoto similarity coefficient and the Floersheim distance", and report that "the use of the integer vectors and the non-binary coefficients **did not improve the recall** of the data fusion searches, and we hence conclude that the use of binary representations does not explain the poor performance of these 2D pharmacophore descriptors" — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf). This is an important negative control: the weakness is intrinsic resolution, not binarisation.
- Hert et al. also had to use a coarser binning scheme for CATS specifically because "a substantial fraction of the keys occurred more than seven times in a molecule" — a direct symptom of a low-cardinality descriptor saturating on drug-sized molecules — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- The SimilogA max/min spread (14,740 vs 1 bits set) is a quantitative measure of **size sensitivity** for triplet descriptors: a large molecule sets thousands of triplet keys and a small one sets a handful, so Tanimoto between a small and a large molecule is structurally forced toward zero — [Hert et al. 2004, Table 7](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).

**Bit collisions from folding**
- O'Boyle & Sayle quantify the damage and attribute it explicitly: the larger multi-assay gains for the diameter-6 fingerprints on going 1,024 -> 16,384 bits are attributed to **"more bit collisions at 1024 bits"**; net differences LECFP6-ECFP6 = +172, LFCFP6-FCFP6 = +96 — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- Hert et al. make the same point for the normalisation schemes: folding "inevitably means that collisions occur, with the result that method-B loses some of the structural information that is retained by method-A; however, the latter representation is dependent on the precise database that is being processed." Measured penalty: ECFP_4A 42.58 -> ECFP_4B 42.26 recall@1% (BKD) — i.e. small for circular FPs — [Hert et al. 2004](https://eprints.whiterose.ac.uk/id/eprint/77602/8/WRRO_77602.pdf).
- Sastry et al. 2010 state it most strongly: **"small bit spaces, such as 1024, caused many collisions and significantly degraded enrichments compared with larger bit spaces"** across 157,872 virtual screens — [record](https://data.zhenzhi.com/details/1/913419974).
- Relevance to Gobbi: RDKit's Gobbi signature is **39,972 bits unfolded with only ~21 set for a small drug** (measured locally; density 0.05%). Folding that to 1,024 or 2,048 bits — the usual practice for ML pipelines — places it in exactly the collision regime Sastry and O'Boyle measured.

**Resolution failures of substructure FPs that pharmacophore/atom-pair descriptors avoid**
- ECFP4 and MHFP6 "have a poor perception of global features such as size and shape, and they miss differences like regioisomers in extended ring systems and linkers of different lengths"; they also "cannot distinguish some scrambled peptides with identical composition" — [Capecchi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).
- HMDB nearest-neighbour degeneracy: **72.9% of 96,456 molecules have an indistinguishable nearest neighbour under ECFP4-1024; 1.7% under AP; 0% under MAP4** — [Capecchi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/). This is the strongest published quantitative case for distance-encoding (pair-type) descriptors over pure substructure keys.

**Cost as a practical weakness**
- Measured locally: Gobbi Pharm2D is **~99x slower than ECFP4** (685 vs 67,900 mol/s, single core, RDKit 2025.09.6).
- Independent: the scikit-fingerprints MoleculeNet benchmark **dropped the pharmacophore fingerprint entirely because of excessive computation time for some molecules** — [arXiv:2407.13291](https://arxiv.org/pdf/2407.13291).
- MAP4's authors concede their Python implementation "is relatively slow" — [Capecchi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7291580/).

**Benchmark-design weaknesses that affect all of these comparisons**
- Riniker & Landrum: "If actives belong to one chemical series, or differ in properties from inactives, most fingerprints can easily separate them"; they recommend a **simple atom-count baseline (ECFC0) as a control** for this bias, and found ECFC0/MACCS "performed no worse than the other fingerprints" on MUV — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- Riniker & Landrum: **differences between targets were larger than differences between fingerprints** — so single-target comparisons of pharmacophore vs structural FPs are statistically weak — [PMC3686626](https://pmc.ncbi.nlm.nih.gov/articles/PMC3686626/).
- O'Boyle & Sayle: cross-benchmark Kendall tau of only **0.35-0.39** between their single-assay benchmark and the virtual-screening benchmark means fingerprint rankings are **task-dependent** — [PMC4932683](https://pmc.ncbi.nlm.nih.gov/articles/PMC4932683).
- AUROC itself "cannot well capture the true positive rate, a more relevant metric in virtual screening" — [Nat. Commun. 14 (2023)](https://www.nature.com/articles/s41467-023-41948-6.pdf).

### Inferences

- There is a clean mechanistic story linking the weaknesses: **low feature cardinality -> high bit density (CATS 8.00% vs ECFP_4A 0.04%) -> high background similarity between unrelated molecules -> poor early recognition.** Hert et al.'s density table and their recall table rank the fingerprints in almost inverse order, which is consistent with this mechanism (the exception is BCI/Daylight/Unity, which are dense but path-rich).
- The Hert integer-vector control experiment is the decisive piece of evidence against the common defence that "pharmacophore FPs just need count-based similarity". It did not help.
- Because the collision penalty scales with feature cardinality, the descriptors most hurt by folding are exactly the 3-point/triplet pharmacophore keys (Gobbi 39,972; Similog 64,248; PharmPrint 10,549). Pair descriptors (CATS 150-1,200; ErG 315) are collision-free but suffer the opposite problem (too few features). There is no folding setting that fixes both.
- Practical recommendation defensible from the evidence: use **unfolded/sparse** Gobbi signatures if using them at all, and never compare a pharmacophore FP against ECFP4 without an ECFC0-style property/size baseline in the comparison.

### Gaps

- I found **no paper that directly measures baseline (background) Tanimoto similarity distributions for Gobbi Pharm2D** on a drug-like library, nor one that measures the size-bias of Gobbi/CATS similarity against heavy-atom count. The density/bits-set data in Hert Table 7 is the closest proxy.
- No peer-reviewed measurement of the Gobbi fingerprint's collision rate at common folded lengths was found.
- No accessible RDKit-blog (rdkit.blogspot.com) or Practical Cheminformatics post on pharmacophore fingerprint benchmarking surfaced in this session's searches; those sources remain unexplored.
