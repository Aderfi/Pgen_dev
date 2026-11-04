# Pharmacophore feature definitions and the overlap/redundancy problem

Scope note: SMARTS and match enumerations below marked "verified locally" were produced by
running RDKit **2025.09.6** from this repo's `.venv`
(`/home/astordna/Projects/dev_Pgen/.venv/lib/python3.14/site-packages/rdkit`). The upstream
sources for those files are
[Data/BaseFeatures.fdef](https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef) and
[rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py).

---

## Q1: RDKit BaseFeatures.fdef — families, overlapping patterns, and what RDKit says about it

### Takeaway

RDKit's `BaseFeatures.fdef` defines **27 feature definitions across 8 families** (`Donor`,
`Acceptor`, `NegIonizable`, `PosIonizable`, `ZnBinder`, `Aromatic`, `Hydrophobe`,
`LumpedHydrophobe`) with **no mutual-exclusion mechanism at all** — the factory returns every
match of every definition, so a benzene ring yields two features on the identical 6 atoms and a
benzoic acid yields 8 features over 9 atoms. Landrum's own comments in the file flag two of the
`PosIonizable` definitions as "too promiscuous?", and the fdef format documentation confirms that
repeating a feature type simply leaves "both definitions ... active" rather than resolving the
conflict.

### Cited Findings

**Families and format**

- `BaseFeatures.fdef` is described in-file as "RDKit base fdef file. Created by Greg Landrum" and
  loads 8 families, verified locally as
  `('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')`
  with `GetNumFeatureDefs() == 27` — [Data/BaseFeatures.fdef](https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef)
- "Chemical features are defined by a Feature Type and a Feature Family." The family is "a general
  classification of the feature (such as 'Hydrogen-bond Donor' or 'Aromatic')" while "the Feature
  Type provides additional, higher-resolution, information about features"; pharmacophore matching
  uses **families**, not types — [RDKit Book, The Feature Definition File Format](https://www.rdkit.org/docs/RDKit_Book.html)
- "Repeating an AtomType results in the two definitions being combined using the SMARTS ',' (or)
  operator" — i.e. OR-union, never intersection or subtraction —
  [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html)
- On duplicate feature types the FAQ states that "both definitions of the HDonor1 feature type will
  be active", and recommends the merged single definition as "more efficient and less confusing" —
  [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html)
- Weights set the feature's position as a weighted average of its defining atoms (so
  `LumpedHydrophobe` ring features and `Aromatic` ring features both sit at the ring centroid) —
  [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html)

**Verbatim SMARTS for the overlap-prone definitions** (all from
[BaseFeatures.fdef](https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef)):

- Donor: `AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]`, plus a **tautomer-driven** extension
  `AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]`, plus `AtomType ChalcDonor [O,S;H1;+0]`;
  `DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]`
- Acceptor: `AtomType Hydroxyl [O;H1;v2]`, `AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]`,
  `AtomType NAcceptor [n;+0;!X3;!$([n;H1](cc)cc)]`,
  `AtomType HalogenAcceptor [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]`;
  `DefineFeature SingleAtomAcceptor [{Hydroxyl},{ChalcAcceptor},{NAcceptor},{HalogenAcceptor}]`.
  Because `Hydroxyl` is unconditional, **every `-OH` is simultaneously a Donor and an Acceptor**,
  including a carboxylic-acid OH.
- NegIonizable: `DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]` — comment: "this one is
  delightfully easy"
- ZnBinder overlaps the acid: `DefineFeature ZnBinder5 [#6]-C(=O)-[O;D1]` with
  `Weights 0,0,1,1` — the carboxylate O's carry weight, so **any carboxylic acid is also a ZnBinder**
- PosIonizable, with Landrum's own doubt in the comments:
  `# imidazole group can be positively charged (too promiscuous?)` /
  `DefineFeature Imidazole c1ncnc1` and
  `# guanidine group is positively charged (too promiscuous?)` /
  `DefineFeature Guanidine NC(=N)N`
- Hydrophobe is **per-atom**: `AtomType Carbon_Polar [#6;$([#6]~[#7,#8,#9])]`,
  `AtomType Carbon_NonPolar [#6;+0;!{Carbon_Polar}]`,
  `DefineFeature ThreeWayAttach [D3,D4;{Carbon_NonPolar}]`,
  `DefineFeature ChainTwoWayAttach [R0;D2;{Carbon_NonPolar}]`. Polarity is judged only on
  **direct** neighbours (`~[#7,#8,#9]`), so the ipso carbon of benzoic acid counts as non-polar.
- LumpedHydrophobe is **per-ring/per-group**: `AtomType Hphobe [c,s,S&H0&v2,Br,I,{Carbon_NonPolar}]`,
  `AtomType RingHphobe [R;{Hphobe}]`, then `RH6_6`/`RH5_5`/`RH4_4`/`RH3_3` ring definitions plus
  `tButyl [C;!R](-[CH3])(-[CH3])-[CH3]` and `iPropyl [CH;!R](-[CH3])-[CH3]`. Note `Hphobe`
  **includes aromatic carbon `c`**, which is what produces the Aromatic/LumpedHydrophobe collision.
- Aromatic rings are ring-size-enumerated (`Arom4`..`Arom8`), e.g.
  `AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]` /
  `DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:...:1`. The in-file comment explains why: large
  patterns "like: a1aaaaaaaa1 will actually match things like 'o1c2cccc2ccc1'", breaking the
  equidistant-centroid assumption, "and 2) It's *way* faster."

**Measured overlap (verified locally, RDKit 2025.09.6, BaseFeatures.fdef)**

| Molecule | atoms | feats | overlap observed |
|---|---|---|---|
| benzene `c1ccccc1` | 6 | 2 | `Aromatic Arom6 (0..5)` **and** `LumpedHydrophobe RH6_6 (0..5)` — identical atom sets |
| toluene `Cc1ccccc1` | 7 | 3 | adds `Hydrophobe ThreeWayAttach (1,)` on the *ring* ipso carbon, already covered by both ring features |
| cyclohexane `C1CCCCC1` | 6 | 1 | `LumpedHydrophobe` only (no per-atom Hydrophobe: all ring CH2 are `D2` but `R1`, excluded by `[R0;D2;...]`) |
| acetic acid `CC(=O)O` | 4 | 5 | `Donor(3)`, `Acceptor(2)`, `Acceptor(3)`, `NegIonizable(1,2,3)`, `ZnBinder5(0,1,2,3)` — the single OH oxygen carries Donor + Acceptor + NegIonizable + ZnBinder |
| benzoic acid `OC(=O)c1ccccc1` | 9 | 8 | the 5 acid features above **plus** `Aromatic(3..8)` + `LumpedHydrophobe(3..8)` + `Hydrophobe ThreeWayAttach(3,)` on the carboxyl-bearing ring carbon |
| imidazole `c1cnc[nH]1` | 5 | 5 | pyridine-type N (atom 2) is **both** `Donor` (via `[$(nc[n;H1])]`) and `Acceptor`; `PosIonizable Imidazole(0..4)` and `Aromatic Arom5(0..4)` share the identical 5 atoms |
| histamine `NCCc1c[nH]cn1` | 8 | 8 | `Donor` on atoms 0, 5 **and** 7; `Acceptor` on 7; `PosIonizable Imidazole(3,7,6,5,4)` co-located with `Aromatic Arom5(3,4,5,6,7)` |
| propranolol `CC(C)NCC(O)COc1cccc2ccccc12` | 19 | 12 | naphthalene alone yields 6 features: 2× `Aromatic`, 2× `LumpedHydrophobe`, 2× `Hydrophobe` on the fusion carbons (13, 18), and atoms 13/18 are each counted in two rings |

- `MinimalFeatures.fdef` **does not exist** in RDKit 2025.09.6: a filesystem-wide `find` returned
  only `BaseFeatures.fdef`, and the GitHub `Data/` listing shows only `BaseFeatures.fdef` among
  `.fdef` files — [github.com/rdkit/rdkit/tree/master/Data](https://github.com/rdkit/rdkit/tree/master/Data)
  (verified locally; see Gaps)
- RDKit ships a **contributed** fdef that exists specifically to fix these overlaps:
  `Contrib/M_Kossner/BaseFeatures_DIP2_NoMicrospecies.fdef`. Its README states it came out of a
  conversation at the Sheffield conference "about the difficulty of defining Positive/Negative
  ionizable features by SMARTS" —
  [Contrib/M_Kossner/README](https://github.com/rdkit/rdkit/blob/master/Contrib/M_Kossner/README)
- That contributed file uses **explicit negative constraints to remove overlap**, which base does
  not: `AtomType AcidicHydroxyl [$([O]C(=[O,S,P]))]` then
  `DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor};!{AcidicHydroxyl}]` (acid OH is no longer a
  Donor); `AtomType NDonor [N&!H0&v3,...;!{AcidicN};!{AmidineTripleN}]`;
  `AtomType NDonor [$([Nv3](-C)(-C)-C);!{AmideN};!{AmidineTripleN}] #Primary Amine but not Amide!`;
  ethers removed from acceptors via `AtomType ChalcAcceptor [O;H0;!$(O=N-*);!$([OD2]([#6])[#6])]`;
  and aromatic oxygen **disabled** with the comment
  `#We will not use aromatic oxygens! cf Leach, Gillet ... J Med Chem 2010,53, 539-558` /
  `#Atomtype ChalcAcceptor [o;+0]` —
  [Contrib/M_Kossner/BaseFeatures_DIP2_NoMicrospecies.fdef](https://github.com/rdkit/rdkit/blob/master/Contrib/M_Kossner/BaseFeatures_DIP2_NoMicrospecies.fdef)
- The same file also adds the ionizable features base lacks (`AcidicN` for acyl/sulfonyl NH,
  `DefineFeature Tetrazole c1nnn[nH,n-1]1`) and cross-references
  `Good, Oprea J.Comput.Aided.Mol.Des (2008) 22:169-178` for similar substructures —
  [same file](https://github.com/rdkit/rdkit/blob/master/Contrib/M_Kossner/BaseFeatures_DIP2_NoMicrospecies.fdef)

### Inferences

- There is **no deduplication layer** in RDKit's `ChemicalFeatures` factory: overlap must be
  engineered into the SMARTS with `!{...}` recursive exclusions (as Kossner does) or removed by
  post-filtering the returned feature list. Choosing `BaseFeatures.fdef` for a 2D pharmacophore
  fingerprint therefore bakes in a systematic, *scaffold-dependent* multiplicity: every aromatic
  ring contributes at least 2 features at the same centroid, every carboxylic acid contributes 4,
  and every fused aromatic system multiplies both.
- The redundancy is **not uniform across chemotypes**, which is the real problem for a fingerprint:
  benzoic acid gets 8 features from 9 atoms (0.89 feat/atom) while cyclohexane gets 1 from 6
  (0.17 feat/atom). For a pair-/triplet-counting 2D pharmacophore fingerprint this means
  aromatic-plus-acid molecules occupy far more bits and dominate Tanimoto denominators regardless
  of actual pharmacology.
- The `Aromatic`/`LumpedHydrophobe` collision is structural, not accidental: `Hphobe` deliberately
  lists aromatic carbon `c` first. Dropping `LumpedHydrophobe` (or dropping `Aromatic`) is the
  cheapest way to halve ring-feature multiplicity, and dropping `ZnBinder` removes a whole family
  that in practice only re-labels carboxylic acids, thiols and phosphonates.
- Landrum's two `(too promiscuous?)` comments are an in-source admission that
  `Imidazole`/`Guanidine` fire on neutral tautomers that may not be protonated at pH 7.4 — and
  they fire on the *whole ring*, co-located with the `Aromatic` feature.

### Gaps

- I could not retrieve a specific rdkit-discuss thread or Greg Landrum blog post discussing the
  Aromatic/LumpedHydrophobe overlap directly; the in-file `(too promiscuous?)` comments and the
  Kossner contrib README are the strongest RDKit-side acknowledgements I could source.
- The assignment presumed a `MinimalFeatures.fdef` in RDKit. I found none in 2025.09.6 (local
  `find /` and the GitHub `Data/` listing). It may have existed in much older releases or be
  confused with the minimal `MinimalLipinski.fdef`-style examples in the docs — **treat the
  "MinimalFeatures vs BaseFeatures" contrast as unverified**.
- The RDKit Book page truncates at ~100k characters in WebFetch; I read the "Feature Definition
  File Format" section but not any later "Chemical Features and Pharmacophores" prose beyond it.

---

## Q2: Gobbi & Poppinger (1998) feature set, as implemented in RDKit's Gobbi_Pharm2D

### Takeaway

RDKit's `Gobbi_Pharm2D` implements **8 families, not 6** — `LH, HD, HA, AR, RR, X, BG, AG` — and
it sidesteps most of the BaseFeatures redundancy by defining ring features as a **single
substituted ring atom** (an "attachment point") rather than a ring centroid, and by making the
acceptor SMARTS exclude hydroxyls bonded to carbonyls. The cost is a blind spot: an unsubstituted
benzene gets **zero** features.

### Cited Findings

- Provenance, verbatim from the module docstring: `""" Definitions for 2D Pharmacophores from:
  Gobbi and Poppinger, Biotech. Bioeng. _61_ 47-54 (1998) """` —
  [Gobbi_Pharm2D.py](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py).
  Note the citation is **Biotechnol Bioeng 61(1):47–54 (1998)**, "Genetic optimization of
  combinatorial libraries", DOI [10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z](https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z)
- Families verified locally: `('LH', 'HD', 'HA', 'AR', 'RR', 'X', 'BG', 'AG')` — i.e. the 6 named
  in the assignment **plus** `RR` (aliphatic ring attachment) and `X` (unusual atom)
- Verbatim SMARTS, all from
  [Gobbi_Pharm2D.py](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py):
  - `Hydrophobic` → Family `LH`:
    `[$([C;H2,H1](!=*)[C;H2,H1][C;H2,H1][$([C;H1,H2,H3]);!$(C=*)]),$(C([C;H2,H3])([C;H2,H3])[C;H2,H3])]`
  - `Donor` → `HD`: `[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]`
  - `Acceptor` → `HA`:
    `[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]`
    — the leading `-[!$(*=[O,N,P,S])]` constraint is exactly the overlap fix that BaseFeatures lacks
  - `AromaticAttachment` → `AR`: `[$([a;D3](@*)(@*)*)]`
  - `AliphaticAttachment` → `RR`: `[$([A;D3](@*)(@*)*)]`
  - `UnusualAtom` → `X`: `[!#1;!#6;!#7;!#8;!#9;!#16;!#17;!#35;!#53]`
  - `BasicGroup` → `BG`:
    `[$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([N,n;X2;+0])]`
  - `AcidicGroup` → `AG`: `[$([C,S](=[O,S,P])-[O;H1])]`
- Topological distance bins, verbatim: `defaultBins = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]`,
  with `SigFactory(featFactory, minPointCount=2, maxPointCount=3)` — a **39,972-bit** signature
  (verified locally via `GetSigSize()`) —
  [Gobbi_Pharm2D.py](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py)
- Measured matches (verified locally, same molecules as Q1):

| Molecule | Gobbi features | vs BaseFeatures |
|---|---|---|
| benzene | **none** | 2 |
| toluene | `AR(1,)` — one atom | 3 |
| cyclohexane | `LH(0,) LH(1,) ... LH(5,)` — 6 per-atom features | 1 |
| n-hexane | `LH(1,) LH(2,) LH(3,) LH(4,)` | — |
| acetic acid | `HD(3), HA(2), AG(1)` — the acid OH is HD **only** | 5 |
| benzoic acid | `HD(0), HA(2), AR(3), AG(1)` | 8 |
| imidazole | `HD(4), HA(2), HA(4), BG(2)` | 5 |
| propranolol | `HD(3), HD(6), HA(3), HA(6), HA(8), AR(9), AR(13), AR(18), BG(3)` | 12 |

### Inferences

- Gobbi's overlap handling is **by construction in the SMARTS**, three distinct mechanisms:
  (1) acceptor patterns carry `-[!$(*=[O,N,P,S])]` so acid/amide heteroatoms are not double-typed;
  (2) basic-group patterns carry `!$([C,a](=O))` so amides are not bases; (3) ring features are
  **single attachment atoms**, not centroids, so a ring contributes one feature per substituted
  ring atom instead of one whole-ring feature *plus* a whole-ring hydrophobe.
- The residual overlaps in Gobbi are deliberate and chemically defensible: a secondary amine is
  `HD + HA + BG` (propranolol atom 3), an `-OH` is `HD + HA`, and an imidazole NH atom is
  `HD + HA`. These are real dual-character atoms, unlike the `Aromatic`+`LumpedHydrophobe`+
  `Hydrophobe` triple that BaseFeatures puts on a single phenyl.
- Despite the name, `LH` in this implementation is **not lumped**: cyclohexane yields 6 separate LH
  features and n-hexane 4, so aliphatic chains/rings still inflate pair and triplet counts roughly
  linearly in chain length. If the goal is a low-redundancy 2D fingerprint, LH is the family that
  most needs post-hoc lumping (e.g. collapse connected LH atoms to one representative).
- Gobbi's `AR` scaling is substitution-driven, not ring-driven: naphthalene in propranolol yields 3
  AR features (one ether-substituted carbon + both fusion carbons), and benzene yields 0. For a
  fingerprint this means **unsubstituted/terminal aromatic rings are invisible**, which is a
  worse failure mode than redundancy for e.g. a plain phenyl terminus — worth checking against
  the drug set before adopting Gobbi unmodified.
- `X` (`UnusualAtom`) is effectively a "P, B, Se, metal, etc." catch-all. For pharmacogenomic drug
  sets with platinum agents, boron-containing drugs (bortezomib), or selenium, this family will
  fire where none of the other 7 do — a useful, non-overlapping discriminator.

### Gaps

- I did **not** obtain the Gobbi & Poppinger 1998 primary text (paywalled, no open copy found), so
  I cannot confirm whether the original paper specified 6 or 8 families, nor whether the
  "attachment point" ring encoding and the acceptor exclusions are Gobbi's or Landrum's
  reimplementation choices. **The 8-family list is verified for RDKit's implementation only.**
  This directly contradicts the "6 features (HA, HD, AR, LH, BG, AG)" premise in the assignment —
  flag it in the report.
- The original paper's own statement about overlap handling is therefore unsourced.

---

## Q3: ErG (Stiefl et al. 2006) and CATS (Schneider 1999) — abstraction as redundancy control

### Takeaway

Both methods attack redundancy by **reducing before typing**: ErG collapses whole ring systems to
single pharmacophoric nodes and permits a node to carry a *combination* of features, while CATS
assigns a small set of coarse PPP types and autocorrelates them over topological distance. ErG
additionally uses "fuzzy" distance spreading instead of hard bins.

### Cited Findings

**ErG**

- Reference: Stiefl, Watson, Baumann, Zaliani, "ErG: 2D pharmacophore descriptions for scaffold
  hopping", *J Chem Inf Model* 2006, 46(1), 208–220 — DOI
  [10.1021/ci050457y](https://doi.org/10.1021/ci050457y) (initials are recorded inconsistently
  across indexes: "Watson, I. A." vs "Watson, T. A." —
  [search record](https://data.zhenzhi.com/details/1/861823400))
- Node typing, quoted from the paper's description: "In ErG, nodes can be one or a combination of
  the following features: hydrogen-bond donor, hydrogen-bond acceptor, positive charge, negative
  charge, hydrophobic group and aromatic ring system. Some featureless nodes also work as links to
  the relevant features. These can be carbon or noncarbon link nodes." —
  [record for Stiefl et al. 2006](https://data.zhenzhi.com/details/1/861823400)
- ErG is explicitly an extension of the Gillet et al. reduced graphs, modified "to better describe
  the pharmacophoric properties, size, and shape of the molecules" —
  [same record](https://data.zhenzhi.com/details/1/861823400)
- RDKit's implementation exposes the fuzzing parameter directly in the signature:
  `GetErGFingerprint((Mol)mol [, (object)atomTypes=0 [, (float)fuzzIncrement=0.3 [, (int)minPath=1 [, (int)maxPath=15]]]])`,
  and a separate `GenerateMolExtendedReducedGraph((Mol)mol [, (object)atomTypes=0]) -> Mol` that
  returns the reduced graph itself (verified locally from `rdkit.Chem.rdReducedGraphs` docstrings;
  source: [rdkit/Chem/ReducedGraphs.py](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/ReducedGraphs.py))
- An open ErG implementation in the Ersilia model hub outputs a **315-dimensional** vector —
  [ersiliaos/eos5guo](https://hub.docker.com/r/ersiliaos/eos5guo)
- ErG + graph edit distance has been used for ligand-based virtual screening and "can identify
  bioactivity similarities in a structurally diverse group of molecules" —
  [record](https://data.zhenzhi.com/details/1/859553012)

**CATS**

- CATS uses **potential pharmacophore point (PPP)** atom typing within an atom-type
  autocorrelation descriptor —
  [Reutlinger et al., CATS review, PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/)
- The five original interaction types are lipophilic, hydrogen-bond donor, hydrogen-bond acceptor,
  positively charged, negatively charged (the D/A/P/N/L set) —
  [Masaryk Univ. chemoinformatics lecture notes](https://is.muni.cz/el/fi/podzim2023/PV290/um/chemoinformatika_prednaska4_descriptory.txt?lang=en)
- A later 4-type variant (lipophilic, aromatic, acceptor, donor) exists, and "CATS1 lacks the
  aromatic atom type" —
  [lecture notes](https://is.muni.cz/el/fi/podzim2023/PV290/um/chemoinformatika_prednaska4_descriptory.txt?lang=en)
- The descriptor is a correlation vector over all pharmacophore points, with **fixed bit positions
  reserved for each feature-pair type at each topological distance** —
  [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/)
- Reutlinger et al.'s CATS implementation applies **"types scaling"**, which "mitigates the
  potential dominance of prevalent pharmacophore feature types", with a maximal correlation
  distance of 10 bonds — [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/)

### Inferences

- "Types scaling" in CATS is the clearest published acknowledgement that **unequal feature
  multiplicity biases the descriptor** — the fix is normalising each pair-type channel by the
  counts of its constituent types, rather than editing the typing rules. This is directly
  transferable: for any count-based 2D pharmacophore fingerprint, dividing a `(T1,T2,d)` bin by
  `n(T1)·n(T2)` removes the "aromatic-acid molecules dominate" effect without touching SMARTS.
- CATS tolerates multi-typed atoms (an atom can be both donor and acceptor) precisely because the
  autocorrelation channels are separate and then scaled — i.e. **redundancy is handled downstream,
  by normalisation, not upstream by exclusion**. This is the opposite design choice from Gobbi
  (upstream SMARTS exclusion) and from Phase's `Exclude` (see Q4).
- ErG's ring-to-node reduction is the strongest structural answer to the
  `Aromatic`/`LumpedHydrophobe`/per-atom-`Hydrophobe` triple-count: a phenyl becomes **one** node
  with an aromatic flag, so multiplicity cannot scale with ring size or fusion. ErG's "one or a
  combination of features" per node is explicitly a *merged* type, not multiple co-located
  features — the redundancy is collapsed at graph-construction time.
- ErG's `fuzzIncrement` spreads a feature-pair's contribution over neighbouring distance bins
  instead of assigning a single hard bin, which removes the binning discontinuity that Gobbi's
  `[(2,3),(3,4),...]` bins introduce.

### Gaps

- I could not retrieve the full text of either primary paper (Stiefl 2006 and Schneider et al. 1999
  *Angew Chem Int Ed* 38:2894–2896, DOI [10.1002/(SICI)1521-3773(19991004)38:19<2894::AID-ANIE2894>3.0.CO;2-F](https://doi.org/10.1002/(SICI)1521-3773(19991004)38:19%3C2894::AID-ANIE2894%3E3.0.CO;2-F) —
  **DOI cited from reference knowledge, not verified by a retrieved page**). The ErG node-type
  quote comes from an indexing site reproducing the abstract, not the paper itself.
- The exact CATS 1999 PPP SMARTS/typing rules, and its explicit statement on atoms qualifying for
  more than one type, are **not sourced**. The Renner, Fechner & Schneider 2006 chapter
  ([bibbase record](https://bibbase.org/network/publication/renner-fechner-schneider-pharmacophoresandpharmacophoresearches-2006))
  is the likely place to find them.
- ErG's exact node-type vocabulary and the number of distance bins behind the 315-dim vector were
  not confirmed from a primary source.

---

## Q4: Commercial feature sets — Phase, MOE, LigandScout, OpenEye Mills-Dean

### Takeaway

The commercial tools all use a ~6-type vocabulary (donor, acceptor, anion/negative,
cation/positive, hydrophobe, ring/aromatic) and — unlike RDKit's BaseFeatures — all ship an
explicit policy for multiply-qualifying atoms: Phase has an `Exclude` flag plus ordered pattern
matching, OpenEye ships two force fields that differ precisely on one overlap case (primary amine
as donor-only vs donor+acceptor), and MOE represents merged types such as `Don&Acc` as first-class
annotations.

### Cited Findings

**Schrödinger Phase**

- "Phase supplies six built-in features"; custom features use letter codes X, Y, Z —
  [Phase Help, Edit Features dialog](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- "With the exception of default hydrophobic features and aromatic rings, the definitions are all
  SMARTS strings" — i.e. hydrophobes and aromatic rings are **group/centroid** definitions, not
  SMARTS — [Phase Help](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- **Overlap mechanism**: patterns are matched in list order, and the `Exclude` option "excludes the
  atoms in this definition from being mapped by other definitions" —
  [Phase Help](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- **Ionization policy in the shipped defaults**: the `Exclude` example is a carboxylic acid removed
  from hydrogen-bond-donor status "because it will be ionized under physiological conditions" —
  [Phase Help](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- Pattern geometry is one of **group, point, or vector**; an aromatic ring is a group feature with a
  projected direction, and a feature can be "an aromatic ring, a donor, an acceptor with one or
  more lone pairs, or none (non-vector feature)" —
  [Phase Help](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- Aromatic/hydrophobe overlap is a **user-selectable option**: "Treat aromatic ring as
  hydrophobic—Treat aromatic rings as if they were hydrophobic features." —
  [Phase Help, Hypothesis Settings](https://sdg26-1.pages.dev/phase/phase_help/hypothesis_settings.html)

**MOE**

- A scheme "sets which attributes are used to build ligand annotation points that can be matched to
  a query"; the **default scheme is Unified**, with EHT (Extended Hückel Theory) as an alternative
  semi-empirical scheme — [MOE pharmacophore course notes](https://dl1.cuni.cz/mod/resource/view.php?id=277228)
- MOE annotation shorthand includes `Don`, `Acc`, `Ani` (anion), `Cat` (cation), `Aro` (aromatic),
  `Hyd` (hydrophobe), **and merged types such as `Don&Acc`** —
  [US 8,618,302 (uses MOE pharmacophore nomenclature)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8618302)
- Annotation points are colour-coded by type (green hydrophobic, purple donor, cyan acceptor,
  orange aromatic) — [MOE course notes](https://dl1.cuni.cz/mod/resource/view.php?id=277228)

**LigandScout**

- The original method defines "a defined set of only six types of chemical features" plus volume
  constraints — Wolber & Langer, *J Chem Inf Model* 2005, 45(1), 160–169, DOI
  [10.1021/ci049885e](https://doi.org/10.1021/ci049885e);
  [record](https://data.zhenzhi.com/details/1/878544680)
- A structure-based LigandScout run lists the features as "aromatic ring, hydrophobic area,
  hydrogen bond donor or acceptor, negative or positive ionisable atom and metal" —
  [PMC5011875](https://pmc.ncbi.nlm.nih.gov/articles/PMC5011875)
- Default distance constraints (attributed to Rarey et al. 1998): hydrophobic interaction 1.0–5.9 Å;
  aromatic ring–ring listed separately for parallel and orthogonal geometries, each 2.8–4.5 Å —
  [Pharmacophores lecture, Univ. Olomouc](https://www.kfc.upol.cz/wp-content/uploads/2023/02/6ADD_09_Pharmacophores_TLanger.pdf)
- Aromatic ring features are **directional** —
  [Olomouc lecture](https://www.kfc.upol.cz/wp-content/uploads/2023/02/6ADD_09_Pharmacophores_TLanger.pdf)

**OpenEye ROCS / Mills-Dean color force field**

- "ROCS comes pre-loaded with two color force fields, Implicit Mills Dean (default) and Explicit
  Mills Dean", stored as `*.cff` files —
  [ROCS Shape and Color theory](https://docs.eyesopen.com/applications/rocs/theory/shape_color.html)
- Six color types, verbatim definitions —
  [ROCS theory](https://docs.eyesopen.com/applications/rocs/theory/shape_color.html):
  - **Donor**: "Functional groups that can act as H-bond donors e.g. acid-OH"
  - **Acceptor**: "Functional groups that can act as H-bond acceptors e.g. carboxylate"
  - **Anion**: "Functional groups with either localized or delocalized negative charge e.g. tetrazole"
  - **Cation**: "Functional groups with either localized or delocalized positive charge e.g. guanidinium"
  - **Hydrophobe**: "Terminal or nonterminal aliphatic groups, including Br and I"
  - **Rings**: "Rings of defined size e.g. 4-7 atoms"
- Each color type is defined by "a set of SMARTS" (patterns not published on that page) —
  [ROCS theory](https://docs.eyesopen.com/applications/rocs/theory/shape_color.html)
- **The explicit overlap policy**: "the Explicit Mills Dean force field allows a primary amine to be
  an acceptor as well as a donor whereas it is a donor only in the Implicit Mills Dean force field"
  — [ROCS theory](https://docs.eyesopen.com/applications/rocs/theory/shape_color.html)
- "Interactions between color atoms of the same type are always attractive"; each type carries an
  attractive/repulsive flag, a weight, and a range term; "ROCS uses only the heavy atoms of
  molecules, hydrogens are ignored" —
  [ROCS theory](https://docs.eyesopen.com/applications/rocs/theory/shape_color.html)

### Inferences

- Two independent design patterns for overlap emerge from the commercial side:
  1. **Exclusion / ordered precedence** (Phase `Exclude` + list order; Implicit Mills-Dean's
     donor-only primary amine). The first matching definition claims the atoms.
  2. **Explicit merged types** (MOE `Don&Acc`). The dual character becomes one annotation, so it
     occupies one slot rather than two.
  RDKit's `BaseFeatures.fdef` uses **neither**, which is the concrete gap to close for a 2D
  fingerprint built on it.
- Phase's hydrophobe/aromatic handling is the direct analogue of the RDKit
  `Aromatic`/`LumpedHydrophobe` collision — except Phase makes it an explicit user toggle that
  defaults to *not* double-counting, whereas RDKit always double-counts.
- All four vendors define hydrophobes as **lumped groups with a centroid** (Phase: non-SMARTS group
  definitions; OpenEye: "terminal or nonterminal aliphatic groups"; LigandScout: "hydrophobic
  area"), not per-atom. RDKit's per-atom `Hydrophobe` family plus the per-atom Gobbi `LH` are
  outliers, and both are the families most prone to count inflation on long aliphatic chains.
- "Rings of defined size e.g. 4-7 atoms" in Mills-Dean matches RDKit's `Arom4`..`Arom8`
  size-enumeration strategy; both exist because a generic ring SMARTS mis-centres fused systems.
- The distinction OpenEye draws (primary amine donor-only vs donor+acceptor) is exactly the
  decision RDKit resolves in favour of "both", by making `ChalcDonor [O,S;H1;+0]` and
  `Hydroxyl [O;H1;v2]` independent and unconditional.

### Gaps

- I did not obtain the **actual Phase letter codes** A, D, H, N, P, R from Schrödinger's own
  documentation. The Phase help page confirms "six built-in features" and the X/Y/Z custom codes
  but does not enumerate the six. **Treat the A/D/H/N/P/R mapping as plausible but unverified.**
- No Phase, MOE, LigandScout or Mills-Dean **SMARTS strings** were retrievable (all behind vendor
  installs/licences). Only the OpenEye *type descriptions* are quotable.
- "MOE PCHD" / "PCH_All" could not be confirmed as a MOE scheme name — searches returned only
  `Unified` and `EHT`. **Do not assert a PCHD scheme without a vendor source.**
- LigandScout's hydrophobic and aromatic feature *definitions* (as opposed to their distance
  constraints) are unsourced; Wolber & Kosara's book chapter 6 is the likely source.
- Mills-Dean's handling of an atom matching two color types is **not documented on the page I
  fetched** beyond the primary-amine example.

---

## Q5: Ionization and tautomer state — neutral or pH 7.4?

### Takeaway

The tools disagree by design: RDKit's `BaseFeatures.fdef` perceives features on whatever
protonation state the input SMILES carries and hedges by firing *both* neutral and ionizable
patterns (and both tautomer donors), while Phase's shipped defaults assume physiological
ionization and remove the acid OH from donors. The consequence is that **ionization state is a
silent, upstream determinant of how much redundancy you get**, and consistent pre-protonation is
the only way to make the perception reproducible.

### Cited Findings

- Phase's default feature set assumes physiological ionization: a carboxylic acid is excluded from
  donor status "because it will be ionized under physiological conditions" —
  [Phase Help, Edit Features](https://sdg26-1.pages.dev/phase/phase_help/edit_features.html)
- RDKit's `BaseFeatures.fdef` handles ionization by **enumerating both states in the SMARTS** rather
  than requiring a protonated input: `AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]` matches both the
  neutral acid and the carboxylate; `ChalcAcceptor` has separate `[O;H0;v2;!$(O=N-*)]` and
  `[O;-;!$(*-N=O)]` entries; `NDonor` covers `N&!H0&v3` and `N&!H0&+1&v4`; and `PosIonizable` has
  both a neutral `BasicGroup` and a charged `PosN [#7;+;!$([N+]-[O-])]` —
  [BaseFeatures.fdef](https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef)
- RDKit's **tautomer hedge** is explicit: `AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]` makes a
  *hydrogen-free* aromatic N a Donor when an adjacent/1,3 aromatic N carries an H — verified
  locally, this is why imidazole atom 2 is both Donor and Acceptor, and why histamine gets 3
  Donors — [BaseFeatures.fdef](https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef)
- The Kossner contrib fdef's filename and README frame this as the **microspecies problem**
  (`BaseFeatures_DIP2_NoMicrospecies.fdef`), motivated by "the difficulty of defining
  Positive/Negative ionizable features by SMARTS" —
  [Contrib/M_Kossner/README](https://github.com/rdkit/rdkit/blob/master/Contrib/M_Kossner/README)
- **Dimorphite-DL** is an open-source enumerator of ionization states that "uses a straightforward
  empirical algorithm that leverages substructure searching" against experimentally characterised
  ionizable molecules, using "the µ and σ values associated with each ionizable moiety" —
  Ropp et al., *J Cheminform* 2019, DOI [10.1186/s13321-019-0336-9](https://doi.org/10.1186/s13321-019-0336-9)
  ([PDF](https://jcheminf.biomedcentral.com/counter/pdf/10.1186/s13321-019-0336-9.pdf))
- Its default window is pH **6.4–8.4** ("considered biologically relevant pH"), so 7.4 is interior;
  setting min = max targets a single pH — [Ropp et al. 2019](https://jcheminf.biomedcentral.com/counter/pdf/10.1186/s13321-019-0336-9.pdf)
- Output logic, verbatim: "If rangePKA is entirely less than rangepH, dimorphite-DL outputs a
  deprotonated molecule. If rangePKA is entirely greater than rangepH, dimorphite-DL outputs a
  protonated molecule. If rangePKA and rangepH overlap, dimorphite-DL outputs both deprotonated and
  protonated molecules." — [Ropp et al. 2019](https://jcheminf.biomedcentral.com/counter/pdf/10.1186/s13321-019-0336-9.pdf)
- **Practical pitfall**: with default precision Dimorphite-DL widens to pKa ± 1 and can return
  *both* forms for groups whose pKa sits near the target pH; one wrapper therefore sets precision
  to 0 and caps output at one variant —
  [pestifer protonation module](https://pestifer.readthedocs.io/en/latest/api/pestifer.charmmff.ligand_paramgen.protonation.html)
- **Known limitation**: "Dimorphite-DL deprotonates indoles and pyrroles around pH 14.5. But these
  substructures can also be protonated around pH -3.5. Dimorphite does not perform the protonation."
  — [dimorphite-dl on PyPI](https://pypi.org/project/dimorphite-dl)
- Tautomer/protomer enumeration is treated as a distinct, additional step from protonation in
  practice — [Auto3D tautomer and protomer analysis](https://auto3d.readthedocs.io/en/latest/example/tautomer_protomer_analysis.html)

### Inferences

- Because RDKit enumerates both ionization states in its SMARTS, **feeding it protonated input does
  not reduce feature count** — it only shifts which pattern fires. The acid case is the clearest:
  `AcidicGroup` matches both `-C(=O)OH` and `-C(=O)[O-]`, so you get `NegIonizable` either way,
  but the neutral form *additionally* gets `Donor` (via `ChalcDonor [O,S;H1;+0]`) and `Acceptor`
  (via `Hydroxyl [O;H1;v2]`) on the same oxygen. Protonating to the carboxylate at pH 7.4
  therefore removes **one spurious Donor per acid** and is the single highest-value preprocessing
  step for reducing BaseFeatures redundancy. (Inference from the SMARTS; I did not find a
  published measurement of this effect.)
- Conversely, protonating a basic amine to `[NH3+]` loses the `BasicGroup` match (which requires
  `+0`) but gains `PosN`, and — because `NDonor` covers `N&!H0&+1&v4` — keeps the Donor. So the
  amine's feature count is roughly state-invariant in RDKit while the acid's is not. **Protonation
  changes redundancy asymmetrically across functional groups**, which is precisely the kind of
  chemotype-dependent bias that corrupts a similarity measure.
- For a reproducible pipeline the defensible recipe is: canonicalise → pick **one** dominant
  microspecies at pH 7.4 (Dimorphite-DL with min = max = 7.4 and reduced precision, or Epik/QUACPAC
  if licensed) → fix the tautomer (RDKit `TautomerEnumerator.Canonicalize`) → *then* perceive
  features. Enumerating multiple protomers/tautomers and unioning their features reintroduces
  exactly the multiplicity the exercise is trying to remove.
- Dimorphite-DL's indole/pyrrole limitation matters for drug sets: those NH's stay as donors, which
  is the chemically right answer at 7.4 anyway.

### Gaps

- I found **no source that directly recommends a protonation tool as a prerequisite for
  pharmacophore perception** — the Dimorphite-DL literature is about ionization enumeration in
  general, and the connection to pharmacophore feature perception is my inference, not a cited
  recommendation.
- No Epik or OpenEye QUACPAC documentation was retrieved; their pH-dependent state-generation
  behaviour is unsourced here.
- Whether RDKit's `GetFeaturesForMol` is documented anywhere as state-agnostic-by-design (vs. it
  just being an emergent property of the SMARTS) is unconfirmed.

---

## Q6: Published discussion of redundant/overlapping features inflating similarity

### Takeaway

The strongest quantitative work is on **2D substructure** fingerprints, not pharmacophore
fingerprints: related/collinear bits in MACCS and PubChem demonstrably shift Tanimoto scores and
"bias the outcome of molecular similarity". For pharmacophore descriptors specifically, the
published responses are CATS's "types scaling" (Q3) and per-feature weighting, and I found no paper
that quantifies redundancy bias in a pharmacophore fingerprint directly.

### Cited Findings

- Related fingerprint bits — "such as those in MACCS and PubChem schemes" — can substantially change
  Tanimoto scores: "their presence in the feature set could have substantial effects on the
  similarity scores and bias the outcome of molecular similarity". The authors quantify relatedness
  with an **eigenvalue-entropy** measure, and their illustrative case has "perfect multicollinearity
  in the first four features" —
  *J Cheminform* 2021, DOI [10.1186/s13321-021-00506-2](https://doi.org/10.1186/s13321-021-00506-2)
  ([PMC7989080](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7989080/))
- Molecular-complexity bias is the adjacent, well-documented effect: complex molecules have higher
  fingerprint bit density and produce artificially high similarity values —
  [Bonn dissertation on molecular complexity effects and fingerprint-based similarity search](https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/4158?show=full)
- Equal weighting of pharmacophore features is the standard but questioned default: "Methods of
  three-dimensional molecular alignment generally treat all pharmacophore features equally when
  superimposing. However, some pharmacophore features can be more important in a specific system."
  — *Mol Divers* 2020, DOI [10.1007/s11030-019-09961-4](https://doi.org/10.1007/s11030-019-09961-4)
- When a molecule carries **several features of the same type**, aggregate color scores lose the
  information about where each one sits, which motivated computing per-atom color overlaps instead
  — [ROCS-Derived Features for Virtual Screening, arXiv:1606.01822](https://arxiv.org/pdf/1606.01822)
- Classical atom-centred pharmacophore *quartet* fingerprints omit information from extension
  points, and adding those as independent feature types "adds complexity" —
  [Abrahamian et al., Efficient generation, storage, and manipulation of fully flexible pharmacophore multiplets](https://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Abrahamian2003Efficient.pdf)
- CATS's countermeasure, restated here for Q6: "types scaling ... mitigates the potential dominance
  of prevalent pharmacophore feature types" —
  [PMC3743170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3743170/)

### Inferences

- The 2021 eigenvalue-entropy result transfers directly to the RDKit `BaseFeatures` case, and the
  transfer is tight rather than loose: `Aromatic Arom6(0..5)` and `LumpedHydrophobe RH6_6(0..5)` on
  benzene are **perfectly collinear** on any molecule whose only hydrophobic rings are aromatic —
  exactly the "perfect multicollinearity in the first four features" pathology the paper uses as
  its worked example. The same holds for `NegIonizable AcidicGroup` and `ZnBinder5` on carboxylic
  acids. A 2D pharmacophore fingerprint built on all 8 BaseFeatures families thus contains pairs of
  channels that are near-duplicates by construction, which inflates the `a` (shared bits) term of
  Tanimoto for any two aromatic or any two carboxylic-acid-bearing molecules, regardless of target
  relevance.
- Combining the complexity-bias finding with the measured feat/atom ratios in Q1 (0.89 for benzoic
  acid vs 0.17 for cyclohexane) suggests the two effects **compound** for BaseFeatures: aromatic
  acids are both more complex *and* more redundantly typed.
- Three mitigation strategies are supported by the sources, in increasing invasiveness:
  (1) **drop collinear families** (`LumpedHydrophobe` or `Aromatic`; `ZnBinder` outright) — the
  Phase "treat aromatic ring as hydrophobic" toggle and Implicit Mills-Dean's donor-only amine are
  precedents; (2) **types scaling / per-channel normalisation** — CATS precedent; (3) **reduce
  before typing** so multiplicity cannot scale with ring size — ErG precedent.

### Gaps

- **No paper quantifies redundancy bias in a pharmacophore fingerprint specifically.** The
  eigenvalue-entropy study covers MACCS/PubChem 2D bits. The report should state this transfer as
  an inference, not a cited result.
- I found no published head-to-head benchmark of `BaseFeatures.fdef` vs `Gobbi_Pharm2D` vs a
  deduplicated fdef on a retrieval task, so there is **no citable recommendation** of one feature
  set over another for 2D pharmacophore fingerprints.
- No rdkit-discuss thread or Landrum blog post on this was retrieved (see Q1 Gaps).

---

## Summary guidance for 2D pharmacophore fingerprints (inference-level, flagged as such)

Supported by the cited design choices above, but **not itself a cited recommendation**:

1. **Prefer `Gobbi_Pharm2D` over `BaseFeatures.fdef`** for a 2D pharmacophore fingerprint. It is
   the only feature set RDKit ships that was designed *for* 2D pharmacophore signatures (it comes
   with a `SigFactory`, 2–3 point counts and distance bins), and its SMARTS carry the overlap
   exclusions BaseFeatures lacks. Measured redundancy is roughly half (benzoic acid: 4 features vs
   8). Verified locally; sources in Q1/Q2.
2. **If using `BaseFeatures.fdef`, drop families before use.** Dropping `LumpedHydrophobe` removes
   the perfectly-collinear ring duplicate; dropping `ZnBinder` removes a family that in practice
   only re-labels acids/thiols/phosphonates. RDKit's `SigFactory` supports restricting families, so
   this needs no fdef editing.
3. **Patch Gobbi's `LH` with post-hoc lumping** — collapse connected `LH` atoms to one
   representative — since cyclohexane currently yields 6 features and n-hexane 4 (verified
   locally). Vendors all lump hydrophobes (Q4).
4. **Check Gobbi's benzene blind spot against your actual drug set** before adopting it: an
   unsubstituted aromatic ring produces no `AR` feature at all (verified locally). If terminal
   phenyls matter, add a centroid-style aromatic definition.
5. **Fix one protonation state and one tautomer before perception**, since RDKit's tautomer-hedging
   `NDonor [$(nc[n;H1])]` and its dual-state acid patterns otherwise make the feature count depend
   on how the input SMILES happened to be written (Q5).
6. **Normalise count channels (CATS-style "types scaling")** rather than relying on raw bit counts,
   to stop prevalent families dominating the similarity (Q3, Q6).
7. **If redundancy is still a problem, move to ErG**, which eliminates ring-size and fusion
   multiplicity structurally by reducing each ring system to one node, and is available in RDKit as
   `rdkit.Chem.rdReducedGraphs.GetErGFingerprint` (Q3).
