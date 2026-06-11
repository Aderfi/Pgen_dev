# ADMET prediction tools — reference for Pharmagen

> Notes for step **(c)** of the drug-tower enrichment: predicting ADMET / CYP
> endpoints from SMILES to feed the drug tower with a **pharmacokinetic /
> enzyme-interaction profile** (the most PGx-causal signal). Companion to the
> feature work in `src/data/library/drugs.py` and the architecture in
> `src/model/architectures/gnn.py`.

---

## 1. What matters for our case

We build the graph library programmatically from ~109k SMILES, so the selection
filter is:

1. **Local + batch** — a Python API or CLI that runs offline over 109k molecules,
   not a click-by-click web form.
2. **Predicts CYP** — ideally **substrate** (the PGx-causal relation), not only
   inhibition.
3. **Returns probabilities** — keep the model's uncertainty; don't stack hard
   binary error.
4. **Reproducible / cacheable** — compute once at library-build time and persist
   on the `.pt` graph (e.g. an `admet_feats` attribute alongside `global_feats`).

**Why ADMET at all?** The molecular graph encodes *structure*; the PGx phenotype
depends on how the drug *interacts with the variant gene product* (metabolism,
transport, target). A predicted ADMET / enzyme profile gives the drug tower that
interaction signal directly instead of asking the GNN to infer it from atoms.

---

## 2. Substrate ≠ inhibitor (read this first)

For pharmacogenetics the causal relation is usually **substrate**: a CYP2D6 poor
metabolizer changes the exposure of drugs that are CYP2D6 *substrates*. But most
public models predict **inhibition** (far more PubChem bioassay data exists), and
fewer predict substrate well.

- **Substrate (PGx-causal):** CypReact (most specialised), ADMETlab 3.0,
  admetSAR, pkCSM, **and ADMET-AI** (CYP2C9/2D6/3A4 via the CarbonMangels
  datasets — see the evaluation in §8).
- **Inhibition:** ADMET-AI (5 isoforms, Veith), SwissADME.

> **Correction (verified empirically, §8):** ADMET-AI is *not* inhibition-only.
> Its output includes `CYP2C9/2D6/3A4_Substrate_CarbonMangels`, so it covers the
> core PGx-causal substrate signal for the three most pharmacogenetically
> relevant isoforms. This is why Pharmagen uses ADMET-AI alone for step (c) and
> defers CypReact.

The ideal enzyme profile combines both: *is it a CYP2D6 substrate?* (causal for
metabolizer phenotype) **and** *is it an inhibitor?* (relevant for drug–drug
interactions).

---

## 3. Tool comparison

| Tool | Predicts | Access | CYP coverage | Fit |
|---|---|---|---|---|
| **ADMET-AI** (TDC / Stanford) | ~40 ADMET endpoints (absorption, BBB, PPB, clearance, half-life, hERG, AMES, DILI, …) | `pip install admet-ai`, **local batch** + web | 5 CYP **inhibition** | ⭐ best overall fit |
| **ADMETlab 3.0** | ~100+ endpoints, very comprehensive | Web (batch CSV) + API | 5 CYP **inhibitor + substrate** | Strong (substrate) |
| **admetSAR 3.0** | Broad ADMET | Web + partial download | CYP **substrate + inhibitor** | Good (substrate) |
| **pkCSM** | Full ADMET | Web (limited batch) | CYP substrate (2D6, 3A4) + inhibitor | Good, web-bound |
| **SwissADME** | Physchem + PK + BOILED-Egg | Web, fast, no login | 5 CYP (**inhibition only**, yes/no) | Convenient, no substrate |
| **CypReact** | **Substrate of the 9 human CYPs** | Java, local | Substrate-specialised | ⭐ for the PGx signal |
| **SMARTCyp / FAME3 / XenoSite** | **Site of metabolism** (which atom) | Java / CLI / web | Atom-level CYP metabolism | Fine-grained complement |
| **OPERA** (NIH / EPA) | Physchem + ADME/tox QSAR | CLI / Java, local | Limited | Gives applicability domain + confidence |

---

## 4. Per-tool notes

### ADMET-AI  ⭐
- Chemprop **D-MPNN** (graph neural net) models trained on the Therapeutics Data
  Commons (TDC) ADMET benchmark, plus RDKit physicochemical descriptors.
- ~40 endpoints across A/D/M/E/T, including CYP1A2, CYP2C9, CYP2C19, CYP2D6,
  CYP3A4 **inhibition** (from the TDC CYP inhibition datasets).
- **Local Python package**, fast batch (thousands of molecules quickly); also a
  hosted web server. Permissive license.
- *Limitation:* CYP endpoints are inhibition, not substrate.

### ADMETlab 3.0
- Very broad endpoint set; explicitly includes CYP **inhibitor and substrate**
  for the major isoforms, plus medicinal-chemistry and applicability-domain flags.
- Primarily a web server with **batch CSV upload** and an API; not a local library.

### admetSAR 3.0
- Long-standing ADMET predictor with **CYP substrate and inhibitor** models and
  many toxicity endpoints. Web-first, some downloadable components.

### pkCSM
- Graph-signature based ADMET; predicts CYP **substrate** (CYP2D6, CYP3A4) and
  several **inhibitor** endpoints. Web server, batch is limited.

### SwissADME
- Fast, no login. Physchem, PK, GI absorption + BBB (BOILED-Egg), Pgp substrate,
  and 5-isoform CYP **inhibition** (yes/no). No substrate prediction. Batch via a
  pasted molecule list (hundreds, not 100k).

### CypReact  ⭐ (substrate)
- Specifically predicts whether a molecule is a **substrate (reactant) of the 9
  major human CYPs**. Java, local, free — the most direct source of the
  PGx-causal substrate flags that ADMET-AI lacks.

### SMARTCyp / FAME3 / XenoSite
- **Site-of-metabolism** predictors (which atom is metabolised), CYP-aware.
  SMARTCyp is ligand-based and fast (Java); FAME3 covers CYP + other enzymes;
  XenoSite is a web/ML model. Useful if we later want atom-level metabolism
  features rather than molecule-level flags.

### OPERA
- NIH/EPA QSAR suite for physchem + ADME/tox, CLI/Java, with **applicability
  domain and confidence** per prediction. CYP coverage is limited but the AD/
  confidence outputs are valuable as features or filters.

### Therapeutics Data Commons (TDC)
- Not a tool but the **benchmark datasets** behind ADMET-AI. Relevant if we ever
  want to train our own Chemprop ADMET models on specific endpoints.

---

## 5. Recommendation for Pharmagen

1. **ADMET-AI as the workhorse** — pip, local, fast batch, D-MPNN. One call gives
   a broad ~40-endpoint ADMET vector (absorption, distribution, clearance,
   toxicity, 5 CYP inhibition) that enriches the drug tower beyond structure.
2. **CypReact (or ADMETlab 3.0 batch) for the CYP *substrate* profile** — the
   flags that most move the needle for PGx, which ADMET-AI does not cover.
3. *(Optional, fine-grained)* **SMARTCyp / FAME3** for site-of-metabolism if we
   later want atom-level metabolism signal.

**Two viable routes:**
- **100% local & reproducible (recommended):** ADMET-AI (pip) + CypReact (jar).
  Fully offline, batchable over the 109k, no web dependency or rate limits.
- **More endpoints, with a server:** ADMETlab 3.0 (substrate + inhibitor + AD)
  via its batch/API, accepting an external dependency and rate limits.

---

## 6. Integration notes (when we build it)

- **Compute once at library-build time** and persist on the `.pt` graph as a
  separate attribute (e.g. `admet_feats`), kept **distinct from `global_feats`**
  so the chemical descriptor (ECFP / physchem) and the predicted pharmacokinetic
  profile stay decoupled. A second model branch would consume it in parallel to
  the existing global branch in `PharmagenTwoTower`.
- **Store probabilities, not binarised classes.**
- **Persist applicability-domain / confidence** when the tool provides it (OPERA,
  ADMETlab) — usable as a feature or to down-weight unreliable predictions.
- **Dim bookkeeping** would follow the same pattern as `DRUG_GLOBAL_DIM`: a
  constant in `drugs.py`, a `models.toml` key, and the `extract_tower_dims` /
  `GraphDims` / `DEFAULT_DIMENSIONS` chain.
- **Licenses:** ADMET-AI (permissive, local), CypReact (free, Java),
  SwissADME / pkCSM / ADMETlab (free for academic use) — all fine for a research
  project. Verify terms before any redistribution.

---

## 7. Starting points (verify URLs before use)

- ADMET-AI — `github.com/swansonk14/admet_ai` · web: `admet.ai`
- Therapeutics Data Commons — `tdcommons.ai`
- ADMETlab 3.0 — `admetlab3.scbdd.com`
- admetSAR — `lmmd.ecust.edu.cn/admetsar`
- pkCSM — `biosig.lab.uq.edu.au/pkcsm`
- SwissADME — `swissadme.ch`
- CypReact — search "CypReact Tian substrate CYP GitHub"
- SMARTCyp — `smartcyp.sund.ku.dk`
- FAME3 — NERDD server, `nerdd.univie.ac.at`
- OPERA — `github.com/NIEHS/OPERA`

---

## 8. Evaluation on the real catalog (2026-06-11)

ADMET-AI 2.0.1 was evaluated on random samples of the real
`data/dicts/cid_smiles_dict.json` (~109k SMILES) before integrating — see
`scripts/eval_admet_ai.py`.

**Findings**

- **Coverage: 100 %.** 0 NaN across 200 + 1000 random molecules; the D-MPNN
  always returns a prediction (RDKit-parseable input assumed).
- **Output: 53 base endpoints + 53 DrugBank-approved percentiles** (106 columns).
- **CYP: substrate *and* inhibition.** `CYP2C9/2D6/3A4_Substrate_CarbonMangels`
  (substrate, PGx-causal) **plus** `CYP1A2/2C19/2C9/2D6/3A4_Veith` (inhibition).
  Distributions are biologically sane (e.g. CYP3A4-substrate mean ≈ 0.63 — most
  drugs are 3A4 substrates).
- **Overlap:** MolWt, LogP, TPSA, HBD, HBA, QED, stereo centres and Lipinski
  duplicate the QSAR block already in `global_feats` — excluded from `admet_feats`.
- **Throughput:** ~0.13 s/mol inference on an RTX 4070 Ti SUPER (GPU) plus a
  one-time ensemble load (~3 min) ⇒ **~3–4 h one-time for 109k**, cached.

**Decision:** ADMET-AI alone for step (c); **CypReact deferred** (ADMET-AI already
covers the 2C9/2D6/3A4 substrate signal).

## 9. Implemented integration (`src/data/library/admet.py`)

A curated **41-endpoint** profile (`DRUG_ADMET_DIM`) is attached to every drug
graph as `admet_feats` `[1, 41]`, **decoupled from `global_feats`**:

| Block | n | Endpoints |
|---|---|---|
| Absorption | 8 | HIA, bioavailability, solubility, lipophilicity, hydration FE, Caco-2, PAMPA, P-gp |
| Distribution | 3 | BBB, PPBR, VDss |
| Metabolism | 8 | CYP1A2/2C19/2C9/2D6/3A4 inhibition + CYP2C9/2D6/3A4 substrate |
| Excretion | 3 | hepatocyte clearance, microsome clearance, half-life |
| Toxicity | 19 | hERG, AMES, DILI, ClinTox, carcinogenicity, skin, LD50, Tox21 (12) |

- **Representation:** classification endpoints keep their raw probability `[0,1]`;
  the 10 regression endpoints use the DrugBank-approved **percentile / 100**
  (bounded, distribution-aware). No binarisation.
- **Pipeline:** `AdmetProvider.from_records(...)` runs ADMET-AI once over the
  catalog (canonicalised on the same largest-fragment moiety the graph uses) and
  caches the table to `data/library/admet_profile.parquet`; the `DrugGraphBuilder`
  then attaches each `cid`'s vector. Missing CIDs get a zero vector (tallied).
- **Model:** `PharmagenTwoTower` consumes `admet_feats` in a parallel
  `drug_admet_mlp` branch fused alongside the `global_feats` branch into the drug
  embedding (single `drug_fuse`), keeping `embedding_dim` and the heads unchanged.
- **Dim chain (sources of truth):** `admet.DRUG_ADMET_DIM` →
  `models.toml: drug_admet_features` → `extract_tower_dims` → `cache.GraphDims` →
  `datasets.DEFAULT_DIMENSIONS` → `build_gnn_model` → `create_gnn_model`.
- **CLI:** `python -m src.data.library` builds ADMET by default;
  `--skip-admet` (zero profiles, no GPU) and `--force-admet` (recompute cache).

**Deferred:** CYP *substrate* via CypReact (ADMET-AI's CarbonMangels substrate
endpoints already cover the core signal); site-of-metabolism (SMARTCyp/FAME3).
