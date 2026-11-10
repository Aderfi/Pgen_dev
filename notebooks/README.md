# Pharmagen demo notebooks

Four notebooks that run the finished parts of the project, on Google Colab or a
local Jupyter. Read them in order.

| Notebook | Runs | Runtime |
|---|---|---|
| `01_overview.ipynb` | Architecture, methodology, environment, repository map | CPU |
| `02_library_build.ipynb` | A live drug-library build, ADMET-AI, the HGVS catalog | CPU |
| `03_graphs.ipynb` | Both towers' real graph inputs, drawn | CPU |
| `04_kg_and_evaluation.ipynb` | Knowledge graph, training rows, splits, metrics | CPU |

None of them needs a GPU. Training is not demonstrated: the drug library was
rebuilt recently, so every checkpoint has to be retrained from scratch.
`04` shows the evaluation protocol and the commands that launch a run.

## How the environment works

Colab runs Python 3.13; Pharmagen and Modephere need 3.14. `colab_setup.py`
therefore keeps the notebook kernel as a driver only: it installs `uv`, lets it
provision Python 3.14 with the locked dependencies, unpacks the data bundle, and
starts a second kernel inside the project's own environment.

Cells marked `%%pgen` run in that 3.14 kernel and relay back whatever they print
or display, figures included. State persists between `%%pgen` cells, exactly as
in a normal kernel. `%pgen_restart` restarts it.

## Running on Colab

1. Build the data bundle once, on a machine with the project checked out:

   ```bash
   uv run python scripts/pack_colab_bundle.py
   ```

   This writes `pharmagen_colab_bundle.zip` (~190 MB): the Modephere compound
   store, the ADMET profiles, `hgvs_catalog.pt`, the PGx exports, the knowledge
   graph snapshot, the staged ChEMBL release, the evaluation splits and the
   reports. `data/` and `reports/` are git-ignored, so a fresh clone has none of
   this.

2. Upload the zip to Google Drive, or publish it as a GitHub release asset.

3. Open a notebook in Colab and set `BUNDLE` in the first cell to either the
   Drive path (the default, `/content/drive/MyDrive/pharmagen_colab_bundle.zip`)
   or an https URL. Then run the cells in order.

The first cell clones the repository and provisions Python 3.14, which takes a
few minutes on a cold Colab runtime. Later notebooks in the same session reuse
the environment and skip straight to the kernel.

## Running locally

Nothing is installed and nothing is downloaded: `BUNDLE` is ignored and the
kernel is started over this checkout's own environment and data.

```bash
uv run --with jupyterlab --with ipykernel jupyter lab notebooks/
```

The notebooks resolve the repository root from `notebooks/`, so open them from
this directory.
