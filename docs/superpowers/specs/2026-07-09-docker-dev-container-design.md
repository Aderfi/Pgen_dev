# Design — Pharmagen GPU dev container

**Date:** 2026-07-09
**Status:** Approved (pending spec review)

## Goal

A single Docker image that runs the *entire* Pharmagen workflow — dependency
install, training, inference, tests, and the FastAPI service — with NVIDIA GPU
(CUDA 13.0) access. The user does all development through the container. Code is
edited on the host and bind-mounted live; the Python virtual environment is
built into the image at build time.

## Non-goals

- No CPU-only variant (deps are pinned to CUDA 13.0 `cu130` wheels).
- No native Windows container. On Windows the same Linux image runs via Docker
  Desktop + WSL2. This is documented, not built.
- No production/deploy image with baked-in code (dev-only, bind-mounted).

## Decisions (from brainstorming)

| Axis | Choice |
|---|---|
| Purpose | General dev image — install/train/infer/test/serve all in-container |
| GPU | GPU, CUDA 13.0 (matches pinned `torch==2.11.0` cu130 wheels) |
| Base OS | Ubuntu 24.04 via `nvidia/cuda:...-cudnn-runtime-ubuntu24.04` |
| Python 3.14 | Installed by `uv` (standalone), not the base-OS python |
| Workflow | Repo bind-mounted live + docker-compose |
| Venv | Built in Dockerfile at `/opt/venv` (outside the bind-mount) |
| Default CMD | `bash` dev shell; API launched manually or via compose override |
| Persistence | Host `./data` (+ `./checkpoints` if used) via the repo bind-mount; only extra named volume is `uv-cache` |

## Architecture — 3 files

### 1. `Dockerfile`

- `FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`
- **System packages** (apt, single layer, `--no-install-recommends`, clean lists):
  - Build/util: `git curl ca-certificates build-essential`
  - pysam / htslib runtime: `libcurl4 zlib1g libbz2-1.0 liblzma5 libdeflate0`
  - torch OpenMP: `libgomp1`
  - rdkit shared libs: `libxrender1 libxext6 libsm6`
- **uv**: copied from the official `ghcr.io/astral-sh/uv:latest` image
  (`COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/`).
- **Env**:
  - `UV_PROJECT_ENVIRONMENT=/opt/venv` — venv lives outside `/workspace`
  - `UV_PYTHON_INSTALL_DIR=/opt/uv-python`
  - `UV_LINK_MODE=copy`
  - `PATH="/opt/venv/bin:$PATH"`
- `RUN uv python install 3.14` — Python 3.14 standalone toolchain.
- **Dependency layer** (cache-friendly): `WORKDIR /workspace`, then
  `COPY pyproject.toml uv.lock ./`, then
  `RUN --mount=type=cache,target=/root/.cache/uv uv sync --extra dev --frozen --no-install-project`.
  Only `pyproject.toml`/`uv.lock` changes bust this layer — code edits do not
  re-download the large cu130 wheels.
  - `--no-install-project` because the project source is bind-mounted at runtime,
    not present at build. `uv sync` still builds the full venv at `/opt/venv`.
- `WORKDIR /workspace`
- `CMD ["bash"]`

Rationale for `/opt/venv`: the compose bind-mount `.:/workspace` would hide any
`/workspace/.venv`. Placing the venv at `/opt/venv` keeps the build-time venv
usable at runtime while allowing live code editing.

### 2. `docker-compose.yml`

- One service, `pharmagen`:
  - `build: .`
  - GPU: `gpus: all` (Compose GPU support) — or the `deploy.resources.reservations.devices` NVIDIA form for older Compose.
  - `stdin_open: true`, `tty: true` (interactive dev shell)
  - `volumes:`
    - `.:/workspace` (live code + host-side persistence of `data/`)
    - `uv-cache:/root/.cache/uv` (named — wheel cache survives rebuilds)
  - `ports: ["8000:8000"]` (FastAPI)
  - `working_dir: /workspace`
- `volumes:` top-level declares `uv-cache`.
- API launch documented as an override:
  `docker compose run --rm --service-ports pharmagen uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.

### 3. `.dockerignore`

Exclude from build context: `.venv .git BACKUPS graphify-out __pycache__
*.pyc *.pyo .pytest_cache .ruff_cache .mypy_cache node_modules data/raw
data/processed *.pt` and other large/generated artifacts. Keeps the build
context (and thus `docker build` upload) small. `pyproject.toml` and `uv.lock`
must NOT be ignored (needed for the dependency layer).

## Data flow / usage

```bash
# one-time host prereq: NVIDIA driver + nvidia-container-toolkit installed
docker compose build

# interactive dev shell, GPU ready
docker compose run --rm pharmagen
#   inside: python main.py --mode train --model TwoTowerGAT --input data/train.tsv
#   inside: uv run pytest tests/unit -q

# run the API
docker compose run --rm --service-ports pharmagen \
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

Training outputs (checkpoints, `data/pgen_model/encoders/…`) land in the
bind-mounted `./data` on the host and persist across container rebuilds.

## Error handling / edge cases

- **No GPU on host / missing toolkit**: `--gpus all` fails at container start.
  Documented as a host prereq; the image itself still builds.
- **cu130 wheel availability for py3.14**: `uv sync --frozen` resolves against
  the committed `uv.lock`. If a wheel needs compilation, `build-essential` is
  present as a fallback.
- **Bind-mount shadowing**: mitigated by `/opt/venv` location (see rationale).
- **Windows host**: same image via Docker Desktop + WSL2 backend; documented.

## Verification

- `docker compose build` completes.
- `docker compose run --rm pharmagen python -c "import torch; print(torch.cuda.is_available())"` prints `True` on a GPU host.
- `docker compose run --rm pharmagen uv run pytest tests/unit -q` passes.
- API responds on `GET /health`.

## Testing

No new application code, so no unit tests. Verification is the build + the
smoke commands above (torch.cuda check, pytest, `/health`).
