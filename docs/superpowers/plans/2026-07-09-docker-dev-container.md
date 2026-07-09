# Docker GPU Dev Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Docker image + compose setup that runs the entire Pharmagen workflow (install, train, infer, test, serve API) with NVIDIA CUDA 13.0 GPU access, code bind-mounted live from the host.

**Architecture:** Single Ubuntu-24.04 CUDA-cudnn base image. `uv` installs Python 3.14 standalone and builds the project venv at `/opt/venv` (outside the bind-mount) at image-build time. docker-compose bind-mounts the repo at `/workspace` for live editing and exposes port 8000 for FastAPI.

**Tech Stack:** Docker, docker-compose, NVIDIA Container Toolkit, `nvidia/cuda` base image, `uv`, Python 3.14, PyTorch 2.11 cu130 + PyTorch Geometric.

## Global Constraints

- Base image: `nvidia/cuda:<tag>-cudnn-runtime-ubuntu24.04` — exact `<tag>` (CUDA 13.0.x) MUST be verified against Docker Hub at build time before pinning.
- Python is **3.14** (`requires-python = ">=3.14,<3.15"`), installed by `uv`, NOT the base-OS python.
- Deps are pinned to CUDA 13.0 (`cu130`) wheels via `[tool.uv.index]` in `pyproject.toml`. Do not alter dependency pins.
- Venv MUST live at `/opt/venv` (env `UV_PROJECT_ENVIRONMENT=/opt/venv`), never `/workspace/.venv` — the repo bind-mount would shadow it.
- `uv sync` uses `--frozen` (respect committed `uv.lock`) and `--extra dev`.
- No emoji in any file. English only.
- Reuse the existing `data/` tree for persistence — training artifacts already route to `data/pgen_model/encoders` etc. via `Settings.paths`.

---

### Task 1: `.dockerignore`

**Files:**
- Create: `.dockerignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a minimal build context. `pyproject.toml` and `uv.lock` MUST remain included (Task 2's dependency layer needs them).

- [ ] **Step 1: Write `.dockerignore`**

```
# VCS / meta
.git
.gitignore
BACKUPS/
docs/superpowers/

# Python caches / envs
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# Graphify / tooling artifacts
graphify-out/

# Large data + generated model artifacts (mounted at runtime, not baked)
data/raw/
data/processed/
data/library/
**/*.pt

# Editor / OS
.vscode/
.idea/
.DS_Store
```

- [ ] **Step 2: Verify build context shrinks**

Run: `docker build --no-cache -f /dev/stdin . <<<'FROM busybox' 2>&1 | head -1`
Expected: prints `Sending build context to Docker daemon` with a size in MB (not GB). If it errors because Docker isn't installed, note it and continue — real verification happens in Task 2.

- [ ] **Step 3: Commit**

```bash
git add .dockerignore
git commit -m "chore(docker): add .dockerignore"
```

---

### Task 2: `Dockerfile`

**Files:**
- Create: `Dockerfile`

**Interfaces:**
- Consumes: `pyproject.toml`, `uv.lock` (dependency layer), `.dockerignore` from Task 1.
- Produces: an image whose `/opt/venv/bin` is on `PATH`; `python` is 3.14; `torch`, `torch_geometric`, `rdkit`, `pysam`, `fastapi`, `uvicorn` are importable. `WORKDIR` is `/workspace`. Default `CMD` is `bash`.

- [ ] **Step 1: Confirm the base image tag exists**

Run: `docker pull nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 || docker search nvidia/cuda`
Expected: a valid CUDA 13.0.x cudnn-runtime ubuntu24.04 tag pulls. If `13.0.0` is absent, list tags via `https://hub.docker.com/r/nvidia/cuda/tags` and pick the newest `13.0.x-cudnn-runtime-ubuntu24.04`. Record the exact tag and use it in Step 2.

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1

# CUDA 13.0 + cuDNN runtime on Ubuntu 24.04. Verify the exact tag on Docker Hub
# (see plan Task 2 Step 1) before changing this line.
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System libraries:
#   build-essential/git/curl/ca-certificates : general build + fetch
#   libcurl4 zlib1g libbz2-1.0 liblzma5 libdeflate0 : pysam / htslib runtime
#   libgomp1 : OpenMP runtime for torch
#   libxrender1 libxext6 libsm6 : rdkit shared-lib deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libcurl4 \
        zlib1g \
        libbz2-1.0 \
        liblzma5 \
        libdeflate0 \
        libgomp1 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned image, statically linked binaries)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# uv / venv configuration.
#   UV_PROJECT_ENVIRONMENT=/opt/venv keeps the venv OUTSIDE /workspace so the
#   compose bind-mount does not shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Python 3.14 standalone toolchain (independent of the base-OS python).
RUN uv python install 3.14

WORKDIR /workspace

# Dependency layer: only pyproject.toml + uv.lock, so editing source code does
# not re-download the large cu130 wheels. --no-install-project because project
# source is bind-mounted at runtime, not present at build.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --frozen --no-install-project

CMD ["bash"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -t pharmagen:dev .`
Expected: build completes. The `uv sync` step downloads torch/pyg/rdkit wheels and creates `/opt/venv`. First build is slow (multi-GB wheels); that is expected.

- [ ] **Step 4: Smoke-test Python version and venv location**

Run: `docker run --rm pharmagen:dev bash -lc 'which python && python --version'`
Expected: path is `/opt/venv/bin/python` and version is `Python 3.14.x`.

- [ ] **Step 5: Smoke-test core imports (CPU-side, no GPU needed)**

Run: `docker run --rm pharmagen:dev python -c "import torch, torch_geometric, rdkit, pysam, fastapi, uvicorn; print('imports ok', torch.__version__)"`
Expected: prints `imports ok 2.11.0` (or the locked torch version) with no ImportError.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile
git commit -m "feat(docker): CUDA 13.0 dev image with uv-managed Python 3.14 venv"
```

---

### Task 3: `docker-compose.yml`

**Files:**
- Create: `docker-compose.yml`

**Interfaces:**
- Consumes: the `Dockerfile` from Task 2 (via `build: .`).
- Produces: a `pharmagen` service with GPU access, repo bind-mounted at `/workspace`, `uv-cache` named volume, port 8000 published. Interactive (`tty`/`stdin_open`).

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  pharmagen:
    build: .
    image: pharmagen:dev
    # Live-mount the repo so host edits are visible in-container. The venv at
    # /opt/venv is outside this mount and is NOT shadowed.
    volumes:
      - .:/workspace
      - uv-cache:/root/.cache/uv
    working_dir: /workspace
    # Publish FastAPI. Only bound when started with --service-ports (compose run)
    # or via `compose up`.
    ports:
      - "8000:8000"
    # Interactive dev shell support.
    stdin_open: true
    tty: true
    # NVIDIA GPU access. Requires nvidia-container-toolkit on the host.
    gpus: all

volumes:
  uv-cache:
```

- [ ] **Step 2: Validate compose file syntax**

Run: `docker compose config`
Expected: prints the normalized config with no error. If `gpus: all` is rejected by an older Compose, replace with the `deploy.resources.reservations.devices` form:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

- [ ] **Step 3: Verify GPU is visible inside the container**

Run: `docker compose run --rm pharmagen python -c "import torch; print('cuda', torch.cuda.is_available())"`
Expected: on a GPU host with the toolkit installed, prints `cuda True`. On a host without a GPU it prints `cuda False` — note this and continue (image is correct; host lacks GPU).

- [ ] **Step 4: Verify the bind-mount does not shadow the venv**

Run: `docker compose run --rm pharmagen bash -lc 'which python && python -c "import torch; print(torch.__version__)"'`
Expected: `python` resolves to `/opt/venv/bin/python` and torch imports — proving the bind-mount at `/workspace` did not hide `/opt/venv`.

- [ ] **Step 5: Run the unit tests inside the container**

Run: `docker compose run --rm pharmagen uv run pytest tests/unit -q`
Expected: the unit suite runs and passes (same result as on the host).

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(docker): compose service with GPU, bind-mount, uv cache volume"
```

---

### Task 4: Document usage in README

**Files:**
- Modify: `README.md` (append a `## Docker` section)

**Interfaces:**
- Consumes: the three files from Tasks 1-3.
- Produces: user-facing run instructions + host prerequisites.

- [ ] **Step 1: Append a Docker section to `README.md`**

Add this section (place it after the existing install/usage content):

```markdown
## Docker (GPU dev container)

Runs the whole workflow — install, train, inference, tests, API — inside a
CUDA 13.0 container. Code is bind-mounted from the host; the Python 3.14 venv
is built into the image at `/opt/venv`.

### Host prerequisites
- NVIDIA GPU + recent driver.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed and configured for Docker.
- Windows: use Docker Desktop with the WSL2 backend — the same Linux image runs
  unchanged.

### Build
```bash
docker compose build
```

### Interactive dev shell (GPU ready)
```bash
docker compose run --rm pharmagen
# inside the container:
python main.py --mode train --model TwoTowerGAT --input data/train.tsv --epochs 100
uv run pytest tests/unit -q
```

### Serve the FastAPI inference API
```bash
docker compose run --rm --service-ports pharmagen \
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# -> http://localhost:8000/docs
```

Training artifacts (checkpoints, `data/pgen_model/encoders/...`) are written to
the bind-mounted `./data` on the host and persist across container rebuilds.
```

- [ ] **Step 2: Verify the markdown renders**

Run: `grep -n "## Docker" README.md`
Expected: the new section header is present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document Docker GPU dev container usage"
```

---

## Self-Review

- **Spec coverage:** 3 files from the spec (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) → Tasks 1-3; usage docs → Task 4. GPU, `/opt/venv`, `uv sync --frozen --extra dev`, Python 3.14, bind-mount, persistence, host prereqs all covered. ✓
- **Placeholder scan:** no TBD/TODO; every file has full content; the one deliberate variable (base image tag) has an explicit verification step (Task 2 Step 1) and a documented fallback. ✓
- **Type/name consistency:** image tag `pharmagen:dev`, service `pharmagen`, venv `/opt/venv`, mount `/workspace`, cache volume `uv-cache`, port `8000` are used identically across all tasks. ✓
```
