"""Bootstrap for the Pharmagen notebooks (Google Colab or a local Jupyter).

Colab's runtime is Python 3.12, while Pharmagen and Modephere need 3.14. So the
notebook kernel only drives things: :func:`prepare` installs ``uv``, lets it
provision Python 3.14 plus the locked dependencies, unpacks the data bundle
(``scripts/pack_colab_bundle.py``) and starts an IPython kernel in the project's
3.14 environment. The ``%%pgen`` cell magic runs a cell in that kernel and
relays what it prints and displays (text, figures, HTML) back to the notebook.
State persists between ``%%pgen`` cells, like in any kernel.

Locally (not in Colab) nothing is installed: the 3.14 kernel is started with
``uv run --with ipykernel`` over the repository's own environment and data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

IN_COLAB = "google.colab" in sys.modules
KERNEL_NAME = "pharmagen-py314"
KERNEL_PACKAGES = ("ipykernel", "plotly")
DATA_MARKER = Path("data/library/hgvs_catalog.pt")
READY_TIMEOUT_S = 300

_KERNEL: ProjectKernel | None = None

# Runs once in the 3.14 kernel: inline figures, readable logs, plotly renderer.
_KERNEL_BOOTSTRAP = """
%matplotlib inline
import logging, warnings
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
for _noisy in ("matplotlib", "PIL", "numba", "fsspec", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ipywidgets cannot work across the relay (widget comms are not forwarded), so
# tqdm correctly falls back to text bars; silence its advice to install them.
warnings.filterwarnings("ignore", message="IProgress not found")
try:
    import plotly.io as _pio
    if {in_colab}:
        _pio.renderers.default = "colab"
except ImportError:
    pass
import sys
print("Pharmagen kernel ready: Python", sys.version.split()[0])
"""


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _uv() -> str:
    exe = shutil.which("uv")
    if exe is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        exe = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    return exe


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def install_environment(repo: Path) -> None:
    """``uv sync`` (Python 3.14 + locked deps) plus the kernel's own packages."""
    uv = _uv()
    os.environ.setdefault(
        "UV_LINK_MODE", "copy"
    )  # Colab: cache and repo on different FS
    # No --extra dev: pytest/ruff/mypy/coverage are not used by any notebook.
    _run([uv, "sync"], repo)
    ensure_kernel_packages(repo)


def ensure_kernel_packages(repo: Path) -> None:
    """Install ``KERNEL_PACKAGES`` into the project environment if missing.

    Runs on every :func:`prepare`, not only when the environment is created.
    A ``.venv`` built by a plain ``uv sync``, or left behind by an interrupted
    install, has no ``ipykernel``; the kernel then dies the instant it is
    launched, and jupyter_client reports only "Kernel died before replying to
    kernel_info" with no cause.
    """
    python = _venv_python(repo)
    if not python.exists():
        msg = f"No project environment at {python}; call install_environment() first"
        raise FileNotFoundError(msg)
    probe = subprocess.run(
        [str(python), "-c", "import ipykernel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return
    print(
        "Installing the kernel's packages into the project environment ...", flush=True
    )
    _run(
        [_uv(), "pip", "install", "-q", "--python", str(python), *KERNEL_PACKAGES], repo
    )


def _kernel_launch_error(repo: Path) -> str:
    """Why the project interpreter cannot host a kernel, or ``''`` if it can."""
    python = _venv_python(repo)
    if not python.exists():
        return f"There is no project environment at {python}."
    probe = subprocess.run(
        [str(python), "-c", "import ipykernel, ipykernel_launcher"],
        capture_output=True,
        text=True,
        cwd=str(repo),
        check=False,
    )
    return "" if probe.returncode == 0 else probe.stderr.strip()


def _mount_drive_if_needed(bundle: str) -> None:
    if (
        IN_COLAB
        and bundle.startswith("/content/drive")
        and not Path("/content/drive/MyDrive").exists()
    ):
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive")


def install_data(repo: Path, bundle: str | None) -> None:
    """Unpack the data bundle (a zip path, e.g. on Drive, or an https URL)."""
    if (repo / DATA_MARKER).exists():
        print(f"Data already present in {repo / 'data'}")
        return
    if not bundle:
        msg = (
            "The data is missing (data/library/...). Set BUNDLE to the zip built by "
            "scripts/pack_colab_bundle.py - a path (e.g. on Google Drive) or an https URL."
        )
        raise FileNotFoundError(msg)
    _mount_drive_if_needed(bundle)
    if bundle.startswith(("http://", "https://")):
        target = Path(tempfile.gettempdir()) / "pharmagen_colab_bundle.zip"
        print(f"Downloading {bundle} ...", flush=True)
        urllib.request.urlretrieve(bundle, target)  # noqa: S310 - user-given URL
        path = target
    else:
        path = Path(bundle)
    print(f"Unpacking {path} into {repo} ...", flush=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(repo)
    if not (repo / DATA_MARKER).exists():
        msg = f"The zip {path} does not contain {DATA_MARKER}"
        raise FileNotFoundError(msg)


# --------------------------------------------------------------------------- #
# The Python 3.14 kernel and %%pgen
# --------------------------------------------------------------------------- #


class PgenCellError(Exception):
    """A ``%%pgen`` cell raised in the 3.14 kernel; shows the remote traceback."""

    def __init__(self, traceback_lines: list[str]) -> None:
        super().__init__("The cell failed in the Pharmagen kernel")
        self.traceback_lines = traceback_lines

    def _render_traceback_(self) -> list[str]:  # IPython hook
        return self.traceback_lines


class ProjectKernel:
    """An IPython kernel running in the project's Python 3.14 environment."""

    def __init__(self, repo: Path, argv: list[str]) -> None:
        from jupyter_client.kernelspec import KernelSpecManager
        from jupyter_client.manager import KernelManager

        spec_dir = Path(tempfile.mkdtemp(prefix="pharmagen-kernelspec-"))
        (spec_dir / KERNEL_NAME).mkdir()
        import json

        (spec_dir / KERNEL_NAME / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "display_name": "Pharmagen (Python 3.14)",
                    "language": "python",
                    "env": {"PYTHONPATH": str(repo), "PYTHONUNBUFFERED": "1"},
                }
            )
        )
        ksm = KernelSpecManager(kernel_dirs=[str(spec_dir)])
        self.km = KernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=ksm)

        # The kernel process's own stderr is where a failed launch explains
        # itself; without this it is discarded and the only symptom is
        # "Kernel died before replying to kernel_info".
        self._stderr_path = spec_dir / "kernel-stderr.log"
        self._stderr_file = self._stderr_path.open("wb")
        self.km.start_kernel(cwd=str(repo), stderr=self._stderr_file)
        self.kc = self.km.client()
        self.kc.start_channels()
        try:
            self.kc.wait_for_ready(timeout=READY_TIMEOUT_S)
        except RuntimeError as exc:
            raise RuntimeError(self._launch_failure(repo)) from exc

    def _launch_failure(self, repo: Path) -> str:
        """Assemble an actionable message for a kernel that never came up."""
        self._stderr_file.flush()
        try:
            captured = self._stderr_path.read_text(errors="replace").strip()
        except OSError:
            captured = ""
        reason = _kernel_launch_error(repo)
        parts = ["The project's Python 3.14 kernel died before it was ready."]
        if reason:
            parts.append(
                f"The project interpreter cannot host a kernel:\n{reason}\n\n"
                "Fix it by rebuilding the environment:\n"
                "    import colab_setup\n"
                f"    colab_setup.install_environment({str(repo)!r})"
            )
        if captured:
            parts.append(f"The kernel process wrote:\n{captured}")
        if not reason and not captured:
            parts.append(
                "It left no output. Check the project environment by hand:\n"
                f"    !{_venv_python(repo)} -m ipykernel_launcher --help"
            )
        return "\n\n".join(parts)

    def run(self, code: str) -> None:
        """Execute ``code`` and relay its output; raise if it failed."""
        from IPython.display import clear_output, publish_display_data

        msg_id = self.kc.execute(code, allow_stdin=False, store_history=False)
        error: list[str] | None = None
        try:
            while True:
                msg = self.kc.get_iopub_msg()
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    out = sys.stdout if content["name"] == "stdout" else sys.stderr
                    out.write(content["text"])
                    out.flush()
                elif kind in ("display_data", "execute_result", "update_display_data"):
                    publish_display_data(content["data"], content.get("metadata", {}))
                elif kind == "clear_output":
                    clear_output(wait=content.get("wait", False))
                elif kind == "error":
                    error = content["traceback"]
                elif kind == "status" and content["execution_state"] == "idle":
                    break
        except KeyboardInterrupt:
            self.km.interrupt_kernel()
            raise
        if error is not None:
            raise PgenCellError(error)

    def shutdown(self) -> None:
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)
        self._stderr_file.close()


def _kernel_argv(repo: Path) -> list[str]:
    launcher = ["-m", "ipykernel_launcher", "-f", "{connection_file}"]
    venv_python = repo / ".venv" / "bin" / "python"
    if IN_COLAB:
        return [str(venv_python), *launcher]
    # Local: don't touch the project's venv; layer the kernel packages on top.
    withs = [arg for pkg in KERNEL_PACKAGES for arg in ("--with", pkg)]
    return [_uv(), "run", "--project", str(repo), *withs, "python", *launcher]


def start_kernel(repo: Path) -> ProjectKernel:
    """Start (or restart) the 3.14 kernel and register ``%%pgen``."""
    global _KERNEL
    if _KERNEL is not None:
        _KERNEL.shutdown()
    print("Starting the project's Python 3.14 kernel ...", flush=True)
    _KERNEL = ProjectKernel(repo, _kernel_argv(repo))
    _KERNEL.run(_KERNEL_BOOTSTRAP.format(in_colab=IN_COLAB))
    _register_magics()
    return _KERNEL


def _register_magics() -> None:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        return

    def pgen(line: str, cell: str) -> None:
        """Run this cell in Pharmagen's Python 3.14 kernel."""
        if _KERNEL is None:
            msg = "Run the setup cell first"
            raise RuntimeError(msg)
        _KERNEL.run(cell)

    def pgen_restart(line: str) -> None:
        """Restart the 3.14 kernel (state is lost)."""
        if _KERNEL is not None:
            _KERNEL.km.restart_kernel(now=True)
            _KERNEL.kc.wait_for_ready(timeout=READY_TIMEOUT_S)
            _KERNEL.run(_KERNEL_BOOTSTRAP.format(in_colab=IN_COLAB))

    ip.register_magic_function(pgen, "cell", "pgen")
    ip.register_magic_function(pgen_restart, "line", "pgen_restart")


def prepare(repo: Path | str, bundle: str | None = None, **_: Any) -> ProjectKernel:
    """Everything a notebook needs before its first ``%%pgen`` cell."""
    repo = Path(repo).resolve()
    if IN_COLAB:
        if not _venv_python(repo).exists():
            install_environment(repo)
        else:
            # The environment may predate this bootstrap, or an earlier install
            # may have been interrupted; make sure it can host a kernel.
            ensure_kernel_packages(repo)
        install_data(repo, bundle)
    elif not (repo / DATA_MARKER).exists():
        install_data(repo, bundle)
    return start_kernel(repo)
