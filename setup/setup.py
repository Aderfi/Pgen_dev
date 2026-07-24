#!/usr/bin/env python3
"""
Project Environment Setup Script.

This script handles the initialization of the development environment,
creation of the project directory structure, and configuration persistence.
It supports multiple environment managers (uv, mamba, conda, venv) with a
preference for faster options like 'uv' or 'mamba'.

Author: Assistant
Date: 2025-12-26
"""

import abc
import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Setup")


class EnvManagerType(Enum):
    """Supported environment managers."""

    UV = "uv"
    MAMBA = "mamba"
    CONDA = "conda"
    VENV = "venv"


class SystemUtils:
    """Utilities for system-level operations."""

    @staticmethod
    def get_os() -> str:
        return platform.system()

    @staticmethod
    def command_exists(command: str) -> bool:
        return shutil.which(command) is not None

    @staticmethod
    def run_command(
        command: list[str], cwd: Path | None = None, dry_run: bool = False
    ) -> bool:
        cmd_str = " ".join(command)
        if dry_run:
            logger.info(f"[DRY-RUN] Would execute: {cmd_str}")
            return True

        try:
            logger.info(f"Executing: {cmd_str}")
            subprocess.run(command, check=True, cwd=cwd)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"Executable not found for command: {command[0]}")
            return False


class DirectoryManager:
    """Responsibility: Managing the project file system structure."""

    REQUIRED_DIRS = [
        "src/config",
        "src/data",
        "src/genomics",
        "src/graphs",
        "src/interface",
        "src/modeling/architectures",
        "src/modeling/engine",
        "src/utils",
        "data/raw",
        "data/processed",
        "data/dicts",
        "logs",
        "library/drugs",
        "library/gene_graphs",
        "reports/figures",
        "results",
        "test",
    ]

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def create_structure(self, dry_run: bool = False):
        """Creates the standard directory tree if it doesn't exist."""
        logger.info("Verifying project directory structure...")
        for dir_rel in self.REQUIRED_DIRS:
            dir_path = self.root_path / dir_rel
            if dry_run:
                if not dir_path.exists():
                    logger.info(f"[DRY-RUN] Would create directory: {dir_rel}")
            elif not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_rel}")
            else:
                logger.debug(f"Directory exists: {dir_rel}")

        # Ensure __init__.py exists in src subdirs
        self._ensure_init_files(dry_run)

    def _ensure_init_files(self, dry_run: bool = False):
        """Ensures python packages are valid."""
        if dry_run:
            logger.info(
                "[DRY-RUN] Would ensure __init__.py files exist in src/ subdirectories."
            )
            return

        src_path = self.root_path / "src"
        if src_path.exists():
            for dir_path in src_path.rglob("*"):
                if dir_path.is_dir() and not dir_path.name.startswith("__"):
                    init_file = dir_path / "__init__.py"
                    if not init_file.exists():
                        init_file.touch()


class EnvStrategy(abc.ABC):
    """Abstract Strategy for Environment Creation and Package Installation."""

    def __init__(
        self,
        env_name: str,
        root_path: Path,
        python_version: str = "3.10",
        dry_run: bool = False,
    ):
        self.env_name = env_name
        self.root_path = root_path
        self.python_version = python_version
        self.dry_run = dry_run
        # Assuming script is in setup/, requirements should be there too
        self.requirements_path = self.root_path / "setup" / "requirements.txt"

    @abc.abstractmethod
    def create_env(self) -> bool:
        pass

    @abc.abstractmethod
    def install_packages(self) -> bool:
        pass


class UVStrategy(EnvStrategy):
    def create_env(self) -> bool:
        logger.info(f"Creating environment '{self.env_name}' using uv...")
        env_path = self.root_path / self.env_name
        cmd = ["uv", "venv", str(env_path), "--python", self.python_version]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)

    def install_packages(self) -> bool:
        logger.info(f"Installing packages using uv in '{self.env_name}'...")
        # With uv, you typically activate the venv or use `uv pip install` pointing to the venv
        # Assuming venv is created at root/env_name
        if SystemUtils.get_os() == "Windows":
            python_executable = (
                self.root_path / self.env_name / "Scripts" / "python.exe"
            )
        else:
            python_executable = self.root_path / self.env_name / "bin" / "python"

        # We can use 'uv pip install -p <python_path> -r requirements.txt'
        cmd = [
            "uv",
            "pip",
            "install",
            "-p",
            str(python_executable),
            "-r",
            str(self.requirements_path),
        ]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)


class MambaStrategy(EnvStrategy):
    def create_env(self) -> bool:
        logger.info(f"Creating environment '{self.env_name}' using mamba...")
        cmd = [
            "mamba",
            "create",
            "-n",
            self.env_name,
            f"python={self.python_version}",
            "-y",
        ]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)

    def install_packages(self) -> bool:
        logger.info(f"Installing packages using mamba in '{self.env_name}'...")
        # Mamba/Conda install from requirements.txt requires pip usually, or mamba install --file
        # but requirements.txt often has pip-specific syntax.
        # Safest is to run pip inside the conda env.
        # Alternatively: mamba install --name env_name --file requirements.txt
        # Let's try mamba install first as it's faster, but if it fails (due to pip deps), we might need pip.
        # Given the requirements.txt structure (hashes), pip is safer.

        if SystemUtils.get_os() == "Windows":
            pip_executable = (  # noqa
                self.root_path / "envs" / self.env_name / "Scripts" / "pip.exe"
            )  # approximate path for conda
            # Conda paths vary. Better to use 'mamba run -n env pip install ...'

        cmd = [
            "mamba",
            "run",
            "-n",
            self.env_name,
            "pip",
            "install",
            "-r",
            str(self.requirements_path),
        ]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)


class CondaStrategy(EnvStrategy):
    def create_env(self) -> bool:
        logger.info(f"Creating environment '{self.env_name}' using conda...")
        cmd = [
            "conda",
            "create",
            "-n",
            self.env_name,
            f"python={self.python_version}",
            "-y",
        ]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)

    def install_packages(self) -> bool:
        logger.info(f"Installing packages using conda in '{self.env_name}'...")
        cmd = [
            "conda",
            "run",
            "-n",
            self.env_name,
            "pip",
            "install",
            "-r",
            str(self.requirements_path),
        ]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)


class VenvStrategy(EnvStrategy):
    def create_env(self) -> bool:
        logger.info(f"Creating environment '{self.env_name}' using standard venv...")

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would look for python{self.python_version} or fallback to system python."
            )
            python_exec = f"python{self.python_version}"
        else:
            python_exec = f"python{self.python_version}"
            if not SystemUtils.command_exists(python_exec):
                if sys.version_info[:2] == (3, 10):
                    python_exec = sys.executable
                else:
                    logger.warning(
                        f"Python {self.python_version} not found in PATH. Using system 'python'."
                    )
                    python_exec = "python"

        env_path = self.root_path / self.env_name
        cmd = [python_exec, "-m", "venv", str(env_path)]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)

    def install_packages(self) -> bool:
        logger.info(f"Installing packages using pip in '{self.env_name}'...")
        if SystemUtils.get_os() == "Windows":
            pip_executable = self.root_path / self.env_name / "Scripts" / "pip.exe"
        else:
            pip_executable = self.root_path / self.env_name / "bin" / "pip"

        cmd = [str(pip_executable), "install", "-r", str(self.requirements_path)]
        return SystemUtils.run_command(cmd, dry_run=self.dry_run)


class ConfigManager:
    """Responsibility: Persisting user configuration."""

    CONFIG_FILE = ".user_env_cfg"

    def __init__(self, root_path: Path):
        self.config_path = root_path / self.CONFIG_FILE

    def save_config(self, manager: str, env_name: str, dry_run: bool = False):
        if dry_run:
            logger.info(f"[DRY-RUN] Would save configuration to {self.CONFIG_FILE}")
            return

        config = {
            "environment_manager": manager,
            "environment_name": env_name,
            "os_system": platform.system(),
            "python_version": "3.10",
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {self.CONFIG_FILE}")
        except OSError as e:
            logger.error(f"Failed to save configuration: {e}")


class SetupOrchestrator:
    """Main application controller."""

    def __init__(self):
        # Assuming script is in setup_external/setup.py, project root is two levels up
        self.root_path = Path(__file__).resolve().parent.parent
        self.dir_manager = DirectoryManager(self.root_path)
        self.config_manager = ConfigManager(self.root_path)

        # Parse arguments
        parser = argparse.ArgumentParser(description="Pharmagen Environment Setup")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Simulate execution without making changes.",
        )
        self.args = parser.parse_args()
        self.dry_run = self.args.dry_run

    def _setup_packages(self, strategy: EnvStrategy):
        """Installs packages using the selected strategy."""
        print("\n=== Installing Dependencies ===")
        if strategy.install_packages():
            logger.info("Packages installed successfully.")
        else:
            logger.error("Failed to install packages.")

    def _get_user_input(self):
        print("\n=== Pharmagen Environment Setup ===")
        print(f"Detected OS: {SystemUtils.get_os()}")
        if self.dry_run:
            print("⚠️  DRY-RUN MODE ACTIVE ⚠️")

        # 1. Choose Manager
        print("\nSelect Environment Manager:")
        available_options = []

        if SystemUtils.command_exists("uv"):
            available_options.append(("1", EnvManagerType.UV))
            print("  1. uv (Recommended - Fast)")

        if SystemUtils.command_exists("mamba"):
            available_options.append(("2", EnvManagerType.MAMBA))
            print("  2. mamba (Recommended - Fast)")

        if SystemUtils.command_exists("conda"):
            available_options.append(("3", EnvManagerType.CONDA))
            print("  3. conda")

        available_options.append(("4", EnvManagerType.VENV))
        print("  4. venv (Standard)")

        choice = input("\nEnter choice [number]: ").strip()
        selected_manager = next(
            (mgr for key, mgr in available_options if key == choice),
            EnvManagerType.VENV,
        )

        # 2. Choose Name
        default_name = (
            ".venv"
            if selected_manager in [EnvManagerType.UV, EnvManagerType.VENV]
            else "pharmagen_env"
        )
        env_name = (
            input(f"Enter environment name [default: {default_name}]: ").strip()
            or default_name
        )

        return selected_manager, env_name

    def _get_strategy(self, manager: EnvManagerType, env_name: str) -> EnvStrategy:
        if manager == EnvManagerType.UV:
            return UVStrategy(env_name, self.root_path, dry_run=self.dry_run)
        elif manager == EnvManagerType.MAMBA:
            return MambaStrategy(env_name, self.root_path, dry_run=self.dry_run)
        elif manager == EnvManagerType.CONDA:
            return CondaStrategy(env_name, self.root_path, dry_run=self.dry_run)
        else:
            return VenvStrategy(env_name, self.root_path, dry_run=self.dry_run)

    def run(self):
        if self.dry_run:
            logger.info("Starting setup in DRY-RUN mode.")

        # 1. Setup Directory Structure
        self.dir_manager.create_structure(dry_run=self.dry_run)

        # 2. Get User Inputs
        manager, env_name = self._get_user_input()

        # 3. Create Environment
        strategy = self._get_strategy(manager, env_name)
        success = strategy.create_env()

        # 4. Save Config & Install Packages
        if success:
            self.config_manager.save_config(
                manager.value, env_name, dry_run=self.dry_run
            )

            # 5. Install Packages
            self._setup_packages(strategy)

            # 6. Final Instructions
            activate_cmd = ""
            if manager in [EnvManagerType.UV, EnvManagerType.VENV]:
                if SystemUtils.get_os() == "Windows":
                    activate_cmd = f"{env_name}\\Scripts\\activate"
                else:
                    activate_cmd = f"source {env_name}/bin/activate"
            else:
                activate_cmd = f"{manager.value} activate {env_name}"

            print("\n" + "=" * 40)
            if self.dry_run:
                print("✅ Dry-Run Complete! No changes were made.")
            else:
                print("✅ Setup & Installation Complete Successfully!")
            print("=" * 40)
            print(f"To activate your environment, run:\n  {activate_cmd}")
        else:
            print("\n❌ Environment creation failed.")


if __name__ == "__main__":
    try:
        app = SetupOrchestrator()
        app.run()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
