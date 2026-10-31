"""Command-line interface: ``python main.py {train,optuna,predict}``."""

from src.cli.app import main, run
from src.cli.parser import build_parser

__all__ = ["build_parser", "main", "run"]
