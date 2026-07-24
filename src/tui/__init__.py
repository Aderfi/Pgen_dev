"""Terminal user interface: argument parsing and headless dispatch.

The interactive menu is shelved (see ``src/interface/__cli.py``) pending a
redesign; this package currently exposes only the headless CLI surface.
"""

from src.tui.app import run
from src.tui.parser import build_parser

__all__ = ["build_parser", "run"]
