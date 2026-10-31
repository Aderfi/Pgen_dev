"""User-facing console output for the CLI (emoji live here, never in logs)."""

from __future__ import annotations

import sys

WIDTH = 60


def header(title: str) -> None:
    print(f"\n{'─' * WIDTH}\n {title}\n{'─' * WIDTH}")


def success(msg: object) -> None:
    print(f"✅  {msg}")


def info(msg: object) -> None:
    print(f"ℹ️  {msg}")


def warning(msg: object) -> None:
    print(f"⚠️  {msg}")


def error(msg: object) -> None:
    print(f"❌  {msg}", file=sys.stderr)
