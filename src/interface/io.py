"""Console-facing IO helpers — JSON utilities and GPL notices."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "__UNKNOWN__"


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write ``data`` as JSON with 2-space indent."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Console banners and GPL notices.
# --------------------------------------------------------------------------- #


def welcome_message() -> None:
    settings = get_settings()
    msg = f"""
    ============================================
            PHARMAGEN v{settings.version}
    ============================================
    Pharmacogenetic prediction with deep learning.

    Logs: {settings.paths.logs}
    ============================================
    """
    print(msg)


def print_gnu_notice() -> None:
    """Print the short GPL boot notice."""
    settings = get_settings()
    start_year = 2025
    current_year = datetime.now().year
    year_str = f"{start_year}-{current_year}" if current_year > start_year else str(start_year)
    author = "Adrim Hamed Outmani (@Aderfi)"
    program = settings.project_name

    notice = f"""
    {program} Copyright (C) {year_str} {author}
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
    """
    print(notice)


def print_warranty_details() -> None:
    """Full warranty text for the ``show w`` command."""
    print("\n" + "=" * 60)
    print("NO WARRANTY")
    print("=" * 60)
    print("""
    BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
    FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN
    OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
    PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
    OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS
    TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE
    PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
    REPAIR OR CORRECTION.
    """)
    input("\nPress [Enter] to return...")


def print_conditions_details() -> None:
    """Full redistribution-conditions text for the ``show c`` command."""
    print("\n" + "=" * 60)
    print("REDISTRIBUTION CONDITIONS")
    print("=" * 60)
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """)
    input("\nPress [Enter] to return...")


__all__ = [
    "UNKNOWN_TOKEN",
    "load_json",
    "print_conditions_details",
    "print_gnu_notice",
    "print_warranty_details",
    "save_json",
    "welcome_message",
]
