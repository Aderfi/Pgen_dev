# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#!/usr/bin/env python3
# coding=utf-8
"""Pharmagen - Main entry point.

Thin launcher: parse the headless CLI arguments and hand off to the ``tui``
dispatch. All argument definitions live in :mod:`src.tui.parser` and all
execution logic in :mod:`src.tui.app`.

The interactive menu is shelved pending a redesign; see
``src/interface/__cli.py``.

Usage:
    python main.py --mode train --model TwoTowerGAT --input data/train.tsv
    python main.py --mode predict --model TwoTowerGAT --input data/patients.csv

Author:
    Adrim Hamed Outmani (@Aderfi)

Copyright:
    (C) 2025 Adrim Hamed Outmani. Licensed under GNU GPLv3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.tui import build_parser, run

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and dispatch to the matching handler."""
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
