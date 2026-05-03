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

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from src.config.manager import DIRS


def setup_logging(name="Pharmagen", level=None, console_level=None):
    logger_root = logging.getLogger()
    if logger_root.handlers:
        return logger_root  # already configured

    log_file = DIRS["logs"] / f"{name}_{datetime.now():%Y-%m-%d}.log"

    logger_root.setLevel(logging.DEBUG)

    fmt_file = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fmt_console = logging.Formatter("%(message)s")

    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7, encoding="utf-8")
    file_handler.setLevel(level or logging.WARNING)
    file_handler.setFormatter(fmt_file)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level or level or logging.WARNING)
    stream_handler.setFormatter(fmt_console)

    logger_root.addHandler(file_handler)
    logger_root.addHandler(stream_handler)

    for lib in ["matplotlib", "optuna", "numba"]:
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logger_root
