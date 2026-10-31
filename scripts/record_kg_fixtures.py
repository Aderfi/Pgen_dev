"""Record real NCBI / ClinPGx responses as test fixtures for ``src/kg``.

Usage::

    uv run python scripts/record_kg_fixtures.py --out tests/fixtures/kg

Re-running overwrites the fixtures; record the retrieval date in the
fixtures README when you do.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
from pathlib import Path

import requests

logger = logging.getLogger("record_kg_fixtures")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
VARIATION = "https://api.ncbi.nlm.nih.gov/variation/v0"
CLINPGX = "https://api.clinpgx.org/v1/download/file/attachment/{gene}_allele_definition_table.xlsx"

ESUMMARY_IDS = "3892097,17879456,4244285,1065852,999999999999"
CANONICAL = [
    "NC_000010.11:94781858:G:A",
    "NC_000002.12:233760234:TATATATATATATA:TATATATATATATATA",
    "NC_000002.12:233760233:ATATATATATATATA:ATATATATATATATATA",
    "NC_000006.12:18149103:AAGT:",
]
CONTEXTUALS = [
    "NC_000002.12:g.233760235TA[8]",
    "NC_000002.12:g.233760235TA[6]",
    "NC_000006.12:g.18149104_18149107del",
    "NC_000006.12:g.18149032_18149033insT",
]
SPDI_HGVS = [
    "NC_000002.12:233760233:ATATATATATATATA:ATATATATATATATATA",
    "NC_000006.12:18149103:AAGT:",
]
TABLE_GENES = ["TPMT", "UGT1A1"]


def fixture_name(key: str) -> str:
    """File-system safe name for a fixture key (``:`` → ``_``)."""
    return key.replace(":", "_").replace("/", "_") + ".json"


def _get(session: requests.Session, url: str) -> requests.Response:
    time.sleep(1.0)  # Variation Services etiquette: <= 1 request/s
    response = session.get(url, timeout=30)
    logger.info("GET %s -> %d", url, response.status_code)
    return response


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def record(out: Path) -> None:
    session = requests.Session()
    params = {"db": "snp", "id": ESUMMARY_IDS, "retmode": "json", "tool": "pharmagen"}
    _write_json(
        out / "esummary_snp.json",
        _get(session, f"{EUTILS}?{urllib.parse.urlencode(params)}").json(),
    )

    for spdi in CANONICAL:
        r = _get(session, f"{VARIATION}/spdi/{spdi}/canonical_representative")
        _write_json(out / "spdi_canonical" / fixture_name(spdi), r.json())
    for spdi in SPDI_HGVS:
        r = _get(session, f"{VARIATION}/spdi/{spdi}/hgvs")
        _write_json(out / "spdi_hgvs" / fixture_name(spdi), r.json())
    for hgvs in CONTEXTUALS:
        r = _get(
            session, f"{VARIATION}/hgvs/{urllib.parse.quote(hgvs, safe='')}/contextuals"
        )
        _write_json(out / "hgvs_contextuals" / fixture_name(hgvs), r.json())

    for gene in TABLE_GENES:
        r = _get(session, CLINPGX.format(gene=gene))
        r.raise_for_status()
        target = out / "allele_tables" / f"{gene}.xlsx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    record(parser.parse_args().out)


if __name__ == "__main__":
    main()
