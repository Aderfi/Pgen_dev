"""HGVS nomenclature parser.

Turns an HGVS string into the structured ``HGVSVariant`` model defined in
``src.domain.hgvs``. Handles every standard expression class:

  * Reference + molecular type (``NC_000017.11:g.``, ``NM_007294.4:c.``,
    ``LRG_8t1:p.``, ``ENST00000012048:c.``...), with optional gene symbol
    inside parentheses (``NM_004006.2(DMD):c.93+1G>T``).
  * Single elementary changes at the nucleotide level: substitution, silent,
    deletion, duplication, insertion, delins, inversion, conversion, repeat.
  * Single elementary changes at the protein level: substitution (incl.
    nonsense ``Gln1812*`` / ``Gln1812Ter``), silent, deletion, duplication,
    insertion, delins, frameshift, N-terminal and C-terminal extension,
    ``p.0``, ``p.?``, and parenthesized predicted variants ``p.(...)``.
  * Coding-DNA coordinate grammar: ``-15`` (5'UTR), ``*15`` (3'UTR),
    ``100+5`` / ``100-5`` (intronic offsets), ``?`` (unknown position).
  * Compound expressions: ``[a;b]`` (cis), ``[a];[b]`` (trans),
    ``[a(;)b]`` (unknown phase), ``[a];[a]`` (homozygous shortcut).
  * Mosaic ``A=/>G`` and chimeric ``A=//>G``.

The grammar follows https://hgvs-nomenclature.org/stable/. Anything not in
the official grammar raises ``BioinformaticsError`` with the offending token.
"""

from __future__ import annotations

import re

from src.core.exceptions import BioinformaticsError
from src.domain.schemas.hgvs import (
    AMINO_ACID_THREE_TO_ONE,
    HGVSVariant,
    MolecularType,
    NucleotideChange,
    ProteinChange,
    ProteinPosition,
    ReferenceSequenceKind,
    SequencePosition,
    VariantKind,
    VariantPhase,
)

# --------------------------------------------------------------------------- #
# Regex building blocks                                                        #
# --------------------------------------------------------------------------- #
# Nucleotide position (incl. 5'UTR -15, 3'UTR *15, intronic 100+5, unknown ?).
_NUCL_POS = r"(?:\*?-?\d+(?:[+-]\d+)?|\?)"
_NUCL_RANGE = rf"(?P<start>{_NUCL_POS})(?:_(?P<end>{_NUCL_POS}))?"
_BASE = r"[ACGTUNacgtun]"
_BASES = rf"{_BASE}+"

# Amino-acid token — 3-letter code (canonical), 1-letter code, Ter, *, or
# the unknown-residue marker Xaa/X.
_AA = r"(?:[A-Z][a-z]{2}|Ter|Xaa|\*|[ARNDCQEGHILKMFPSTWYVUOX])"
_PROT_POS = rf"(?P<aa>{_AA})(?P<idx>\d+)"

# --- Nucleotide elementary-change patterns -------------------------------- #
_RE_NUCL_UNKNOWN = re.compile(r"^\?$")
_RE_NUCL_SUB = re.compile(rf"^(?P<pos>{_NUCL_POS})(?P<ref>{_BASE})>(?P<alt>{_BASE})$")
_RE_NUCL_SILENT = re.compile(rf"^(?P<pos>{_NUCL_POS})(?P<ref>{_BASE})?=$")
_RE_NUCL_DELINS = re.compile(rf"^{_NUCL_RANGE}delins(?P<ins>.+)$")
_RE_NUCL_DEL = re.compile(rf"^{_NUCL_RANGE}del(?P<ref>{_BASE}*)$")
_RE_NUCL_DUP = re.compile(rf"^{_NUCL_RANGE}dup(?P<ref>{_BASE}*)$")
_RE_NUCL_INS = re.compile(
    rf"^(?P<start>{_NUCL_POS})_(?P<end>{_NUCL_POS})ins(?P<ins>.+)$"
)
_RE_NUCL_INV = re.compile(rf"^(?P<start>{_NUCL_POS})_(?P<end>{_NUCL_POS})inv$")
_RE_NUCL_CON = re.compile(
    rf"^(?P<start>{_NUCL_POS})_(?P<end>{_NUCL_POS})con(?P<target>.+)$"
)
# Repeat: pos[N], pos_pos[N], or posUNIT[N] (e.g. `c.10CAG[5]`).
_RE_NUCL_REPEAT = re.compile(rf"^{_NUCL_RANGE}(?P<unit>{_BASES})?\[(?P<n>\d+)\]$")

# --- Protein elementary-change patterns ----------------------------------- #
_RE_PROT_UNKNOWN = re.compile(r"^\?$")
_RE_PROT_ZERO = re.compile(r"^0(?P<unk>\?)?$")
_RE_PROT_SILENT = re.compile(rf"^{_PROT_POS}=$")
_RE_PROT_DELINS = re.compile(
    rf"^(?P<aa1>{_AA})(?P<idx1>\d+)"
    rf"(?:_(?P<aa2>{_AA})(?P<idx2>\d+))?"
    rf"delins(?P<ins>(?:{_AA})+)$"
)
_RE_PROT_DEL = re.compile(
    rf"^(?P<aa1>{_AA})(?P<idx1>\d+)"
    rf"(?:_(?P<aa2>{_AA})(?P<idx2>\d+))?del$"
)
_RE_PROT_DUP = re.compile(
    rf"^(?P<aa1>{_AA})(?P<idx1>\d+)"
    rf"(?:_(?P<aa2>{_AA})(?P<idx2>\d+))?dup$"
)
_RE_PROT_INS = re.compile(
    rf"^(?P<aa1>{_AA})(?P<idx1>\d+)_(?P<aa2>{_AA})(?P<idx2>\d+)"
    rf"ins(?P<ins>(?:{_AA})+)$"
)
# Frameshift: Ser1982Argfs*22, Ser1982fs, Arg97Profs*?, Phe59fsTer10, Arg97ProfsTer23.
# The new stop may be written as '*', 'Ter' or 'X' per HGVS — all accepted here.
_RE_PROT_FS = re.compile(
    rf"^{_PROT_POS}(?P<new>{_AA})?fs(?:(?:\*|Ter|X)(?P<term>\d+|\?))?$"
)
# C-terminal extension: *110Trpext*17, Ter110Trpext*?
_RE_PROT_EXT_C = re.compile(
    rf"^(?:\*|Ter)(?P<idx>\d+)(?P<new>{_AA})?ext\*(?P<term>\d+|\?)$"
)
# N-terminal extension: Met1ext-5, Met1Valext-12
_RE_PROT_EXT_N = re.compile(
    rf"^(?:Met|M)(?P<idx>\d+)(?P<new>{_AA})?ext(?P<off>-\d+|-\?)$"
)
# Substitution last — its pattern (AA-pos-AA) is the most permissive and
# would shadow several of the above (fs, ext, del, ...) if tried first.
_RE_PROT_SUB = re.compile(rf"^{_PROT_POS}(?P<new>{_AA})$")

_RE_AA_TOKEN = re.compile(_AA)
_RE_REF_PREFIX = re.compile(r"^[A-Z]+")


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def parse(hgvs: str) -> HGVSVariant:
    """Parse an HGVS expression into an ``HGVSVariant``.

    Raises ``BioinformaticsError`` when the string is not valid HGVS.
    """
    if not isinstance(hgvs, str) or not hgvs.strip():
        msg = "HGVS string must be a non-empty str"
        raise BioinformaticsError(msg)

    raw = hgvs.strip()
    reference, gene, body = _split_prefix(raw)
    mol, body = _consume_molecular_type(raw, body)
    phase, fragments = _split_phase(body)

    parser = _parse_protein if mol is MolecularType.PROTEIN else _parse_nucleotide
    changes: list[NucleotideChange | ProteinChange] = []
    for fragment in fragments:
        local_phase, stripped = _detect_mosaic(fragment)
        if local_phase is not VariantPhase.SINGLE and phase is VariantPhase.SINGLE:
            phase = local_phase
        changes.append(parser(stripped, raw))

    if phase is VariantPhase.TRANS and len(changes) == 2 and changes[0] == changes[1]:
        phase = VariantPhase.HOMOZYGOUS

    return HGVSVariant(
        raw=raw,
        reference_sequence=reference,
        reference_kind=_classify_reference(reference),
        gene_symbol=gene,
        molecular_type=mol,
        phase=phase,
        changes=changes,
    )


# --------------------------------------------------------------------------- #
# Prefix + molecular type                                                      #
# --------------------------------------------------------------------------- #
def _split_prefix(raw: str) -> tuple[str | None, str | None, str]:
    """Split reference accession, gene symbol and the post-colon body."""
    if ":" not in raw:
        return None, None, raw
    prefix, _, body = raw.partition(":")
    m = re.fullmatch(r"\s*([^()\s]+)\s*(?:\(([^)]+)\))?\s*", prefix)
    if not m:
        msg = f"invalid reference prefix {prefix!r} in {raw!r}"
        raise BioinformaticsError(msg)
    return m.group(1), m.group(2), body


def _consume_molecular_type(raw: str, body: str) -> tuple[MolecularType, str]:
    if len(body) < 2 or body[1] != ".":
        msg = f"missing molecular-type prefix (expected '<letter>.') in {raw!r}"
        raise BioinformaticsError(msg)
    letter = body[0]
    try:
        mol = MolecularType(letter)
    except ValueError as exc:
        msg = f"unknown molecular-type letter {letter!r} in {raw!r}"
        raise BioinformaticsError(msg) from exc
    return mol, body[2:]


def _classify_reference(ref: str | None) -> ReferenceSequenceKind | None:
    if ref is None:
        return None
    m = _RE_REF_PREFIX.match(ref)
    if not m:
        return ReferenceSequenceKind.UNKNOWN
    prefix = m.group(0)
    for kind in ReferenceSequenceKind:
        if kind is ReferenceSequenceKind.UNKNOWN:
            continue
        if prefix == kind.value:
            return kind
    return ReferenceSequenceKind.UNKNOWN


# --------------------------------------------------------------------------- #
# Phase / brackets / mosaic                                                    #
# --------------------------------------------------------------------------- #
def _split_phase(body: str) -> tuple[VariantPhase, list[str]]:
    """Split a bracketed expression into elementary fragments + phase."""
    body = body.strip()
    if not body.startswith("["):
        return VariantPhase.SINGLE, [body]

    groups = _split_top_level(body, separators=";")
    if len(groups) > 1:
        # Trans: [a];[b];... — each group keeps its own brackets.
        return VariantPhase.TRANS, [
            _strip_outer_brackets(group, body) for group in groups
        ]

    inner = _strip_outer_brackets(body, body)
    if "(;)" in inner:
        return (
            VariantPhase.UNKNOWN_PHASE,
            [piece.strip() for piece in inner.split("(;)")],
        )
    if ";" in inner:
        return (
            VariantPhase.CIS,
            [piece.strip() for piece in inner.split(";")],
        )
    return VariantPhase.SINGLE, [inner.strip()]


def _strip_outer_brackets(token: str, ctx: str) -> str:
    token = token.strip()
    if not (token.startswith("[") and token.endswith("]")):
        msg = f"expected bracketed group, got {token!r} (in {ctx!r})"
        raise BioinformaticsError(msg)
    return token[1:-1].strip()


def _split_top_level(body: str, *, separators: str) -> list[str]:
    """Split `body` on `separators` chars only when they sit at bracket depth 0."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(body):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch in separators and depth == 0:
            parts.append(body[start:i].strip())
            start = i + 1
    parts.append(body[start:].strip())
    return parts


def _detect_mosaic(fragment: str) -> tuple[VariantPhase, str]:
    """Strip mosaic / chimeric markers and return the residual change."""
    if "=//>" in fragment:
        return VariantPhase.CHIMERIC, fragment.replace("=//>", ">")
    if "=/>" in fragment:
        return VariantPhase.MOSAIC, fragment.replace("=/>", ">")
    return VariantPhase.SINGLE, fragment


# --------------------------------------------------------------------------- #
# Nucleotide change parser                                                     #
# --------------------------------------------------------------------------- #
def _parse_nucleotide(fragment: str, raw: str) -> NucleotideChange:  # noqa: PLR0911 — regex dispatch table
    s = fragment.strip()
    if not s:
        msg = f"empty nucleotide change in {raw!r}"
        raise BioinformaticsError(msg)
    # Predicted-variant parentheses — accept and unwrap.
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    if _RE_NUCL_UNKNOWN.fullmatch(s):
        return NucleotideChange(
            kind=VariantKind.UNKNOWN, start=SequencePosition(unknown=True)
        )

    m = _RE_NUCL_SUB.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.SUBSTITUTION,
            start=_parse_seq_position(m["pos"], raw),
            reference_allele=m["ref"].upper(),
            alternate_allele=m["alt"].upper(),
        )

    m = _RE_NUCL_SILENT.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.SILENT,
            start=_parse_seq_position(m["pos"], raw),
            reference_allele=m["ref"].upper() if m["ref"] else None,
        )

    m = _RE_NUCL_DELINS.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.DELINS,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw) if m["end"] else None,
            inserted_sequence=_normalize_inserted(m["ins"]),
        )

    m = _RE_NUCL_DEL.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.DELETION,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw) if m["end"] else None,
            reference_allele=m["ref"].upper() if m["ref"] else None,
        )

    m = _RE_NUCL_DUP.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.DUPLICATION,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw) if m["end"] else None,
            inserted_sequence=m["ref"].upper() if m["ref"] else None,
        )

    m = _RE_NUCL_INS.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.INSERTION,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw),
            inserted_sequence=_normalize_inserted(m["ins"]),
        )

    m = _RE_NUCL_INV.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.INVERSION,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw),
        )

    m = _RE_NUCL_CON.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.CONVERSION,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw),
            inserted_sequence=m["target"],
        )

    m = _RE_NUCL_REPEAT.fullmatch(s)
    if m:
        return NucleotideChange(
            kind=VariantKind.REPEAT,
            start=_parse_seq_position(m["start"], raw),
            end=_parse_seq_position(m["end"], raw) if m["end"] else None,
            repeat_unit=m["unit"].upper() if m["unit"] else None,
            repeat_count=int(m["n"]),
        )

    msg = f"unrecognized nucleotide change {fragment!r} in {raw!r}"
    raise BioinformaticsError(msg)


_RE_PURE_BASES = re.compile(rf"^{_BASE}+$")


def _normalize_inserted(value: str) -> str:
    """Uppercase the inserted sequence when it is plain bases; otherwise keep it."""
    return value.upper() if _RE_PURE_BASES.fullmatch(value) else value


def _parse_seq_position(raw_pos: str, raw: str) -> SequencePosition:
    if raw_pos == "?":
        return SequencePosition(unknown=True)
    m = re.fullmatch(r"(\*)?(-?\d+)([+-]\d+)?", raw_pos)
    if not m:
        msg = f"invalid sequence position {raw_pos!r} in {raw!r}"
        raise BioinformaticsError(msg)
    return SequencePosition(
        utr3=m.group(1) == "*",
        base=int(m.group(2)),
        offset=int(m.group(3)) if m.group(3) else 0,
    )


# --------------------------------------------------------------------------- #
# Protein change parser                                                        #
# --------------------------------------------------------------------------- #
def _parse_protein(fragment: str, raw: str) -> ProteinChange:  # noqa: PLR0911, PLR0912 — regex dispatch table
    s = fragment.strip()
    if not s:
        msg = f"empty protein change in {raw!r}"
        raise BioinformaticsError(msg)

    uncertain = False
    if s.startswith("(") and s.endswith(")"):
        uncertain = True
        s = s[1:-1].strip()

    if _RE_PROT_UNKNOWN.fullmatch(s):
        return ProteinChange(kind=VariantKind.UNKNOWN, uncertain=uncertain)

    m = _RE_PROT_ZERO.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.NO_PROTEIN,
            uncertain=uncertain or bool(m["unk"]),
        )

    m = _RE_PROT_SILENT.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.SILENT,
            start=_make_protein_position(m["aa"], m["idx"], raw),
            uncertain=uncertain,
        )

    m = _RE_PROT_DELINS.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.DELINS,
            start=_make_protein_position(m["aa1"], m["idx1"], raw),
            end=(
                _make_protein_position(m["aa2"], m["idx2"], raw) if m["aa2"] else None
            ),
            inserted_residues=_split_aa_chain(m["ins"], raw),
            uncertain=uncertain,
        )

    m = _RE_PROT_DEL.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.DELETION,
            start=_make_protein_position(m["aa1"], m["idx1"], raw),
            end=(
                _make_protein_position(m["aa2"], m["idx2"], raw) if m["aa2"] else None
            ),
            uncertain=uncertain,
        )

    m = _RE_PROT_DUP.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.DUPLICATION,
            start=_make_protein_position(m["aa1"], m["idx1"], raw),
            end=(
                _make_protein_position(m["aa2"], m["idx2"], raw) if m["aa2"] else None
            ),
            uncertain=uncertain,
        )

    m = _RE_PROT_INS.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.INSERTION,
            start=_make_protein_position(m["aa1"], m["idx1"], raw),
            end=_make_protein_position(m["aa2"], m["idx2"], raw),
            inserted_residues=_split_aa_chain(m["ins"], raw),
            uncertain=uncertain,
        )

    m = _RE_PROT_FS.fullmatch(s)
    if m:
        term = m["term"]
        return ProteinChange(
            kind=VariantKind.FRAMESHIFT,
            start=_make_protein_position(m["aa"], m["idx"], raw),
            fs_new_residue=m["new"] if m["new"] else None,
            fs_terminator=int(term) if term and term != "?" else None,
            uncertain=uncertain,
        )

    m = _RE_PROT_EXT_C.fullmatch(s)
    if m:
        term = m["term"]
        return ProteinChange(
            kind=VariantKind.EXTENSION,
            start=ProteinPosition(amino_acid="Ter", pos=int(m["idx"])),
            new_amino_acid=m["new"] if m["new"] else None,
            ext_terminator=int(term) if term and term != "?" else None,
            uncertain=uncertain,
        )

    m = _RE_PROT_EXT_N.fullmatch(s)
    if m:
        off = m["off"]
        return ProteinChange(
            kind=VariantKind.EXTENSION,
            start=ProteinPosition(amino_acid="Met", pos=int(m["idx"])),
            new_amino_acid=m["new"] if m["new"] else None,
            ext_offset=int(off) if off != "-?" else None,
            uncertain=uncertain,
        )

    m = _RE_PROT_SUB.fullmatch(s)
    if m:
        return ProteinChange(
            kind=VariantKind.SUBSTITUTION,
            start=_make_protein_position(m["aa"], m["idx"], raw),
            new_amino_acid=m["new"],
            uncertain=uncertain,
        )

    msg = f"unrecognized protein change {fragment!r} in {raw!r}"
    raise BioinformaticsError(msg)


def _make_protein_position(aa: str, idx: str, raw: str) -> ProteinPosition:
    try:
        return ProteinPosition(amino_acid=aa, pos=int(idx))
    except ValueError as exc:
        msg = f"invalid protein position {aa}{idx} in {raw!r}: {exc}"
        raise BioinformaticsError(msg) from exc


def _split_aa_chain(chain: str, raw: str) -> list[str]:
    """Split a string of concatenated amino-acid codes into individual tokens."""
    residues: list[str] = []
    pos = 0
    while pos < len(chain):
        m = _RE_AA_TOKEN.match(chain, pos)
        if not m:
            msg = f"unrecognized residue in {chain!r} at offset {pos} (in {raw!r})"
            raise BioinformaticsError(msg)
        residues.append(m.group(0))
        pos = m.end()
    return residues


# --------------------------------------------------------------------------- #
# Examples                                                                     #
# --------------------------------------------------------------------------- #
MUTATION_EXAMPLES: list[str] = [
    "NC_000017.11:g.43045712G>A",
    "NM_007294.4:c.5434C>T",
    "p.Gln1812*",
    "p.Gln1812Ter",
    "NC_000007.14:g.117559590_117559592del",
    "NM_000492.4:c.1521_1523delCTT",
    "p.Phe508del",
    "NC_000013.11:g.32338763_32338764del",
    "NM_000059.4:c.5946_5947delTG",
    "p.Ser1982Argfs*22",
    "NC_000017.11:g.27104422_27104425dup",
]


def _demo() -> None:  # pragma: no cover - manual demo
    print("Example HGVS parsing:")
    for example in MUTATION_EXAMPLES:
        print("=" * 80)
        try:
            variant = parse(example)
            print(f"{example} →")
            print(variant.model_dump_json(indent=2, exclude_none=True))
        except BioinformaticsError as exc:
            print(f"FAILED  {example}: {exc}")


__all__ = [
    "AMINO_ACID_THREE_TO_ONE",
    "MUTATION_EXAMPLES",
    "parse",
]


if __name__ == "__main__":  # pragma: no cover
    _demo()
