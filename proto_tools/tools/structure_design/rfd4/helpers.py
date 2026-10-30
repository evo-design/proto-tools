"""Plain-type helpers for preparing RFD4-Proteina inputs (CIF prep, residue renumbering).

RFD4 resolves every ``contig`` and ``conditions`` residue id against the mmCIF
``label_seq_id``, not the ``auth_seq_id`` that PDB files display and that the rest of
proto-tools uses. The two need not agree -- upstream's own example inputs number
``label_seq_id`` from 17 -- so these helpers write an RFD4-readable CIF and return a
per-chain author-to-label map, letting callers keep using the numbering they see in
their input. What ``label_seq_id`` the prepared CIF carries depends on the source
format: gemmi preserves an existing ``label_seq_id``, so a CIF input keeps the
numbering it came with, while a PDB input (which has none) gets a fresh 1-based
per-chain counter. The map covers either case.

Kept free of ``proto_tools`` imports and typed on plain types so the hosted
execution backend can reuse them.

References:
    https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/reference/yaml-json.md
"""

import re
from collections.abc import Callable

# Reserved by RFD4 to mark an already-realized expandable segment.
MASKED_SEGMENT_NAME = "<M>"
CHAINBREAK = "/0"


def prepare_rfd4_cif(structure_content: str, structure_format: str) -> tuple[str, dict[str, dict[str, int]]]:
    """Write ``structure_content`` as a CIF that RFD4 can resolve selections against.

    A CIF converted from PDB carries no ``label_asym_id`` / ``label_seq_id``, and
    gemmi's own entity setup invents subchain ids such as ``Axp``. Both break RFD4
    selections, which address chains by ``label_asym_id`` and residues by
    ``label_seq_id``. This assigns label numbering and forces ``label_asym_id`` to
    equal the author chain id so ``sel('A')`` and ``A42-48`` mean what a caller expects.

    Args:
        structure_content (str): PDB or CIF text.
        structure_format (str): ``"pdb"`` or ``"cif"``.

    Returns:
        tuple[str, dict[str, dict[str, int]]]: The prepared CIF text, and a per-chain
            map from author residue label (``"17"``, or ``"17A"`` when an insertion
            code is present) to ``label_seq_id``.

    Raises:
        ValueError: If the structure cannot be parsed.
    """
    import gemmi

    if structure_format == "cif":
        doc = gemmi.cif.read_string(structure_content)
        structure = gemmi.make_structure_from_block(doc.sole_block())
    else:
        structure = gemmi.read_pdb_string(structure_content)

    structure.setup_entities()
    structure.assign_label_seq_id(True)

    # Pin only the polymer subchain: atomworks rejects a label_asym_id mixing set and unset label_seq_id.
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.label_seq is not None:
                    residue.subchain = chain.name

    auth_to_label: dict[str, dict[str, int]] = {}
    if len(structure) > 0:
        for chain in structure[0]:
            chain_map: dict[str, int] = {}
            for residue in chain:
                if residue.label_seq is None:
                    # Non-polymer (ligand, metal, water): addressed by res_name, not label_seq_id.
                    continue
                icode = residue.seqid.icode.strip()
                key = f"{residue.seqid.num}{icode}"
                # Two gemmi residues may share one author id (microheterogeneity, altlocs). That is
                # only ambiguous when they disagree about the label it maps to.
                if key in chain_map and chain_map[key] != int(residue.label_seq):
                    raise ValueError(
                        f"Chain {chain.name!r} maps author residue {key!r} to two different label ids "
                        f"({chain_map[key]} and {int(residue.label_seq)}), so it cannot be translated; "
                        "renumber the input, or pass residue_numbering='label'."
                    )
                chain_map[key] = int(residue.label_seq)
            if chain_map:
                auth_to_label[chain.name] = chain_map

    doc_out = structure.make_mmcif_document()
    return doc_out.as_string(), auth_to_label


def _identity_mapping(chain_map: dict[str, int]) -> bool:
    """Whether a chain's author numbering already equals its label numbering."""
    return all(key.isdigit() and int(key) == value for key, value in chain_map.items())


def chains_needing_translation(auth_to_label: dict[str, dict[str, int]]) -> list[str]:
    """Return the chain ids whose author numbering differs from their label numbering."""
    return sorted(chain for chain, chain_map in auth_to_label.items() if not _identity_mapping(chain_map))


def _translate_residue_id(chain_id: str, res_id: int, auth_to_label: dict[str, dict[str, int]], context: str) -> int:
    """Map one author residue id on ``chain_id`` to its ``label_seq_id``."""
    chain_map = auth_to_label.get(chain_id)
    if chain_map is None:
        raise ValueError(
            f"{context}: chain {chain_id!r} has no polymer residues in the input structure "
            f"(available chains: {sorted(auth_to_label)})."
        )
    label = chain_map.get(str(res_id))
    if label is None:
        numeric = sorted(int(k) for k in chain_map if k.isdigit())
        hint = f" (chain {chain_id} spans {numeric[0]}-{numeric[-1]})" if numeric else ""
        raise ValueError(
            f"{context}: residue {chain_id}{res_id} is not present in the input structure{hint}. "
            "Residue ids are read in author numbering; pass residue_numbering='label' to use "
            "mmCIF label_seq_id numbering instead."
        )
    return label


def translate_contig(contig: str, auth_to_label: dict[str, dict[str, int]]) -> str:
    """Rewrite a contig's preserved-segment residue ids from author to label numbering.

    Only preserved segments (``A42-48``, ``AA52``) carry residue ids; chainbreaks
    (``/0``) and expandable segments (``20-30``, ``E1:2-20R``) pass through untouched.

    Args:
        contig (str): Contig string in author numbering.
        auth_to_label (dict[str, dict[str, int]]): Per-chain author→label map.

    Returns:
        str: The contig with preserved-segment residue ids in label numbering.

    Raises:
        ValueError: If a token is malformed or references a missing residue.
    """
    translated: list[str] = []
    for raw_segment in contig.split(","):
        segment = raw_segment.strip()
        if not segment:
            raise ValueError(f"Empty contig token in {contig!r} (check for stray or trailing commas).")
        if segment == CHAINBREAK or ":" in segment:
            translated.append(segment)
            continue
        if not segment[0].isalpha():
            translated.append(segment)
            continue
        translated.append(_translate_preserved_segment(segment, auth_to_label))
    return ", ".join(translated)


def _translate_preserved_segment(token: str, auth_to_label: dict[str, dict[str, int]]) -> str:
    """Translate one preserved-segment contig token (``A42-48`` / ``AA52``)."""
    match = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", token)
    if match is None:
        raise ValueError(
            f"Malformed preserved-segment contig token {token!r}: expected <chain><res_id> or "
            "<chain><start>-<end> (RFD4 contigs cannot express insertion codes or negative ids)."
        )
    chain_id, start_str, end_str = match.group(1), match.group(2), match.group(3)
    context = f"contig token {token!r}"
    start = _translate_residue_id(chain_id, int(start_str), auth_to_label, context)
    if end_str is None:
        return f"{chain_id}{start}"
    end = _translate_residue_id(chain_id, int(end_str), auth_to_label, context)
    if end < start:
        raise ValueError(f"{context}: range end {end_str} precedes start {start_str} after translation.")
    _reject_inexact_span(chain_id, int(start_str), int(end_str), auth_to_label, context)
    return f"{chain_id}{start}-{end}"


def _reject_inexact_span(
    chain_id: str, start: int, end: int, auth_to_label: dict[str, dict[str, int]], context: str
) -> None:
    """Refuse an author range whose label span does not cover exactly the residues it names.

    An author range is only translatable when it maps onto a contiguous label run of the same
    length. It does not when an insertion code sits inside the span (``A52-53`` over 52, 52A,
    52B, 53 would silently cover four residues) or when the chain's author numbering is not
    monotone (the label span then covers fewer residues than were named). A plain gap in the
    author numbering is fine: those residues are simply absent from the structure.
    """
    chain_map = auth_to_label.get(chain_id, {})
    named = [key for key in chain_map if key.isdigit() and start <= int(key) <= end]
    span = chain_map[str(end)] - chain_map[str(start)] + 1
    if span == len(named):
        return
    inserted = sorted(k for k in chain_map if not k.isdigit() and k[:-1].isdigit() and start <= int(k[:-1]) <= end)
    detail = (
        f"insertion-coded residue(s) {inserted} sit inside it"
        if inserted
        else "the chain's author numbering is not monotone across it"
    )
    raise ValueError(
        f"{context}: author range {start}-{end} on chain {chain_id} covers {len(named)} residue(s) "
        f"but maps to a {span}-residue label span, because {detail}. A contig cannot address "
        "insertion codes; renumber the input, or pass residue_numbering='label' and use "
        "label_seq_id values."
    )


_SEL_CALL_RE = re.compile(r"\bsel\s*\(\s*(['\"])(.*?)\1\s*\)")
_BARE_RES_ID_RE = re.compile(r"\bres_id\b")


def contig_segment_names(contig: str | None) -> frozenset[str]:
    """Return the ``res_name`` values a contig's named expandable segments will carry.

    Those residues are generated, not taken from the input, so their ids are not author ids
    and must not be translated.
    """
    if not contig:
        return frozenset()
    return frozenset(segment.strip().split(":", 1)[0] for segment in contig.split(",") if ":" in segment)


def translate_selection(
    select: str,
    auth_to_label: dict[str, dict[str, int]],
    generated_names: frozenset[str] = frozenset(),
) -> str:
    """Rewrite the residue-id field of every ``sel('...')`` path in a selection query.

    Args:
        select (str): An atomworks query string in author numbering.
        auth_to_label (dict[str, dict[str, int]]): Per-chain author→label map.
        generated_names (frozenset[str]): ``res_name`` values of the contig's named expandable
            segments, whose residues are generated and so carry no author ids to translate.

    Returns:
        str: The query with ``sel(...)`` residue ids in label numbering.

    Raises:
        ValueError: If the query compares ``res_id`` directly on a chain whose numbering
            differs between author and label, which cannot be translated reliably.
    """
    if _BARE_RES_ID_RE.search(_SEL_CALL_RE.sub("", select)) and chains_needing_translation(auth_to_label):
        raise ValueError(
            f"Selection {select!r} compares 'res_id' directly, but this structure's author "
            f"numbering differs from its mmCIF label numbering on chain(s) "
            f"{chains_needing_translation(auth_to_label)}, so the comparison cannot be "
            "translated. Rewrite it as a sel('<chain>/<res_name>/<res_id>') path, or pass "
            "residue_numbering='label' and use label_seq_id values directly."
        )

    def _rewrite(match: re.Match[str]) -> str:
        quote, argument = match.group(1), match.group(2)
        # A sel() argument unions whole paths on top-level commas, so each translates independently.
        translated = ", ".join(
            _translate_selection_path(token.strip(), auth_to_label, select, generated_names)
            for token in _split_top_level_commas(argument)
        )
        return f"{match.group(0)[: match.group(0).index(quote)]}{quote}{translated}{quote})"

    return _SEL_CALL_RE.sub(_rewrite, select)


def _split_top_level_commas(argument: str) -> list[str]:
    """Split a selection argument on commas that are not inside a ``[...]`` list."""
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for char in argument:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            tokens.append("".join(current))
            current = []
            continue
        current.append(char)
    tokens.append("".join(current))
    return [token for token in tokens if token.strip()] or [argument]


def _translate_selection_path(
    path: str,
    auth_to_label: dict[str, dict[str, int]],
    select: str,
    generated_names: frozenset[str] = frozenset(),
) -> str:
    """Translate the ``res_id`` field of one ``<chain>/<res_name>/<res_id>/...`` path."""
    fields = path.split("/")
    if len(fields) < 3:
        # No res_id field present, so nothing to translate.
        return path
    if fields[1].strip() in generated_names:
        # The residues are generated by the contig, so their ids are not input author ids.
        return path

    res_id_field = fields[2].strip()
    if res_id_field in ("", "*"):
        return path

    chain_ids = _selection_chain_ids(fields[0].strip(), auth_to_label)
    context = f"selection {select!r}"

    def _map_one(value: str) -> str:
        value = value.strip()
        if not value or value == "*":
            return value
        if not value.lstrip("-").isdigit():
            raise ValueError(f"{context}: non-numeric residue id {value!r} in path {path!r}.")
        labels = {_translate_residue_id(chain, int(value), auth_to_label, context) for chain in chain_ids}
        if len(labels) != 1:
            raise ValueError(
                f"{context}: residue {value} maps to different label ids across chains {chain_ids}; "
                "select one chain per sel() path, or pass residue_numbering='label'."
            )
        return str(labels.pop())

    def _check_span(start: str, end: str) -> None:
        if not (start.lstrip("-").isdigit() and end.lstrip("-").isdigit()):
            return
        for chain in chain_ids:
            _reject_inexact_span(chain, int(start), int(end), auth_to_label, context)

    fields[2] = _map_bracketed_values(res_id_field, _map_one, _check_span)
    return "/".join(fields)


def _selection_chain_ids(chain_field: str, auth_to_label: dict[str, dict[str, int]]) -> list[str]:
    """Resolve a path's chain field to the concrete chain ids it addresses."""
    if chain_field in ("", "*"):
        return sorted(auth_to_label)
    if chain_field.startswith("[") and chain_field.endswith("]"):
        return [part.strip() for part in chain_field[1:-1].split(",") if part.strip()]
    return [chain_field]


def _map_bracketed_values(
    field: str, map_one: Callable[[str], str], check_span: Callable[[str, str], None] = lambda _a, _b: None
) -> str:
    """Apply ``map_one`` to each scalar, list entry, and range endpoint in a path field.

    ``check_span`` is called with each range's endpoints before they are mapped, so a selection
    range gets the same exactness check a contig range does.
    """
    if not (field.startswith("[") and field.endswith("]")):
        return map_one(field)

    parts: list[str] = []
    for raw_part in field[1:-1].split(","):
        part = raw_part.strip()
        if not part:
            continue
        # A range uses a '-' that is not a leading sign on either endpoint.
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if range_match is not None:
            check_span(range_match.group(1), range_match.group(2))
            parts.append(f"{map_one(range_match.group(1))}-{map_one(range_match.group(2))}")
        else:
            parts.append(map_one(part))
    return "[" + ", ".join(parts) + "]"
