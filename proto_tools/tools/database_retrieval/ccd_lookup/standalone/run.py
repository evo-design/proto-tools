"""CCD lookup standalone runner.

Enriches wwPDB Chemical Component Dictionary (CCD) entries via pdbeccdutils.
Accepts CCD codes or SMILES strings and returns chemical metadata (formula,
descriptors, parent component, RDKit-derived physchem properties) plus
optional UniChem cross-references and PDB usage information.

Atom/bond detail and 2D/3D rendering are not produced here — the calling
tool reconstructs an RDKit ``Mol`` from the canonical SMILES, and
``Fragment.visualize()`` covers visualization on the consumer side.

This module runs in an isolated micromamba environment and MUST NOT import
from `proto_tools`. All heavy chemistry deps are imported lazily inside
function bodies to keep subprocess startup cost minimal.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from typing import Any

from standalone_helpers import get_logger, resolve_weights_dir

logger = get_logger(__name__)

# ============================================================================
# Module-level caches (per-subprocess, lazy)
# ============================================================================

# Maps canonical SMILES -> CCD code (built once on first SMILES lookup).
_SMILES_INDEX: dict[str, str] | None = None
# Maps InChIKey -> CCD code (built once on first SMILES lookup).
_INCHIKEY_INDEX: dict[str, str] | None = None
# The wwPDB CCD bundle is ~115 MB compressed and ~500 MB uncompressed (the
# decompressed components.cif on disk). Parsing or re-reading it per identifier
# is the dominant cost (~10-60s); cache aggressively at module level.

# Per-CCD-code parsed component cache, to avoid re-parsing the same code.
_COMPONENT_CACHE: dict[str, Any] = {}
# pdbeccdutils-parsed components dict, keyed by (path, sanitize) since the
# parser's behavior on edge-case entries depends on sanitize.
_PARSED_BUNDLES: dict[tuple[str, bool], dict[str, Any]] = {}
# gemmi cif.Document for parent-CCD-code extraction (pdbeccdutils' Component
# does not expose mon_nstd_parent_comp_id, so we read it directly from the CIF).
_GEMMI_DOCS: dict[str, Any] = {}

# Characters that disqualify a string from being a CCD code.
_SMILES_FORBIDDEN_CHARS = set("=()[]#/\\+.@*:;%")


def _disable_native_chemistry_loggers() -> None:
    """Globally silence RDKit's ``rdApp.*`` log domain for the lifetime of the subprocess."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


_disable_native_chemistry_loggers()


@contextlib.contextmanager
def _silence_rdkit_logger() -> Any:
    """Redirect file descriptor 2 to /dev/null for the block.

    Some chemistry warnings (notably InChI library messages) bypass RDLogger
    and write to stderr from C++ directly. RDLogger silencing alone is not
    enough during whole-bundle iteration over the CCD's exotic entries.
    """
    sys.stderr.flush()
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


# ============================================================================
# Identifier classification
# ============================================================================


def _is_likely_ccd_code(s: str) -> bool:
    """Heuristic: is `s` plausibly a wwPDB CCD code rather than a SMILES?

    CCD codes are 1-5 characters, alphanumeric, and conventionally uppercase.
    SMILES contain bond/branch/special characters or lowercase atoms.

    Args:
        s (str): Candidate identifier.

    Returns:
        bool: True if `s` looks like a CCD code, False if it looks like SMILES.
    """
    if not s:
        return False
    s = s.strip()
    if len(s) < 1 or len(s) > 5:
        return False
    if any(c in _SMILES_FORBIDDEN_CHARS for c in s):
        return False
    # Must be alphanumeric only.
    if not re.fullmatch(r"[A-Za-z0-9]+", s):
        return False
    # SMILES like "CCO" or "c1ccccc1" pass the regex above. The wwPDB CCD
    # bundle uses uppercase codes; lowercase atoms in chemistry indicate
    # aromatic SMILES atoms. If the string contains lowercase letters AND
    # those letters are aromatic-atom symbols, treat it as SMILES.
    if any(c.islower() for c in s):
        # Lowercase aromatic atom symbols in SMILES.
        aromatic_atoms = {"c", "n", "o", "s", "p", "b"}
        if any(c in aromatic_atoms for c in s):
            return False
    return True


# ============================================================================
# CCD index for SMILES/InChIKey -> code resolution
# ============================================================================


def _build_ccd_indices(components_path: str) -> None:
    """Build canonical SMILES and InChIKey -> CCD code maps from the CCD bundle.

    Populates module-level `_SMILES_INDEX` and `_INCHIKEY_INDEX`. This is
    expensive (parses the entire CCD), so it is invoked lazily on the first
    SMILES lookup. Ambiguous InChIKeys (multiple codes mapping to one key)
    are dropped to keep lookups deterministic.

    Args:
        components_path (str): Path to the uncompressed components.cif.
    """
    global _SMILES_INDEX, _INCHIKEY_INDEX
    if _SMILES_INDEX is not None and _INCHIKEY_INDEX is not None:
        return

    from pdbeccdutils.core import ccd_reader
    from rdkit import Chem

    logger.info("Building CCD SMILES/InChIKey indices from %s ...", components_path)
    smiles_index: dict[str, str] = {}
    inchikey_counts: dict[str, int] = {}
    inchikey_index: dict[str, str] = {}

    with _silence_rdkit_logger():
        # sanitize=False here is independent of the user-facing CcdLookupConfig.sanitize
        # (which controls only the output component). For indexing we want maximum
        # coverage — sanitize=True would drop entries where RDKit's standard valence
        # rules reject the CCD's recorded structure (organometallics, hypervalent atoms).
        parsed = ccd_reader.read_pdb_components_file(components_path, sanitize=False)
        # Cache the parsed bundle so a later sanitize=False _load_component call
        # avoids a second 10-60s parse.
        _PARSED_BUNDLES[(components_path, False)] = parsed

        for ccd_id, result in parsed.items():
            try:
                component = result.component
                mol = component.mol
                if mol is None:
                    continue
                try:
                    canon = Chem.MolToSmiles(mol)
                except Exception:
                    canon = None
                if canon and canon not in smiles_index:
                    smiles_index[canon] = ccd_id
                ikey = getattr(component, "inchikey", None)
                if ikey:
                    inchikey_counts[ikey] = inchikey_counts.get(ikey, 0) + 1
                    if inchikey_counts[ikey] == 1:
                        inchikey_index[ikey] = ccd_id
            except Exception as exc:
                logger.debug("Skipping %s while indexing: %s", ccd_id, exc)
                continue

    # Drop ambiguous InChIKeys.
    for ikey, n in inchikey_counts.items():
        if n > 1:
            inchikey_index.pop(ikey, None)

    _SMILES_INDEX = smiles_index
    _INCHIKEY_INDEX = inchikey_index
    logger.info(
        "CCD indices built: %d unique SMILES, %d unique InChIKeys.",
        len(smiles_index),
        len(inchikey_index),
    )


def _smiles_to_ccd_code(smiles: str, components_path: str) -> str | None:
    """Resolve a SMILES string to a CCD code via canonical SMILES / InChIKey match.

    Args:
        smiles (str): Input SMILES string.
        components_path (str): Path to the uncompressed components.cif.

    Returns:
        str | None: Matching CCD code, or None if no match.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi as rdkit_inchi

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        canon = Chem.MolToSmiles(mol)
    except Exception:
        canon = None
    try:
        ikey = rdkit_inchi.MolToInchiKey(mol)
    except Exception:
        ikey = None

    _build_ccd_indices(components_path)
    assert _SMILES_INDEX is not None and _INCHIKEY_INDEX is not None

    if canon and canon in _SMILES_INDEX:
        return _SMILES_INDEX[canon]
    if ikey and ikey in _INCHIKEY_INDEX:
        return _INCHIKEY_INDEX[ikey]
    return None


# ============================================================================
# Component parsing
# ============================================================================


def _load_component(code: str, components_path: str, sanitize: bool) -> Any | None:
    """Load a single CCD entry by code, with per-code caching.

    Args:
        code (str): Uppercased CCD code.
        components_path (str): Path to the uncompressed components.cif.
        sanitize (bool): Whether to sanitize the parsed RDKit molecule.

    Returns:
        Any | None: pdbeccdutils Component, or None if not found / parse failed.
    """
    cache_key = f"{code}|{int(sanitize)}"
    if cache_key in _COMPONENT_CACHE:
        return _COMPONENT_CACHE[cache_key]

    bundle_key = (components_path, sanitize)
    parsed = _PARSED_BUNDLES.get(bundle_key)
    if parsed is None:
        from pdbeccdutils.core import ccd_reader

        # Silence the C++ logger across the bundle parse — the noise (organometallics, hypervalent
        # atoms) isn't tied to the user's code. Per-code warnings during _extract_record still surface.
        with _silence_rdkit_logger():
            parsed = ccd_reader.read_pdb_components_file(
                components_path,
                sanitize=sanitize,
            )
        _PARSED_BUNDLES[bundle_key] = parsed

    result = parsed.get(code) or parsed.get(code.upper())
    if result is None:
        _COMPONENT_CACHE[cache_key] = None
        return None
    component = result.component
    _COMPONENT_CACHE[cache_key] = component
    return component


def _parse_parent_ccd_code(code: str, components_path: str) -> str | None:
    """Best-effort lookup of ``_chem_comp.mon_nstd_parent_comp_id`` via gemmi.

    pdbeccdutils' ``Component`` does not expose the parent component, so we read
    it directly from the CIF block. The parsed gemmi document is cached per
    path so a batch shares a single read of the ~500 MB uncompressed bundle.
    """
    try:
        import gemmi

        doc = _GEMMI_DOCS.get(components_path)
        if doc is None:
            doc = gemmi.cif.read(components_path)
            _GEMMI_DOCS[components_path] = doc
        block = doc.find_block(code) or doc.find_block(code.upper())
        if block is None:
            return None
        val = block.find_value("_chem_comp.mon_nstd_parent_comp_id")
        if val is None:
            return None
        cleaned = val.strip().strip("'\"")
        if cleaned in ("", "?", "."):
            return None
        return cleaned
    except Exception as exc:
        logger.debug("Could not parse parent CCD code for %s: %s", code, exc)
        return None


# Map pdbeccdutils' RDKit-native keys to a stable snake_case schema.
_PDBECCDUTILS_KEY_TO_CANONICAL: dict[str, str] = {
    "amw": "molecular_weight",
    "exactmw": "exact_molecular_weight",
    "CrippenClogP": "logp",
    "tpsa": "tpsa",
    "lipinskiHBD": "num_h_donors",
    "lipinskiHBA": "num_h_acceptors",
    "NumRotatableBonds": "num_rotatable_bonds",
    "NumHeavyAtoms": "num_heavy_atoms",
    "NumAromaticRings": "num_aromatic_rings",
    "FractionCSP3": "fraction_csp3",
}


def _physchem_properties(component: Any, mol: Any) -> dict[str, float]:
    """Compute canonical snake_case physchem properties for the component.

    Pulls documented values from ``component.physchem_properties`` (RDKit
    Properties() output) when available, falling back to direct RDKit calls for
    any missing key. Always returns the same key schema; pdbeccdutils' bulk
    descriptor output (chi*/kappa*/etc.) is intentionally dropped.
    """
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    props: dict[str, float] = {}
    raw = component.physchem_properties or {}
    for raw_key, canonical in _PDBECCDUTILS_KEY_TO_CANONICAL.items():
        if raw_key in raw:
            with contextlib.suppress(TypeError, ValueError):
                props[canonical] = float(raw[raw_key])

    rdkit_fallbacks: tuple[tuple[str, Any, str], ...] = (
        ("molecular_weight", Descriptors, "MolWt"),
        ("exact_molecular_weight", Descriptors, "ExactMolWt"),
        ("logp", Crippen, "MolLogP"),
        ("tpsa", Descriptors, "TPSA"),
        ("num_h_donors", Lipinski, "NumHDonors"),
        ("num_h_acceptors", Lipinski, "NumHAcceptors"),
        ("num_rotatable_bonds", Lipinski, "NumRotatableBonds"),
    )
    for canonical, module, attr in rdkit_fallbacks:
        if canonical in props:
            continue
        with contextlib.suppress(Exception):
            props[canonical] = float(getattr(module, attr)(mol))

    if "num_heavy_atoms" not in props:
        with contextlib.suppress(Exception):
            props["num_heavy_atoms"] = float(mol.GetNumHeavyAtoms())
    return props


def _extract_record(component: Any, components_path: str) -> dict[str, Any]:
    """Build a result record from a parsed CCD component."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    warnings: list[str] = []
    mol = component.mol
    code = component.id

    smiles: str | None = None
    formula_weight: float | None = None
    physchem: dict[str, float] = {}
    if mol is not None:
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            warnings.append(f"Could not compute canonical SMILES: {exc}")
        try:
            # MolWt (average) matches the wwPDB-recorded _chem_comp.formula_weight;
            # ExactMolWt would give a slightly different monoisotopic mass.
            formula_weight = float(Descriptors.MolWt(mol))
        except Exception as exc:
            warnings.append(f"Could not compute formula_weight: {exc}")
        try:
            physchem = _physchem_properties(component, mol)
        except Exception as exc:
            warnings.append(f"Physchem properties failed: {exc}")

    return {
        "ccd_code": code,
        "name": component.name or None,
        "formula": component.formula or None,
        "formula_weight": formula_weight,
        "smiles": smiles,
        "inchi": component.inchi or None,
        "inchikey": component.inchikey or None,
        "released": component.released,
        "release_status": component.pdbx_release_status.name,
        "parent_ccd_code": _parse_parent_ccd_code(code, components_path) if code else None,
        "physchem_properties": physchem,
        "cross_references": None,
        "pdb_structures": None,
        "warnings": warnings,
        "errors": [],
    }


# ============================================================================
# Optional network enrichment
# ============================================================================


_UNICHEM_V1_URL = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
_RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
# Cap on RCSB result rows fetched by default. ATP, HEM, NAG, etc. each appear
# in thousands of PDB entries; without a cap we'd download a lot of JSON. If
# the true count exceeds this, we surface a warning so the caller knows the
# list was truncated.
_RCSB_MAX_ROWS = 1000


def _fetch_unichem_cross_refs(ccd_code: str, inchikey: str | None) -> tuple[dict[str, list[str]], str | None]:
    """Fetch UniChem cross-references for a CCD entry. Best-effort.

    Uses UniChem's public v1 REST API (docs:
    https://www.ebi.ac.uk/unichem/api/docs) keyed by InChIKey. Returns a
    mapping of source short name (``"chembl"``, ``"drugbank"``, ``"pubchem"``,
    ``"chebi"``, ``"hmdb"``, ...) to the list of compound IDs.

    Args:
        ccd_code (str): CCD code (used only in error messages).
        inchikey (str | None): InChIKey to query against UniChem.

    Returns:
        tuple[dict[str, list[str]], str | None]: (mapping, warning). Empty mapping on failure.
    """
    if not inchikey:
        return {}, f"UniChem lookup for {ccd_code!r} skipped: no InChIKey on the parsed component"

    try:
        import requests

        resp = requests.post(
            _UNICHEM_V1_URL,
            json={"compound": inchikey, "type": "inchikey"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}, f"UniChem v1 returned HTTP {resp.status_code}"
        data = resp.json()
    except Exception as exc:
        return {}, f"UniChem v1 request failed: {exc}"

    mapping: dict[str, list[str]] = {}
    compounds = data.get("compounds") if isinstance(data, dict) else None
    if not compounds:
        # InChIKey not present in UniChem.
        return {}, None
    for compound in compounds:
        if not isinstance(compound, dict):
            continue
        for source in compound.get("sources") or []:
            if not isinstance(source, dict):
                continue
            short = source.get("shortName") or source.get("longName")
            cid = source.get("compoundId")
            if not short or cid is None:
                continue
            mapping.setdefault(str(short), []).append(str(cid))
    return mapping, None


def _fetch_pdb_usage(ccd_code: str) -> tuple[list[str], str | None]:
    """Query the RCSB search API for PDB entries containing this CCD ligand.

    Uses RCSB's text_chem service against ``rcsb_chem_comp_container_identifiers.comp_id``
    (the search-enabled attribute for chemical-component IDs). Results are
    capped at ``_RCSB_MAX_ROWS``; if the true total exceeds the cap, the
    return tuple's warning notes the truncation.

    Args:
        ccd_code (str): CCD code.

    Returns:
        tuple[list[str], str | None]: (pdb_ids, warning). Empty list on failure.
    """
    try:
        import requests

        payload = {
            "query": {
                "type": "terminal",
                "service": "text_chem",
                "parameters": {
                    "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
                    "operator": "exact_match",
                    "value": ccd_code.upper(),
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": _RCSB_MAX_ROWS},
                "return_counts": False,
            },
        }
        resp = requests.post(_RCSB_SEARCH_URL, json=payload, timeout=30)
        if resp.status_code == 204:
            return [], None
        if resp.status_code != 200:
            return [], f"RCSB search returned HTTP {resp.status_code}"
        data = resp.json()
    except Exception as exc:
        return [], f"RCSB search failed: {exc}"

    ids: list[str] = []
    for hit in data.get("result_set") or []:
        if isinstance(hit, dict):
            pid = hit.get("identifier")
            if pid:
                ids.append(str(pid))
        elif isinstance(hit, str):
            ids.append(hit)

    total = data.get("total_count")
    warning: str | None = None
    if isinstance(total, int) and total > len(ids):
        warning = f"RCSB returned {total} PDB entries for {ccd_code!r}; truncated to {len(ids)}"
    return ids, warning


# ============================================================================
# Main entry
# ============================================================================


def _empty_record(ccd_code: str | None, errors: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    """Build an empty record for failed lookups.

    Args:
        ccd_code (str | None): Resolved CCD code, if any.
        errors (list[str]): Error messages.
        warnings (list[str] | None): Warning messages.

    Returns:
        dict[str, Any]: Record skeleton with defaults.
    """
    return {
        "ccd_code": ccd_code,
        "name": None,
        "formula": None,
        "formula_weight": None,
        "smiles": None,
        "inchi": None,
        "inchikey": None,
        "released": False,
        "release_status": None,
        "parent_ccd_code": None,
        "physchem_properties": {},
        "cross_references": None,
        "pdb_structures": None,
        "warnings": list(warnings) if warnings else [],
        "errors": list(errors),
    }


def run_ccd_lookup(input_data: dict[str, Any]) -> dict[str, Any]:
    """Enrich CCD entries for the given identifiers.

    Args:
        input_data (dict[str, Any]): Dispatched input with `identifiers` and `config`.

    Returns:
        dict[str, Any]: `{"records": [...]}` matching the standalone contract.
    """
    identifiers: list[str] = list(input_data.get("identifiers", []) or [])
    cfg: dict[str, Any] = dict(input_data.get("config", {}) or {})
    include_xrefs = bool(cfg.get("include_cross_references", False))
    include_pdb_usage = bool(cfg.get("include_pdb_usage", False))
    sanitize = bool(cfg.get("sanitize", True))

    weights_dir = resolve_weights_dir("ccd_lookup")
    components_path = os.path.join(weights_dir, "components.cif")
    if not os.path.isfile(components_path):
        # Every record fails the same way; surface a clear error per identifier.
        msg = f"CCD bundle not found at {components_path}; rerun setup.sh"
        return {"records": [_empty_record(None, [msg]) for _ in identifiers]}

    # Silence the C++ logger across the batch — pdbeccdutils re-emits "bond type above 3" /
    # "Invalid InChI prefix" warnings even after scoped silencers; real failures already land on record["errors"].
    with _silence_rdkit_logger():
        records = _process_identifiers(identifiers, components_path, sanitize, include_xrefs, include_pdb_usage)
    return {"records": records}


def _process_identifiers(
    identifiers: list[str],
    components_path: str,
    sanitize: bool,
    include_xrefs: bool,
    include_pdb_usage: bool,
) -> list[dict[str, Any]]:
    """Iterate identifiers and build their enrichment records.

    Args:
        identifiers (list[str]): Per-identifier inputs (CCD codes or SMILES).
        components_path (str): Path to components.cif.
        sanitize (bool): Whether to sanitize parsed RDKit molecules.
        include_xrefs (bool): Fetch UniChem cross-references when True.
        include_pdb_usage (bool): Fetch RCSB PDB usage when True.

    Returns:
        list[dict[str, Any]]: One record per input identifier in input order.
    """
    records: list[dict[str, Any]] = []
    for raw_id in identifiers:
        identifier = (raw_id or "").strip()
        if not identifier:
            records.append(_empty_record(None, ["Empty identifier"]))
            continue

        try:
            is_ccd = _is_likely_ccd_code(identifier)
            resolved_code: str | None = None
            warnings: list[str] = []

            if is_ccd:
                resolved_code = identifier.upper()
            else:
                try:
                    resolved_code = _smiles_to_ccd_code(identifier, components_path)
                except Exception as exc:
                    records.append(_empty_record(None, [f"SMILES resolution failed: {exc}"]))
                    continue
                if resolved_code is None:
                    records.append(_empty_record(None, ["No CCD match for SMILES"]))
                    continue

            component = _load_component(resolved_code, components_path, sanitize=sanitize)
            if component is None:
                records.append(
                    _empty_record(
                        resolved_code,
                        [f"CCD code {resolved_code!r} not found in components.cif"],
                        warnings,
                    )
                )
                continue

            record = _extract_record(component, components_path)
            if warnings:
                record["warnings"] = list(record.get("warnings", [])) + warnings

            if include_xrefs and resolved_code is not None:
                xrefs, warn = _fetch_unichem_cross_refs(resolved_code, record.get("inchikey"))
                record["cross_references"] = xrefs
                if warn:
                    record["warnings"].append(warn)

            if include_pdb_usage and resolved_code is not None:
                pdbs, warn = _fetch_pdb_usage(resolved_code)
                record["pdb_structures"] = pdbs
                if warn:
                    record["warnings"].append(warn)

            records.append(record)
        except Exception as exc:
            logger.exception("Unhandled error processing identifier %r", identifier)
            records.append(_empty_record(None, [f"Unhandled error: {exc}"]))

    return records


# ============================================================================
# Persistent-worker hooks
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution.

    Args:
        input_dict (dict[str, Any]): Standalone input payload.

    Returns:
        dict[str, Any]: Standalone output payload.
    """
    return run_ccd_lookup(input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CPU-only tool.

    Args:
        device (str): Requested device string.

    Returns:
        dict[str, Any]: Acknowledgement payload.
    """
    return {"success": True, "device": device, "note": "CPU tool"}


# ============================================================================
# CLI entrypoint (one-shot subprocess mode)
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(sys.argv[1]) as f:
        _input_data = json.load(f)
    _output_data = run_ccd_lookup(_input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(_output_data, f)
