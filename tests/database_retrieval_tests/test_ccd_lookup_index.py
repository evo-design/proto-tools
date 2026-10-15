"""Tests for the ccd-lookup SMILES/InChIKey index saved beside components.cif.

Kept out of test_ccd_lookup.py, whose module-wide persistent worker would hold the index in memory
and never see the per-test weights directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from proto_tools.tools.database_retrieval import CcdLookupConfig, CcdLookupInput, run_ccd_lookup
from proto_tools.utils.proto_home import get_proto_home

_WEIGHTS_DIR_VAR = "PROTO_CCD_LOOKUP_WEIGHTS_DIR"
_INDEX_FILENAME = "ccd_index.json"
_ATP_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"


def _bundle_weights_dir() -> Path:
    """The weights directory the ccd_lookup worker resolves when no test override is set."""
    if override := os.environ.get(_WEIGHTS_DIR_VAR):
        return Path(override)
    cache = os.environ.get("PROTO_MODEL_CACHE", "")
    if cache in ("IN_ENV", "NONE"):
        pytest.skip(f"PROTO_MODEL_CACHE={cache} keeps the CCD bundle inside the tool env")
    return Path(cache or get_proto_home() / "proto_model_cache") / "ccd_lookup"


@pytest.fixture
def isolated_weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """A weights dir linking the real components.cif, seeded with a copy of the setup-built index."""
    # A CCD-code lookup builds the env (and with it the bundle and index) without touching the index.
    run_ccd_lookup(CcdLookupInput(identifiers=["ATP"]), CcdLookupConfig())
    source = _bundle_weights_dir()
    (tmp_path / "components.cif").symlink_to(source / "components.cif")
    index = json.loads((source / _INDEX_FILENAME).read_text())
    monkeypatch.setenv(_WEIGHTS_DIR_VAR, str(tmp_path))
    return tmp_path, index


def _redirect_atp(index: dict, code: str) -> None:
    """Point every SMILES and InChIKey entry for ATP at `code`, so a lookup reveals which index answered."""
    for table in (index["smiles"], index["inchikey"]):
        for key, value in table.items():
            if value == "ATP":
                table[key] = code


@pytest.mark.skip_ci
@pytest.mark.integration
def test_setup_builds_an_index_stamped_with_its_dependencies(isolated_weights_dir):
    """setup.sh leaves an index whose stamp records the versions and bundle it was built from."""
    _, index = isolated_weights_dir

    assert {"format", "rdkit", "pdbeccdutils", "components_size", "components_mtime_ns"} <= set(index["stamp"])
    assert index["inchikey"]["ZKHQWZAMYRWXGA-KQYNXXCUSA-N"] == "ATP"


@pytest.mark.skip_ci
@pytest.mark.integration
def test_smiles_lookup_reads_a_current_index(isolated_weights_dir):
    """A saved index with a matching stamp is used as-is rather than rebuilt."""
    weights_dir, index = isolated_weights_dir
    _redirect_atp(index, "GTP")
    (weights_dir / _INDEX_FILENAME).write_text(json.dumps(index))

    result = run_ccd_lookup(CcdLookupInput(identifiers=[_ATP_SMILES]), CcdLookupConfig())

    assert result.results[0].fragment.ccd_code == "GTP"


@pytest.mark.skip_ci
@pytest.mark.integration
def test_smiles_lookup_rebuilds_a_stale_index(isolated_weights_dir):
    """A stamp from another RDKit version triggers a rebuild, which is saved for later workers."""
    weights_dir, index = isolated_weights_dir
    current_stamp = dict(index["stamp"])
    _redirect_atp(index, "GTP")
    index["stamp"]["rdkit"] = "0.0.0"
    (weights_dir / _INDEX_FILENAME).write_text(json.dumps(index))

    result = run_ccd_lookup(CcdLookupInput(identifiers=[_ATP_SMILES]), CcdLookupConfig())

    assert result.results[0].fragment.ccd_code == "ATP"
    rebuilt = json.loads((weights_dir / _INDEX_FILENAME).read_text())
    assert rebuilt["stamp"] == current_stamp
    assert rebuilt["inchikey"]["ZKHQWZAMYRWXGA-KQYNXXCUSA-N"] == "ATP"
