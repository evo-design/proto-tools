"""Tests for PyRosetta FastRelax wrapper."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta import pyrosetta_relax as relax_module
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
    PyRosettaRelaxConfig,
    PyRosettaRelaxInput,
    RelaxResult,
    run_pyrosetta_relax,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from tests._structure_fixtures import synthetic_cif
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")


# ── Validation ────────────────────────────────────────────────────────────────


def test_relax_input_normalizes_single_structure():
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaRelaxInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], ScoringStructureInput)


def test_relax_input_accepts_dict_with_chains_to_score():
    inp = PyRosettaRelaxInput(inputs=[{"structure": TEST_PDB, "chains_to_score": ["A"]}])
    assert inp.inputs[0].chains_to_score is not None
    assert inp.inputs[0].chains_to_score.chains == ["A"]


def test_relax_forwards_bindcraft_fastrelax_options(monkeypatch):
    captured = {}

    def fake_dispatch(toolkit, input_data, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["input_data"] = input_data
        return {
            "results": [
                {
                    "relaxed_pdb": input_data["pdb_contents"][0],
                    "total_score": -1.0,
                    "relax_cycles": input_data["relax_cycles"],
                    "dropped_residues": [],
                }
            ]
        }

    monkeypatch.setattr(relax_module.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    config = PyRosettaRelaxConfig(
        scorefxn="beta_nov16",
        relax_cycles=1,
        max_iter=200,
        disable_jumps=True,
        min_type="lbfgs_armijo_nonmonotone",
        align_to_start=True,
        copy_b_factors_from_start=True,
        seed=7,
    )
    result = run_pyrosetta_relax(PyRosettaRelaxInput(inputs=[TEST_PDB]), config)

    assert result.success
    assert captured["toolkit"] == "pyrosetta"
    payload = captured["input_data"]
    assert len(payload["pdb_contents"]) == 1
    assert payload["operation"] == "relax"
    assert payload["scorefxn"] == "beta_nov16"
    assert payload["relax_cycles"] == 1
    assert payload["max_iter"] == 200
    assert payload["disable_jumps"] is True
    assert payload["min_type"] == "lbfgs_armijo_nonmonotone"
    assert payload["align_to_start"] is True
    assert payload["copy_b_factors_from_start"] is True
    assert payload["seed"] == 7


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_relax_returns_relaxed_structure():
    """End-to-end FastRelax: dispatch, return, parse, verify atoms moved."""
    structure = Structure(structure=TEST_PDB)

    result = run_pyrosetta_relax(
        PyRosettaRelaxInput(inputs=[structure]),
        PyRosettaRelaxConfig(relax_cycles=1, seed=42),
    )
    assert_metrics_in_spec(result)
    assert result.success
    assert result.tool_id == "pyrosetta-relax"
    assert len(result.results) == 1

    metrics = result.results[0]
    assert isinstance(metrics.total_score, float)
    # Renin in the dummy_data is a single-chain protein; relaxed total_score
    # should be negative, well above -2000 REU for ~340 residues
    assert -2000 <= metrics.total_score <= 0, f"Unreasonable total_score {metrics.total_score}"

    relax: RelaxResult = metrics.relax
    assert isinstance(relax.relaxed_structure, Structure)
    assert relax.relax_cycles == 1
    # Drop-in replacement contract: chain labels match the input exactly,
    # and the source format (PDB) is preserved.
    assert relax.relaxed_structure.get_chain_ids() == structure.get_chain_ids()
    assert relax.relaxed_structure.structure_format == "pdb"

    # Relaxation should have moved at least one atom (>0.01 Å on at least one CA).
    in_atoms = list(structure.gemmi_struct[0][0])  # model 0, chain 0
    out_atoms = list(relax.relaxed_structure.gemmi_struct[0][0])
    assert len(in_atoms) == len(out_atoms), "Relax dropped or added residues unexpectedly"
    moved = False
    for r_in, r_out in zip(in_atoms, out_atoms, strict=True):
        ca_in = next((a for a in r_in if a.name == "CA"), None)
        ca_out = next((a for a in r_out if a.name == "CA"), None)
        if ca_in is None or ca_out is None:
            continue
        delta = ca_in.pos.dist(ca_out.pos)
        if delta > 0.01:
            moved = True
            break
    assert moved, "FastRelax did not move any CA atoms by >0.01 Å"


@pytest.mark.integration
def test_relax_chains_into_energy():
    """The relaxed Structure should feed cleanly into pyrosetta-energy with relax=False."""
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
        PyRosettaEnergyConfig,
        PyRosettaEnergyInput,
        run_pyrosetta_energy,
    )

    relaxed_out = run_pyrosetta_relax(
        PyRosettaRelaxInput(inputs=[Structure(structure=TEST_PDB)]),
        PyRosettaRelaxConfig(relax_cycles=1, seed=42),
    )
    assert relaxed_out.success
    relaxed_structure = relaxed_out.results[0].relax.relaxed_structure

    energy_out = run_pyrosetta_energy(
        PyRosettaEnergyInput(inputs=[relaxed_structure]),
        PyRosettaEnergyConfig(),  # default: no preprocess relax
    )
    assert energy_out.success
    # Total energy on the already-relaxed pose should match the relax tool's score
    # within Rosetta numerical tolerance (a few REU).
    assert abs(energy_out.results[0].total_energy - relaxed_out.results[0].total_score) < 5.0


@pytest.mark.integration
def test_relax_preserves_multichar_chain_ids():
    """CIF input with multi-char chain labels round-trips correctly.

    PyRosetta internally shortens "Heavy" → "A" for PDB compatibility, but the
    wrapper restores the originals via Structure.with_renamed_chains. The
    source format (CIF) is also preserved.
    """
    structure = Structure(structure=synthetic_cif(["Heavy", "Light"]))
    assert structure.get_chain_ids() == ["Heavy", "Light"]
    assert structure.structure_format == "cif"

    result = run_pyrosetta_relax(
        PyRosettaRelaxInput(inputs=[structure]),
        PyRosettaRelaxConfig(relax_cycles=1, seed=42),
    )
    assert result.success

    relaxed = result.results[0].relax.relaxed_structure
    # Drop-in replacement: chains match and format matches.
    assert relaxed.get_chain_ids() == ["Heavy", "Light"]
    assert relaxed.structure_format == "cif"
    # Multi-char chains can be selected by their original labels.
    heavy_only = relaxed.select_chain("Heavy")
    assert heavy_only.get_chain_ids() == ["Heavy"]


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pyrosetta-relax")
@pytest.mark.slow
def test_pyrosetta_relax_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyrosetta-relax: 1 renin_af3 (~340 aa), default scorefxn ref2015, relax_cycles=1 (cold + warm)."""
    inputs = PyRosettaRelaxInput(inputs=[Structure(structure=TEST_PDB)])
    config = PyRosettaRelaxConfig(relax_cycles=1, seed=42)

    result = benchmark_twice(request, "pyrosetta", lambda: run_pyrosetta_relax(inputs, config))
    validate_output(result)

    assert result.tool_id == "pyrosetta-relax"
    assert len(result.results) == 1
    assert isinstance(result.results[0].total_score, float)
