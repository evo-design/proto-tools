"""Tests for the pDockQ2 interface quality scoring tool.

End-to-end tests marked ``@pytest.mark.integration`` run the real algorithm
against real PDB + PAE fixtures (no mocks). Unmarked tests cover the custom
input validators and the ported sigmoid constants.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_scoring.pdockq2 import PDockQ2Config, PDockQ2Input, run_pdockq2
from proto_tools.tools.structure_scoring.pdockq2.pdockq2 import _pmidockq_sigmoid, example_input
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

FIXTURE_DIR = Path(__file__).parent.parent.parent / "proto_tools" / "tools" / "structure_scoring" / "pdockq2"


def _bundled_structure(*, with_pae: bool = True, b_factor_type: BFactorType = BFactorType.PLDDT) -> Structure:
    pdb_path = FIXTURE_DIR / "example_input_fixture.pdb"
    metrics: dict[str, object] = {}
    if with_pae:
        metrics["pae_matrix"] = json.loads((FIXTURE_DIR / "example_input_fixture_pae.json").read_text())
    return Structure.from_file(pdb_path, b_factor_type=b_factor_type, metrics=metrics)


def _write_ca_pdb(path: Path, residues: list[tuple[str, int, tuple[float, float, float], float]]) -> None:
    """Write a CA-only PDB with rows ``(chain_id, residue_index, (x,y,z), bfactor)``."""
    lines = [
        f"ATOM  {i:5d}  CA  ALA {chain}{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C\n"
        for i, (chain, resi, (x, y, z), b) in enumerate(residues, start=1)
    ]
    lines.append("END\n")
    path.write_text("".join(lines))


# ── Custom input validators (one parametrized test) ──────────────────────────


def _pae_shape_mismatch_structure() -> Structure:
    return Structure.from_file(
        FIXTURE_DIR / "example_input_fixture.pdb",
        b_factor_type=BFactorType.PLDDT,
        metrics={"pae_matrix": [[0.0] * 4 for _ in range(4)]},
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"structure": lambda: _bundled_structure(b_factor_type=BFactorType.UNSPECIFIED)}, "per_residue_plddt"),
        ({"structure": lambda: _bundled_structure(with_pae=False)}, r"metrics\['pae_matrix'\]"),
        ({"structure": _pae_shape_mismatch_structure}, "does not match residue count"),
        ({"binder_chain": "Z"}, "not found in structure"),
        ({"target_chains": ["A", "B"]}, "must not appear in target_chains"),
        ({"binder_chain": "AA"}, "single character"),
    ],
    ids=["no_plddt", "no_pae", "pae_shape_mismatch", "unknown_chain", "binder_in_target", "multichar_chain"],
)
def test_input_validator_rejects(overrides, message):
    kwargs = {"structure": _bundled_structure(), "binder_chain": "A", "target_chains": ["B"]}
    for key, value in overrides.items():
        kwargs[key] = value() if callable(value) else value
    with pytest.raises(ValidationError, match=message):
        PDockQ2Input(**kwargs)


# ── Sigmoid constants (unit test of the ported formula) ──────────────────────


def test_pmidockq_sigmoid_pins_zhu_2023_constants():
    """Pin Zhu 2023 fit (L=1.31034849, x0=84.7326239, k=0.0747157696, b=0.00501886443)."""
    assert _pmidockq_sigmoid(0.0) == pytest.approx(
        1.31034849 / (1.0 + np.exp(0.0747157696 * 84.7326239)) + 0.00501886443
    )
    assert _pmidockq_sigmoid(84.7326239) == pytest.approx(1.31034849 / 2.0 + 0.00501886443)


# ── Integration: real algorithm runs against real PDB + PAE fixtures ─────────


@pytest.mark.integration
def test_run_pdockq2_matches_hand_computation():
    """End-to-end algorithm pin on the bundled fixture at the 10 A default cutoff.

    Fixture: chain A (binder, pLDDT=90) and chain B (target, pLDDT=70), 4 residues
    each. At 10 A, 7 cross-chain pairs. PAE is 3.0 on the A[0:2] x B[0:2] block
    (4 of the 7 contacts) and 15.0 elsewhere.
    """
    out = run_pdockq2(example_input(), PDockQ2Config())

    assert out.success
    assert out.tool_id == "pdockq2"
    assert out.metrics.num_interface_contacts == 7
    assert out.metrics.avg_interface_plddt == pytest.approx(70.0)

    pae_vals = np.array([3.0, 3.0, 15.0, 3.0, 3.0, 15.0, 15.0])
    expected_norm_pae = float(np.mean(1.0 / (1.0 + (pae_vals / 10.0) ** 2)))
    assert out.metrics.avg_interface_pae == pytest.approx(expected_norm_pae)
    assert out.metrics.pdockq2 == pytest.approx(_pmidockq_sigmoid(70.0 * expected_norm_pae))

    by_chain = {row.chain_id: row for row in out.metrics.interfaces}
    assert by_chain["A"].if_plddt == pytest.approx(90.0)
    assert by_chain["B"].if_plddt == pytest.approx(70.0)
    assert by_chain["A"].neighbor_chains == "B"
    assert by_chain["B"].neighbor_chains == "A"
    assert_metrics_in_spec(out)


@pytest.mark.integration
def test_export_json_round_trip(tmp_path):
    out = run_pdockq2(example_input(), PDockQ2Config())
    out.export(name="pdockq2_result", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "pdockq2_result.json").read_text())
    assert payload["pdockq2"] == pytest.approx(out.metrics.pdockq2)
    assert payload["num_interface_contacts"] == out.metrics.num_interface_contacts


# ── Integration: target_chains aggregation on a 3-chain fixture with a decoy ──


def _three_chain_structure(tmp_path: Path) -> Structure:
    """Binder A contacts both target B and decoy C within the 8 A default cutoff."""
    residues = [
        ("A", 1, (0.0, 0.0, 0.0), 90.0),
        ("A", 2, (3.8, 0.0, 0.0), 90.0),
        ("B", 1, (0.0, 5.0, 0.0), 70.0),
        ("B", 2, (3.8, 5.0, 0.0), 70.0),
        ("C", 1, (0.0, -6.0, 0.0), 50.0),
        ("C", 2, (3.8, -6.0, 0.0), 50.0),
    ]
    _write_ca_pdb(tmp_path / "three_chain.pdb", residues)
    pae = [[15.0] * 6 for _ in range(6)]
    for i in (0, 1):
        for j in (2, 3):
            pae[i][j] = pae[j][i] = 3.0
        for j in (4, 5):
            pae[i][j] = pae[j][i] = 8.0
    return Structure.from_file(
        tmp_path / "three_chain.pdb", b_factor_type=BFactorType.PLDDT, metrics={"pae_matrix": pae}
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "target_chains,scoring_chains",
    [(["B"], ["B"]), (["B", "C"], ["B", "C"]), ("B,C", ["B", "C"])],
    ids=["single_target_excludes_decoy", "multichain_target_list_includes_both", "multichain_target_csv_includes_both"],
)
def test_target_chains_aggregation(tmp_path, target_chains, scoring_chains):
    """Only chains listed in ``target_chains`` contribute to the final pdockq2 average."""
    out = run_pdockq2(
        PDockQ2Input(structure=_three_chain_structure(tmp_path), binder_chain="A", target_chains=target_chains),
        PDockQ2Config(),
    )
    by_chain = {row.chain_id: row for row in out.metrics.interfaces}
    expected = float(np.mean([by_chain[c].pmidockq for c in scoring_chains]))
    assert out.metrics.pdockq2 == pytest.approx(expected)
    # Decoy C is always detected as a neighbor of A even when it's excluded from the score.
    assert by_chain["C"].neighbor_chains == "A"


@pytest.mark.integration
def test_interface_plddt_is_contact_pair_weighted(tmp_path):
    """Heterogeneous pLDDT: Zhu 2023 weights IF_plddt per contact pair, not per residue.

    B[0] (pLDDT=60) contacts both A residues (2 pairs); B[1] (pLDDT=90) contacts
    only A[1] (1 pair). Zhu-exact mean is ``(60+60+90)/3 = 70``; per-residue would
    be ``(60+90)/2 = 75``.
    """
    _write_ca_pdb(
        tmp_path / "hetero.pdb",
        [
            ("A", 1, (0.0, 0.0, 0.0), 50.0),
            ("A", 2, (4.0, 0.0, 0.0), 50.0),
            ("B", 1, (2.0, 3.0, 0.0), 60.0),
            ("B", 2, (8.0, 8.0, 0.0), 90.0),
        ],
    )
    pae = [[6.0] * 4 for _ in range(4)]
    structure = Structure.from_file(
        tmp_path / "hetero.pdb", b_factor_type=BFactorType.PLDDT, metrics={"pae_matrix": pae}
    )

    out = run_pdockq2(PDockQ2Input(structure=structure, binder_chain="A", target_chains=["B"]), PDockQ2Config())
    by_chain = {row.chain_id: row for row in out.metrics.interfaces}
    assert by_chain["B"].if_plddt == pytest.approx(70.0)


@pytest.mark.integration
def test_disjoint_chains_score_zero_with_warning(tmp_path, caplog):
    """No cross-chain contacts → pdockq2 = 0.0 with a warning logged."""
    pdb_path = tmp_path / "disjoint.pdb"
    _write_ca_pdb(
        pdb_path,
        [
            ("A", 1, (0.0, 0.0, 0.0), 90.0),
            ("A", 2, (0.0, 100.0, 0.0), 90.0),
            ("B", 1, (0.0, -200.0, 0.0), 70.0),
            ("B", 2, (0.0, -300.0, 0.0), 70.0),
        ],
    )
    structure = Structure.from_file(
        pdb_path, b_factor_type=BFactorType.PLDDT, metrics={"pae_matrix": [[15.0] * 4 for _ in range(4)]}
    )

    with caplog.at_level(logging.WARNING, logger="proto_tools.tools.structure_scoring.pdockq2.pdockq2"):
        out = run_pdockq2(PDockQ2Input(structure=structure, binder_chain="A", target_chains=["B"]), PDockQ2Config())

    assert out.metrics.pdockq2 == 0.0
    assert out.metrics.num_interface_contacts == 0
    assert any("pdockq2=0.0" in rec.message for rec in caplog.records)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pdockq2")
@pytest.mark.slow
def test_pdockq2_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pdockq2: 20 sequential calls on the bundled fixture (cold + warm).

    pdockq2 is an in-process numpy compute tool with no persistent worker, so
    we time both passes directly. The "cold" pass captures any first-call
    lazy-import cost; the "warm" pass measures pure compute.
    """
    import time

    inp = example_input()
    config = PDockQ2Config()

    def run_batch():
        last = None
        for _ in range(20):
            last = run_pdockq2(inp, config)
        return last

    t0 = time.perf_counter()
    _ = run_batch()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = run_batch()
    warm = time.perf_counter() - t0
    request.node.user_properties.append(("cold_seconds", cold))
    request.node.user_properties.append(("warm_seconds", warm))

    validate_output(result)
    assert result.tool_id == "pdockq2"
    assert result.metrics.pdockq2 >= 0.0
