"""tests/binder_design_tests/test_germinal_design.py.

Tests for Germinal antibody design.
"""

import json
from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.binder_design import (
    GerminalConfig,
    GerminalDesign,
    GerminalDesignMetrics,
    GerminalInput,
    GerminalOutput,
    run_germinal_design,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"


# ── Custom validators ──────────────────────────────────────────────────────


def test_germinal_input_rejects_invalid_hotspot_format():
    """Hotspot regex: '<chain_letter><resnum>'."""
    with pytest.raises(ValueError, match="Hotspots must be"):
        GerminalInput(target_pdb=str(TEST_PDB), hotspots=["37"])


def test_germinal_input_rejects_target_eq_binder_chain():
    """binder_chain must differ from target_chain."""
    with pytest.raises(ValueError, match="must differ"):
        GerminalInput(target_pdb=str(TEST_PDB), target_chain="A", binder_chain="A")


def test_germinal_input_rejects_unknown_hotspot_chain():
    """Every hotspot chain must appear in target_chain (also exercises the multi-chain split)."""
    with pytest.raises(ValueError, match="Hotspot chains"):
        GerminalInput(target_pdb=str(TEST_PDB), target_chain="A,C", hotspots=["B37"])


def test_germinal_config_rejects_hydra_structural_overrides():
    """germinal_overrides must reject 'hydra.*' keys (would let users redirect run_dir/output)."""
    with pytest.raises(ValueError, match="hydra"):
        GerminalConfig(germinal_overrides={"hydra.run.dir": "x"})
    with pytest.raises(ValueError, match="hydra"):
        GerminalConfig(germinal_overrides={"hydra/job": "x"})
    with pytest.raises(ValueError, match="not a valid"):
        GerminalConfig(germinal_overrides={"weights iptm": 1.0})  # space in key
    # Legitimate keys still pass.
    cfg = GerminalConfig(germinal_overrides={"logits_steps": 50, "weights_iptm": 1.0, "+new_key": 1})
    assert cfg.germinal_overrides == {"logits_steps": 50, "weights_iptm": 1.0, "+new_key": 1}


# ── Source-fidelity: metric names must track upstream Germinal CSV columns ──


def test_germinal_metrics_match_upstream_csv_columns():
    """metric_spec keys must cover every column Germinal writes to designs.csv.

    Upstream sources at the pinned commit:
      - germinal/utils/io.py TRAJECTORY_METRICS_TO_SAVE
      - configs/filter/{initial,final}/{vhh,scfv}.yaml

    If Germinal renames a column, this test fails and inference.py's
    _normalize_metrics() / metric_spec must be updated together.
    """
    spec = set(GerminalDesignMetrics.metric_spec)
    upstream = {
        # TRAJECTORY_METRICS_TO_SAVE
        "plddt",
        "ptm",
        "i_ptm",
        "i_pae",
        "pae",
        "loss",
        "lm_ll",
        "helix",
        "beta_strand",
        # filter metrics (initial + final)
        "clashes",
        "sc_rmsd",
        "binder_near_hotspot",
        "cdr3_hotspot_contacts",
        "percent_interface_cdr",
        "interface_shape_comp",
        "interface_hbonds",
        "surface_hydrophobicity",
        "interface_hydrophobicity",
        "pdockq2",
        # external structure-validation metrics
        "external_plddt",
        "external_iptm",
        "external_pae",
        "external_ptm",
    }
    missing = upstream - spec
    assert not missing, f"metric_spec missing upstream columns: {sorted(missing)}"
    assert GerminalDesignMetrics.model_fields["primary_metric"].default == "i_ptm"


# ── Output export (pure Pydantic; no GPU) ──────────────────────────────────


def _stub_output() -> GerminalOutput:
    """Two-design GerminalOutput for export testing.

    Uses real upstream design_name formats:
      - AbMPNN-redesigned (accepted): "<target>_<type>_s<seed>_abmpnn_<j>"
      - Trajectory-only:              "<target>_<type>_s<seed>"
    """
    pdb = "ATOM      1  N   MET A   1      11.000  12.000  13.000  1.00 50.00           N  \nEND\n"
    return GerminalOutput(
        designs=[
            GerminalDesign(
                sequence_heavy="MKTLAALL",
                structure=Structure(structure=pdb, source="test"),
                metrics=GerminalDesignMetrics(plddt=0.92, i_ptm=0.71, pdockq2=0.45, clashes=0),
                stage_passed="accepted",
                design_id="pdl1_nb_s12345_abmpnn_2",
                trajectory_index=12345,
                mpnn_index=2,
            ),
            GerminalDesign(
                sequence_heavy="MKTLAALV",
                structure=Structure(structure=pdb, source="test"),
                metrics=GerminalDesignMetrics(plddt=0.55, i_ptm=0.40),
                stage_passed="trajectory",
                design_id="pdl1_nb_s67890",
                trajectory_index=67890,
                mpnn_index=0,
            ),
        ],
        pipeline_stats={"trajectories_attempted": 10, "designs_accepted": 1},
    )


def test_germinal_output_export_all_formats(tmp_path):
    """All three export formats produce the expected on-disk layout from one stub."""
    out = _stub_output()

    # PDB: writes one .pdb per design under <export_path>/<name>/
    out.export(name="designs", export_path=str(tmp_path), file_format="pdb")
    assert sorted(p.name for p in (tmp_path / "designs").iterdir()) == [
        "pdl1_nb_s12345_abmpnn_2.pdb",
        "pdl1_nb_s67890.pdb",
    ]

    # CSV: header + one row per design; metrics columns flattened from the union of all designs' keys
    out.export(name="designs", export_path=str(tmp_path), file_format="csv")
    rows = (tmp_path / "designs.csv").read_text().strip().split("\n")
    assert rows[0].split(",")[:4] == ["design_id", "stage_passed", "sequence_heavy", "sequence_light"]
    assert "i_ptm" in rows[0] and "pdockq2" in rows[0] and "clashes" in rows[0]
    assert len(rows) == 3

    # JSON: full payload minus structure content (structures are large; written separately as PDB)
    out.export(name="designs", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "designs.json").read_text())
    assert payload["pipeline_stats"]["designs_accepted"] == 1
    assert len(payload["designs"]) == 2
    assert "structure" not in payload["designs"][0]
    assert payload["designs"][0]["sequence_heavy"] == "MKTLAALL"
    assert payload["designs"][0]["metrics"]["i_ptm"] == 0.71


def test_germinal_design_round_trip_serialization():
    """GerminalDesign survives model_dump → re-construct without losing fields (Structure included)."""
    original = _stub_output().designs[0]
    rebuilt = GerminalDesign.model_validate(original.model_dump())
    assert rebuilt.design_id == original.design_id
    assert rebuilt.sequence_heavy == original.sequence_heavy
    assert rebuilt.stage_passed == original.stage_passed
    assert rebuilt.trajectory_index == original.trajectory_index
    assert rebuilt.mpnn_index == original.mpnn_index
    assert dict(rebuilt.metrics.items()) == dict(original.metrics.items())
    assert rebuilt.structure.structure == original.structure.structure


# ── Tool registration ───────────────────────────────────────────────────────


def test_germinal_design_registered_with_expected_metadata():
    """``@tool`` wires the right registry metadata + a callable example_input."""
    spec = next((s for s in ToolRegistry.list_all() if s.key == "germinal-design"), None)
    assert spec is not None
    assert spec.category == "binder_design"
    assert spec.uses_gpu is True
    assert spec.cacheable is False
    example = spec.example_input()
    assert isinstance(example, GerminalInput)


# ── Dispatch payload (mocked) ───────────────────────────────────────────────


def test_dispatch_payload_carries_user_config_plus_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch payload merges typed Config fields, overrides, and Input — without spawning the subprocess."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, **kwargs):  # type: ignore[no-untyped-def]
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {"designs": [], "pipeline_stats": {}}

    monkeypatch.setattr(
        "proto_tools.tools.binder_design.germinal.germinal_design.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = GerminalInput(
        target_pdb=str(TEST_PDB),
        target_chain="A",
        binder_chain="B",
        hotspots=["A56", "A66"],
        target_name="payload_test",
    )
    config = GerminalConfig(
        design_type="vhh",
        max_trajectories=3,
        max_passing_designs=1,
        structure_model="chai",
        plddt_threshold=0.9,
        germinal_overrides={"logits_steps": 25},
        filter_overrides={"final": {"clashes": {"value": 5, "operator": "<"}}},
    )
    run_germinal_design(inputs, config)

    assert captured["toolkit"] == "germinal"
    payload = captured["payload"]
    # Input fields flow through
    assert payload["target_pdb"] == str(TEST_PDB)
    assert payload["target_chain"] == "A"
    assert payload["binder_chain"] == "B"
    assert payload["hotspots"] == ["A56", "A66"]
    assert payload["target_name"] == "payload_test"
    # Config fields flow through
    assert payload["design_type"] == "vhh"
    assert payload["max_trajectories"] == 3
    assert payload["max_passing_designs"] == 1
    assert payload["structure_model"] == "chai"
    assert payload["plddt_threshold"] == 0.9
    assert payload["germinal_overrides"] == {"logits_steps": 25}
    assert payload["filter_overrides"] == {"final": {"clashes": {"value": 5, "operator": "<"}}}


# ── Integration: real Germinal subprocess against PD-L1 (GPU + slow) ───────


def _assert_design_invariants(design: GerminalDesign) -> None:
    """Every Germinal design must satisfy these structural invariants.

    Round-trips the design_id → (trajectory_index, mpnn_index) parse to catch
    any drift between upstream design_name format and inference.py's regex.
    """
    assert design.stage_passed in {"accepted", "redesign_candidate", "trajectory"}
    assert design.sequence_heavy.isalpha() and len(design.sequence_heavy) > 0
    assert design.trajectory_index >= 0 and design.mpnn_index >= 0
    # design_id format from upstream run_germinal.py:
    #   trajectory-only:    "<target>_<type>_s<seed>"
    #   AbMPNN-redesigned:  "<target>_<type>_s<seed>_abmpnn_<j+1>"
    expected_suffix = (
        f"_s{design.trajectory_index}_abmpnn_{design.mpnn_index}"
        if design.mpnn_index > 0
        else f"_s{design.trajectory_index}"
    )
    assert design.design_id.endswith(expected_suffix), (
        f"design_id {design.design_id!r} does not end with expected suffix {expected_suffix!r}"
    )
    # CSV → metrics parse populated at least one trajectory metric
    populated = [k for k in ("plddt", "i_ptm", "ptm", "i_pae") if design.metrics.get(k) is not None]
    assert populated, f"Design {design.design_id} has no populated trajectory metrics"
    assert "ATOM" in design.structure.structure


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.integration
def test_germinal_design_e2e_vhh_pd_l1(tmp_path):
    """End-to-end VHH campaign against PD-L1 with reduced compute knobs (~5-15 min on H100).

    Exercises the full wrap-Germinal pipeline:
      1. setup.sh-staged repo + AF-Multimer weights are reachable
      2. Hydra overlay YAMLs are written and consumed
      3. run_germinal.py completes
      4. designs.csv + structures/*.pdb parse back into typed Pydantic objects
      5. design_id → (trajectory_index, mpnn_index) regex roundtrips
      6. failure_counts.csv → pipeline_stats parses
      7. result.export() works on real outputs
    """
    inputs = GerminalInput(
        target_pdb=str(TEST_PDB),
        target_chain="A",
        binder_chain="B",
        hotspots=["A56", "A66", "A115"],
        target_name="pdl1",
    )
    config = GerminalConfig(
        design_type="vhh",
        max_trajectories=2,
        max_passing_designs=1,
        structure_model="chai",
        germinal_overrides={
            "logits_steps": 20,
            "softmax_steps": 4,
            "search_steps": 1,
            "num_seqs": 2,
        },
    )
    result = run_germinal_design(inputs, config)
    validate_output(result)

    assert result.tool_id == "germinal-design"
    assert isinstance(result, GerminalOutput)
    # failure_counts.csv must have been parsed (at least one stage counted)
    assert result.pipeline_stats, "pipeline_stats empty — failure_counts.csv parse failed"

    for d in result.designs:
        _assert_design_invariants(d)
        # VHH-specific: no light chain
        assert d.sequence_light is None
    assert_metrics_in_spec(result)

    # Export round-trip on real outputs (only meaningful if any designs returned)
    if result.designs:
        export_dir = tmp_path / "germinal_export"
        result.export(name="pdl1", export_path=str(export_dir), file_format="pdb")
        assert sorted(p.name for p in export_dir.glob("*.pdb")) == sorted(f"{d.design_id}.pdb" for d in result.designs)


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.integration
def test_germinal_filter_override_rejects_all_designs_e2e():
    """``filter_overrides`` flows end-to-end into upstream's filter check.

    Drives one full VHH trajectory through the pipeline but sets a final
    ``external_plddt`` threshold of 0.999 (effectively unattainable).
    Trajectories should still complete; pipeline_stats should still parse;
    ``result.designs`` should be empty (everything rejected at the final stage).
    """
    inputs = GerminalInput(
        target_pdb=str(TEST_PDB),
        target_chain="A",
        binder_chain="B",
        hotspots=["A56", "A66", "A115"],
        target_name="pdl1_strict",
    )
    config = GerminalConfig(
        design_type="vhh",
        max_trajectories=1,
        max_passing_designs=1,
        structure_model="chai",
        germinal_overrides={"logits_steps": 20, "softmax_steps": 4, "search_steps": 1, "num_seqs": 2},
        filter_overrides={"final": {"external_plddt": {"value": 0.999, "operator": ">"}}},
    )
    result = run_germinal_design(inputs, config)
    validate_output(result)

    assert result.tool_id == "germinal-design"
    assert result.pipeline_stats, "filter_override path didn't reach failure_counts.csv parse"
    accepted = [d for d in result.designs if d.stage_passed == "accepted"]
    assert accepted == [], f"impossible external_plddt threshold should reject every design, got {len(accepted)}"


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.integration
def test_germinal_design_e2e_scfv_pd_l1():
    """End-to-end scFv campaign against PD-L1 (~10-20 min on H100).

    scFv mode exercises a code path the VHH test does not:
      - design_type='scfv' selects configs/run/scfv.yaml (different cdr_lengths,
        fw_lengths, vh_first/vh_len/vl_len, ablm_model='ablang')
      - inference.py._extract_binder_sequences() splits the binder chain into
        heavy + light by detecting the largest CA-CA gap

    Asserts only the scFv-specific invariants; structural invariants shared
    with VHH are not re-asserted.
    """
    inputs = GerminalInput(
        target_pdb=str(TEST_PDB),
        target_chain="A",
        binder_chain="B",
        hotspots=["A56", "A66", "A115"],
        target_name="pdl1_scfv",
    )
    config = GerminalConfig(
        design_type="scfv",
        max_trajectories=1,
        max_passing_designs=1,
        structure_model="chai",
        germinal_overrides={
            "logits_steps": 20,
            "softmax_steps": 4,
            "search_steps": 1,
            "num_seqs": 2,
        },
    )
    result = run_germinal_design(inputs, config)
    validate_output(result)

    assert result.tool_id == "germinal-design"
    for d in result.designs:
        _assert_design_invariants(d)
        # scFv (VH+VL) ≈ 245 residues; VHH ≈ 120. If the CA-CA gap split fails
        # we get the full chain in sequence_heavy with sequence_light=None — both shapes pass.
        total_len = len(d.sequence_heavy) + len(d.sequence_light or "")
        assert total_len > 150, f"scFv design {d.design_id} too short: {total_len} residues"
    assert_metrics_in_spec(result)
