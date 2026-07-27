"""Tests for AutoDock Vina rigid-receptor molecular docking."""

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from proto_tools.tools import ToolRegistry
from proto_tools.tools.molecular_docking.vina import (
    VinaDockingConfig,
    VinaDockingInput,
    VinaDockingOutput,
    VinaDockingPose,
    VinaDockingPoseMetrics,
    VinaReferenceLigandBox,
    VinaSearchBox,
    run_vina_docking,
)
from proto_tools.tools.molecular_docking.vina.standalone import run as standalone
from proto_tools.utils import ToolInstance
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_TOOLKIT_DIR = Path(__file__).resolve().parents[2] / "proto_tools" / "tools" / "molecular_docking" / "vina"
_RECEPTOR_PATH = _TOOLKIT_DIR / "example_receptor_1iep.pdb"
_REFERENCE_LIGAND_PATH = _TOOLKIT_DIR / "example_reference_imatinib.pdb"
_IMATINIB_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
_CANONICAL_BOX = VinaSearchBox(
    center=(15.190, 53.903, 16.917),
    size=(20.0, 20.0, 20.0),
)

_POSE_PDBQT = """MODEL 1
REMARK VINA RESULT:    -8.250      0.000      0.000
ATOM      1  C1  LIG     1      15.000  54.000  17.000  1.00  0.00     0.000 C
ENDMDL
"""
_POSE_SDF = """ligand
  proto-tools

  1  0  0  0  0  0  0  0  0  0999 V2000
   15.0000   54.0000   17.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


def _example_input() -> VinaDockingInput:
    """Build the bundled c-Abl/imatinib input."""
    return VinaDockingInput(
        receptor=_RECEPTOR_PATH,  # type: ignore[arg-type]
        ligand=_IMATINIB_SMILES,  # type: ignore[arg-type]
        search_box=_CANONICAL_BOX,
    )


def _fake_output(seed: int = 7) -> dict[str, Any]:
    """Return a valid one-pose standalone payload."""
    return {
        "poses": [
            {
                "rank": 1,
                "affinity": -8.25,
                "rmsd_lower_bound": 0.0,
                "rmsd_upper_bound": 0.0,
                "sdf": _POSE_SDF,
                "pdbqt": _POSE_PDBQT,
            }
        ],
        "seed": seed,
        "poses_sdf": _POSE_SDF,
        "poses_pdbqt": _POSE_PDBQT,
        "vina_version": "1.2.7",
        "meeko_version": "0.7.1",
        "rdkit_version": "2025.09.6",
        "grid_point_count": 166375,
        "receptor_preparation": {
            "ignored_residue_ids": [],
            "template_assignments": [
                {
                    "residue_id": "A:246",
                    "input_residue_name": "HIS",
                    "assigned_template": "HIE",
                }
            ],
        },
        "ligand_preparation": {
            "optimization_method": "MMFF94",
            "optimization_converged": True,
            "optimization_status": 0,
            "optimization_attempts": 2,
            "optimization_iteration_limits": [200, 500],
        },
        "warnings": [],
    }


def _crystallographic_heavy_atom_rmsd(pose_sdf: str) -> float:
    """Calculate symmetry-aware pose RMSD in the receptor frame without superposition."""
    template = Chem.MolFromSmiles(_IMATINIB_SMILES)
    reference_raw = Chem.MolFromPDBFile(
        str(_REFERENCE_LIGAND_PATH),
        sanitize=False,
        removeHs=True,
        proximityBonding=False,
    )
    pose = Chem.MolFromMolBlock(pose_sdf, removeHs=True)
    assert template is not None
    assert reference_raw is not None
    assert pose is not None

    reference = AllChem.AssignBondOrdersFromTemplate(template, reference_raw)
    Chem.SanitizeMol(reference)
    return float(
        rdMolAlign.CalcRMS(
            pose,
            reference,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )


# Validation and registry


def test_vina_tool_is_registered_with_expected_contract() -> None:
    """The registry exposes Vina as a cacheable stochastic CPU tool."""
    spec = ToolRegistry.get("vina-docking")

    assert spec.key == "vina-docking"
    assert spec.category == "molecular_docking"
    assert spec.input_model is VinaDockingInput
    assert spec.config_model is VinaDockingConfig
    assert spec.output_model is VinaDockingOutput
    assert spec.metrics_model is VinaDockingPoseMetrics
    assert spec.uses_gpu is False
    assert spec.cacheable is True
    assert spec.stochastic is True
    assert isinstance(ToolRegistry.get_example_input("vina-docking"), VinaDockingInput)
    assert ToolRegistry.get_links("vina-docking") == {"github": "https://github.com/ccsb-scripps/AutoDock-Vina"}
    assert VinaDockingPoseMetrics.metric_spec["rmsd_lower_bound"]["better_values_are"] == "context-dependent"
    assert VinaDockingPoseMetrics.metric_spec["rmsd_upper_bound"]["better_values_are"] == "context-dependent"


def test_vina_input_accepts_bare_smiles() -> None:
    """A bare SMILES ligand is normalized to a single Fragment."""
    inputs = _example_input()

    assert inputs.ligand.smiles is not None
    assert inputs.ligand.heavy_atom_count == 37
    assert inputs.search_box == _CANONICAL_BOX


def test_vina_input_round_trips_without_mutating_dumped_dict() -> None:
    """Reconstruction preserves an input's serialized transport payload."""
    inputs = _example_input()
    dumped = inputs.model_dump(exclude_none=True)
    expected = deepcopy(dumped)

    reconstructed = VinaDockingInput(**dumped)

    assert dumped == expected
    assert reconstructed.model_dump(exclude_none=True) == expected


@pytest.mark.parametrize("ligand", ["not-a-smiles", "CC.C"])
def test_vina_input_rejects_invalid_or_multicomponent_smiles(ligand: str) -> None:
    """Ligands must be one parseable molecular graph."""
    with pytest.raises(ValidationError):
        VinaDockingInput(
            receptor=_RECEPTOR_PATH,  # type: ignore[arg-type]
            ligand=ligand,  # type: ignore[arg-type]
            search_box=_CANONICAL_BOX,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": (math.inf, 0.0, 0.0), "size": (20.0, 20.0, 20.0)},
        {"center": (0.0, 0.0, 0.0), "size": (20.0, 0.0, 20.0)},
        {"center": (0.0, 0.0, 0.0), "size": (20.0, math.nan, 20.0)},
        {"center": (0.0, 0.0, 0.0), "size": (20.0, 100.1, 20.0)},
    ],
)
def test_vina_search_box_rejects_nonfinite_or_nonpositive_values(kwargs: dict[str, Any]) -> None:
    """Every search-box coordinate must be finite and every dimension positive."""
    with pytest.raises(ValidationError):
        VinaSearchBox(**kwargs)


def test_reference_ligand_box_resolves_bundled_imatinib_coordinates() -> None:
    """Reference-ligand bounds produce the expected center and padded dimensions."""
    search_box = VinaReferenceLigandBox(
        reference_ligand=_REFERENCE_LIGAND_PATH,  # type: ignore[arg-type]
        padding=4.0,
    ).resolve()

    assert search_box.center == pytest.approx((15.190, 53.9025, 16.917))
    assert search_box.size == pytest.approx((16.664, 24.739, 21.526))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exhaustiveness": 0},
        {"exhaustiveness": 2**31},
        {"num_poses": 0},
        {"num_poses": 2**31},
        {"energy_range": 0.0},
        {"energy_range": math.inf},
        {"min_rmsd": 0.0},
        {"min_rmsd": math.nan},
        {"max_evaluations": -1},
        {"max_evaluations": 2**31},
        {"cpu": -1},
        {"cpu": 2**31},
        {"grid_spacing": 0.099},
        {"grid_spacing": math.inf},
        {"seed": 0},
        {"seed": 2**31},
    ],
)
def test_vina_config_rejects_out_of_range_values(kwargs: dict[str, Any]) -> None:
    """Search controls reject values outside Vina's supported ranges."""
    with pytest.raises(ValidationError):
        VinaDockingConfig(**kwargs)


def test_vina_minimal_config_reduces_search_cost() -> None:
    """The infrastructure smoke-test config performs one single-threaded search."""
    config = VinaDockingConfig.minimal(seed=9)

    assert config.exhaustiveness == 1
    assert config.num_poses == 1
    assert config.cpu == 1
    assert config.seed == 9


def test_vina_rejects_a_grid_allocation_above_the_safety_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid box and spacing combination cannot trigger an excessive native allocation."""

    def unexpected_dispatch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("dispatch must not run for an oversized grid")

    monkeypatch.setattr(ToolInstance, "dispatch", unexpected_dispatch)
    inputs = VinaDockingInput(
        receptor=_RECEPTOR_PATH,  # type: ignore[arg-type]
        ligand=_IMATINIB_SMILES,  # type: ignore[arg-type]
        search_box=VinaSearchBox(center=(0.0, 0.0, 0.0), size=(100.0, 100.0, 100.0)),
    )

    with pytest.raises(ValueError, match="grid points"):
        run_vina_docking(inputs, VinaDockingConfig.minimal(grid_spacing=0.1, seed=7))


# Wrapper and export behavior


def test_vina_wrapper_dispatches_normalized_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public wrapper serializes receptor, ligand, box, and all search controls."""
    captured: dict[str, Any] = {}

    def fake_dispatch(
        cls: type[ToolInstance],
        toolkit: str,
        input_dict: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["toolkit"] = toolkit
        captured["input"] = input_dict
        captured["kwargs"] = kwargs
        return _fake_output(seed=input_dict["config"]["seed"])

    monkeypatch.setattr(ToolInstance, "dispatch", classmethod(fake_dispatch))
    inputs = _example_input()
    config = VinaDockingConfig(
        scoring_function="vinardo",
        exhaustiveness=3,
        num_poses=2,
        energy_range=5.0,
        min_rmsd=1.5,
        max_evaluations=500,
        cpu=2,
        grid_spacing=0.5,
        allow_bad_residues=True,
        seed=17,
        verbose=2,
    )

    output = run_vina_docking(inputs, config)
    payload = captured["input"]

    assert captured["toolkit"] == "vina"
    assert "\nATOM" in payload["receptor_pdb"]
    assert payload["receptor_pdb"].endswith("END\n")
    assert payload["ligand_smiles"] == inputs.ligand.smiles
    assert payload["search_box"] == {
        "mode": "coordinates",
        "center": (15.190, 53.903, 16.917),
        "size": (20.0, 20.0, 20.0),
    }
    assert payload["config"] == {
        "scoring_function": "vinardo",
        "exhaustiveness": 3,
        "num_poses": 2,
        "energy_range": 5.0,
        "min_rmsd": 1.5,
        "max_evaluations": 500,
        "cpu": 2,
        "grid_spacing": 0.5,
        "allow_bad_residues": True,
        "seed": 17,
        "verbose": 2,
    }
    assert captured["kwargs"]["config"] is config
    assert output.seed == 17
    assert output.scoring_function == "vinardo"
    assert output.metadata["vina_version"] == "1.2.7"
    assert output.metadata["meeko_version"] == "0.7.1"
    assert output.metadata["rdkit_version"] == "2025.09.6"
    assert output.metadata["requested_num_poses"] == 2
    assert output.metadata["returned_num_poses"] == 1
    assert output.metadata["energy_range"] == 5.0
    assert output.metadata["min_rmsd"] == 1.5
    assert output.metadata["max_evaluations"] == 500
    assert output.metadata["cpu"] == 2
    assert output.metadata["grid_spacing"] == 0.5
    assert output.metadata["allow_bad_residues"] is True
    assert output.metadata["ignored_receptor_residues"] == []
    assert output.metadata["receptor_template_assignments"] == [
        {
            "residue_id": "A:246",
            "input_residue_name": "HIS",
            "assigned_template": "HIE",
        }
    ]
    assert output.metadata["ligand_optimization_method"] == "MMFF94"
    assert output.metadata["ligand_optimization_converged"] is True
    assert output.metadata["ligand_optimization_attempts"] == 2
    assert output.metadata["ligand_optimization_iteration_limits"] == [200, 500]
    assert output.poses[0].metrics.affinity == -8.25


def test_vina_wrapper_concretizes_an_omitted_zero_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An omitted seed is converted to Vina's positive signed 32-bit range."""
    dispatched_seed: int | None = None

    def fake_dispatch(
        cls: type[ToolInstance],
        toolkit: str,
        input_dict: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal dispatched_seed
        dispatched_seed = input_dict["config"]["seed"]
        return _fake_output(seed=dispatched_seed)

    monkeypatch.setattr(ToolInstance, "dispatch", classmethod(fake_dispatch))
    monkeypatch.setattr(VinaDockingConfig, "get_random_int", staticmethod(lambda: 0))

    output = run_vina_docking(_example_input(), VinaDockingConfig.minimal())

    assert dispatched_seed == 1
    assert output.seed == 1


def test_vina_output_exports_all_supported_formats(tmp_path: Path) -> None:
    """SDF, PDBQT, CSV, and JSON exports preserve poses and provenance."""
    output = VinaDockingOutput(
        poses=[
            VinaDockingPose(
                rank=1,
                metrics=VinaDockingPoseMetrics(
                    affinity=-8.25,
                    rmsd_lower_bound=0.0,
                    rmsd_upper_bound=0.0,
                ),
                sdf=_POSE_SDF,
                pdbqt=_POSE_PDBQT,
            )
        ],
        seed=7,
        search_box=_CANONICAL_BOX,
        scoring_function="vina",
        poses_sdf=_POSE_SDF,
        poses_pdbqt=_POSE_PDBQT,
    )

    for file_format in output.output_format_options:
        output.export("docking", tmp_path, file_format=file_format)

    assert (tmp_path / "docking.sdf").read_text().rstrip() == _POSE_SDF.rstrip()
    assert (tmp_path / "docking.pdbqt").read_text().rstrip() == _POSE_PDBQT.rstrip()

    with (tmp_path / "docking.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "rank": "1",
            "affinity": "-8.25",
            "rmsd_lower_bound": "0.0",
            "rmsd_upper_bound": "0.0",
            "seed": "7",
            "scoring_function": "vina",
            "center_x": "15.19",
            "center_y": "53.903",
            "center_z": "16.917",
            "size_x": "20.0",
            "size_y": "20.0",
            "size_z": "20.0",
        }
    ]
    exported_json = json.loads((tmp_path / "docking.json").read_text())
    assert exported_json["seed"] == 7
    assert exported_json["poses"][0]["metrics"]["affinity"] == -8.25


# Standalone parsing


def test_split_pdbqt_models_retains_complete_model_blocks() -> None:
    """Multi-pose PDBQT is split without dropping MODEL or ENDMDL records."""
    second_pose = _POSE_PDBQT.replace("MODEL 1", "MODEL 2").replace("-8.250", "-7.500")

    models = standalone._split_pdbqt_models(_POSE_PDBQT + second_pose)

    assert len(models) == 2
    assert models[0].startswith("MODEL 1")
    assert models[0].endswith("ENDMDL\n")
    assert models[1].startswith("MODEL 2")


def test_split_sdf_records_restores_delimiters() -> None:
    """Combined SDF payloads yield one delimiter-terminated string per pose."""
    records = standalone._split_sdf_records(_POSE_SDF + _POSE_SDF.replace("ligand", "ligand-2", 1))

    assert len(records) == 2
    assert all(record.endswith("$$$$\n") for record in records)
    assert all(Chem.MolFromMolBlock(record, removeHs=False) is not None for record in records)


def test_pose_metrics_parses_vina_result_remark() -> None:
    """Affinity and RMSD bounds are parsed from a Vina MODEL remark."""
    assert standalone._pose_metrics(_POSE_PDBQT) == (-8.25, 0.0, 0.0)


def test_pose_metrics_rejects_missing_result_remark() -> None:
    """A pose without score metadata fails instead of returning incomplete metrics."""
    with pytest.raises(ValueError, match="REMARK VINA RESULT"):
        standalone._pose_metrics("MODEL 1\nENDMDL\n")


def test_standalone_grid_validation_rejects_excessive_allocations() -> None:
    """The worker independently guards the native Vina map allocation."""
    with pytest.raises(ValueError, match="grid points"):
        standalone._validated_grid_point_count(
            center=[0.0, 0.0, 0.0],
            size=[100.0, 100.0, 100.0],
            spacing=0.1,
        )


# Real managed execution


@pytest.mark.integration
def test_vina_redocks_imatinib_through_tool_instance() -> None:
    """Dock imatinib into the 1IEP c-Abl pocket through the managed worker."""
    output = run_vina_docking(
        _example_input(),
        VinaDockingConfig(
            exhaustiveness=4,
            num_poses=5,
            energy_range=5.0,
            cpu=1,
            seed=7,
        ),
    )

    validate_output(output)
    assert_metrics_in_spec(output)
    assert output.tool_id == "vina-docking"
    assert output.seed == 7
    assert output.search_box == _CANONICAL_BOX
    assert output.scoring_function == "vina"
    assert output.metadata["vina_version"] == "1.2.7"
    assert output.metadata["meeko_version"] == "0.7.1"
    assert output.metadata["rdkit_version"]
    assert output.metadata["ignored_receptor_residues"] == []
    assert output.metadata["receptor_template_assignments"] == [
        {
            "residue_id": residue_id,
            "input_residue_name": "HIS",
            "assigned_template": "HIE",
        }
        for residue_id in ("A:246", "A:295", "A:361", "A:375", "A:396", "A:490")
    ]
    assert output.metadata["ligand_optimization_method"] == "MMFF94"
    assert output.metadata["ligand_optimization_converged"] is True
    assert 1 <= len(output.poses) <= 5
    assert output.poses[0].rank == 1
    assert output.poses[0].metrics.affinity < -8.0
    assert output.poses[0].metrics.rmsd_lower_bound == 0.0
    assert min(_crystallographic_heavy_atom_rmsd(pose.sdf) for pose in output.poses) < 2.0
    assert output.poses[0].sdf.endswith("$$$$\n")
    assert output.poses[0].pdbqt.startswith("MODEL 1")


@pytest.mark.integration
def test_vina_supports_vinardo_with_a_reference_derived_box() -> None:
    """Vinardo docking resolves its search box from the bundled crystal ligand."""
    inputs = VinaDockingInput(
        receptor=_RECEPTOR_PATH,  # type: ignore[arg-type]
        ligand=_IMATINIB_SMILES,  # type: ignore[arg-type]
        search_box=VinaReferenceLigandBox(
            reference_ligand=_REFERENCE_LIGAND_PATH,  # type: ignore[arg-type]
            padding=4.0,
        ),
    )

    output = run_vina_docking(
        inputs,
        VinaDockingConfig.minimal(scoring_function="vinardo", seed=19),
    )

    validate_output(output)
    assert_metrics_in_spec(output)
    assert output.scoring_function == "vinardo"
    assert output.search_box == inputs.search_box.resolve()
    assert output.seed == 19
    assert output.poses[0].metrics.affinity < -4.0
    assert output.metadata["scoring_function"] == "vinardo"
    assert output.metadata["requested_num_poses"] == 1
    assert output.metadata["returned_num_poses"] == 1


@pytest.mark.benchmark("vina-docking")
@pytest.mark.slow
def test_vina_docking_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark vina-docking on drug-like imatinib in the 1IEP pocket (cold + warm)."""
    inputs = _example_input()
    config = VinaDockingConfig(
        exhaustiveness=4,
        num_poses=3,
        energy_range=5.0,
        cpu=1,
        seed=7,
    )

    output = benchmark_twice(request, "vina", lambda: run_vina_docking(inputs, config))

    validate_output(output)
    assert_metrics_in_spec(output)
    assert output.tool_id == "vina-docking"
    assert output.seed == 7
    assert 1 <= len(output.poses) <= 3
    assert output.poses[0].metrics.affinity < -8.0
    assert [pose.rank for pose in output.poses] == list(range(1, len(output.poses) + 1))
