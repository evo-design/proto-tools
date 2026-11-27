"""proto_tools/tools/molecular_docking/vina/standalone/run.py.

AutoDock Vina standalone runner for rigid-receptor docking.
"""

import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

_SIGNED_INT32_MAX = 2**31 - 1
_MAX_SEARCH_BOX_AXIS = 100.0
_MIN_GRID_SPACING = 0.1
_MAX_GRID_POINTS = 2_000_000
_INITIAL_OPTIMIZATION_ITERATIONS = 200
_RETRY_OPTIMIZATION_ITERATIONS = 500

_VINA_RESULT_RE = re.compile(
    r"^REMARK VINA RESULT:\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
    re.MULTILINE,
)


def _update_status(message: str) -> None:
    """Update the managed-worker spinner, falling back to a debug log in direct CLI use."""
    update_status = getattr(logger, "update_status", None)
    if update_status is None:
        logger.debug(message)
    else:
        update_status(message)


def _prepare_receptor(
    receptor_pdb: str,
    allow_bad_residues: bool,
) -> tuple[str, dict[str, Any]]:
    """Parameterize a rigid receptor and return PDBQT plus preparation provenance."""
    from meeko import (  # type: ignore[import-not-found]
        MoleculePreparation,
        PDBQTWriterLegacy,
        Polymer,
        ResidueChemTemplates,
    )

    polymer = Polymer.from_pdb_string(
        receptor_pdb,
        ResidueChemTemplates.create_from_defaults(),
        MoleculePreparation(),
        allow_bad_res=allow_bad_residues,
    )
    rigid_pdbqt: str
    flexible_pdbqt: str
    rigid_pdbqt, flexible_pdbqt = PDBQTWriterLegacy.write_from_polymer(polymer)
    if flexible_pdbqt:
        raise RuntimeError("vina: rigid-receptor preparation unexpectedly produced flexible residues")
    if not rigid_pdbqt.strip():
        raise ValueError("vina: receptor preparation produced an empty PDBQT model")
    ignored_residue_ids = sorted(polymer.get_ignored_monomers())
    template_assignments = [
        {
            "residue_id": residue_id,
            "input_residue_name": monomer.input_resname,
            "assigned_template": monomer.residue_template_key,
        }
        for residue_id, monomer in sorted(polymer.monomers.items())
        if monomer.residue_template_key is not None and monomer.input_resname != monomer.residue_template_key
    ]
    return rigid_pdbqt, {
        "ignored_residue_ids": ignored_residue_ids,
        "template_assignments": template_assignments,
    }


def _prepare_ligand(ligand_smiles: str, seed: int) -> tuple[str, dict[str, Any]]:
    """Generate, minimize, and parameterize one seeded ligand conformer."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(ligand_smiles)
    if molecule is None:
        raise ValueError("vina: RDKit could not parse ligand_smiles")
    molecule = Chem.AddHs(molecule)

    embed_parameters = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    embed_parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, embed_parameters) != 0:  # type: ignore[attr-defined]
        raise ValueError("vina: RDKit failed to generate a ligand conformer")

    optimization_method: str | None = None
    optimization_status: int | None = None
    iteration_limits: list[int] = []
    if AllChem.MMFFHasAllMoleculeParams(molecule):  # type: ignore[attr-defined]
        optimization_method = "MMFF94"
        optimizer = AllChem.MMFFOptimizeMolecule  # type: ignore[attr-defined]
    elif AllChem.UFFHasAllMoleculeParams(molecule):  # type: ignore[attr-defined]
        optimization_method = "UFF"
        optimizer = AllChem.UFFOptimizeMolecule  # type: ignore[attr-defined]
    else:
        optimizer = None

    if optimizer is not None:
        iteration_limits.append(_INITIAL_OPTIMIZATION_ITERATIONS)
        optimization_status = int(optimizer(molecule, maxIters=_INITIAL_OPTIMIZATION_ITERATIONS))
        if optimization_status == 1:
            iteration_limits.append(_RETRY_OPTIMIZATION_ITERATIONS)
            optimization_status = int(optimizer(molecule, maxIters=_RETRY_OPTIMIZATION_ITERATIONS))
        if optimization_status < 0:
            raise ValueError(f"vina: RDKit {optimization_method} force-field setup failed")

    setups = MoleculePreparation().prepare(molecule)
    if len(setups) != 1:
        raise ValueError(f"vina: ligand preparation produced {len(setups)} variants; expected exactly one")
    pdbqt: str
    pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0], add_index_map=True)
    if not success:
        raise ValueError(f"vina: Meeko could not write ligand PDBQT: {error}")
    return pdbqt, {
        "optimization_method": optimization_method,
        "optimization_converged": optimization_status == 0 if optimization_status is not None else None,
        "optimization_status": optimization_status,
        "optimization_attempts": len(iteration_limits),
        "optimization_iteration_limits": iteration_limits,
    }


def _validated_grid_point_count(center: list[float], size: list[float], spacing: float) -> int:
    """Validate native map inputs and return the number of allocated grid points."""
    if len(center) != 3 or len(size) != 3:
        raise ValueError("vina: search-box center and size must each contain three values")
    if not all(math.isfinite(value) for value in center):
        raise ValueError("vina: search-box center coordinates must be finite")
    if not all(math.isfinite(value) and 0 < value <= _MAX_SEARCH_BOX_AXIS for value in size):
        raise ValueError(
            f"vina: search-box dimensions must be finite, positive, and at most "
            f"{_MAX_SEARCH_BOX_AXIS:g} angstroms per axis"
        )
    if not math.isfinite(spacing) or spacing < _MIN_GRID_SPACING:
        raise ValueError(f"vina: grid_spacing must be finite and at least {_MIN_GRID_SPACING:g} angstroms")

    dimensions = tuple(math.ceil(axis_size / spacing) + 1 for axis_size in size)
    grid_points = math.prod(dimensions)
    if grid_points > _MAX_GRID_POINTS:
        raise ValueError(
            f"vina: search box and grid_spacing require {grid_points:,} grid points "
            f"({dimensions[0]} x {dimensions[1]} x {dimensions[2]}); "
            f"the maximum is {_MAX_GRID_POINTS:,}"
        )
    return grid_points


def _native_int(config: dict[str, Any], key: str, minimum: int) -> int:
    """Coerce one SWIG-bound integer while enforcing signed 32-bit bounds."""
    value = int(config[key])
    if not minimum <= value <= _SIGNED_INT32_MAX:
        raise ValueError(f"vina: {key} must be between {minimum} and {_SIGNED_INT32_MAX}")
    return value


def _split_pdbqt_models(pdbqt: str) -> list[str]:
    """Split a Vina multi-model PDBQT payload into complete MODEL blocks."""
    models: list[str] = []
    current: list[str] | None = None
    for line in pdbqt.splitlines(keepends=True):
        if line.startswith("MODEL"):
            if current:
                models.append("".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
            if line.startswith("ENDMDL"):
                models.append("".join(current))
                current = None
    if current:
        models.append("".join(current))
    if not models:
        raise ValueError("vina: docking produced no PDBQT MODEL blocks")
    return models


def _split_sdf_records(sdf: str) -> list[str]:
    """Split a combined SDF payload while retaining each record delimiter."""
    records = [record.lstrip("\r\n").rstrip("\r\n") + "\n$$$$\n" for record in sdf.split("$$$$") if record.strip()]
    if not records:
        raise ValueError("vina: pose conversion produced no SDF records")
    return records


def _pose_metrics(pdbqt_model: str) -> tuple[float, float, float]:
    """Parse affinity and RMSD bounds from one Vina result remark."""
    match = _VINA_RESULT_RE.search(pdbqt_model)
    if match is None:
        raise ValueError("vina: pose is missing a REMARK VINA RESULT line")
    return tuple(float(value) for value in match.groups())  # type: ignore[return-value]


def _convert_poses_to_sdf(poses_pdbqt: str) -> str:
    """Reconstruct bond-correct SDF poses from Meeko metadata in PDBQT."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pdbqt_molecule = PDBQTMolecule(
        poses_pdbqt,
        name="ligand",
        skip_typing=True,
    )
    poses_sdf: str
    poses_sdf, failures = RDKitMolCreate.write_sd_string(pdbqt_molecule)
    if failures:
        raise ValueError(f"vina: Meeko could not convert pose indices {failures} to SDF")
    if not poses_sdf.strip():
        raise ValueError("vina: Meeko returned an empty SDF payload")
    return poses_sdf


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Prepare inputs, run Vina, and return ranked poses."""
    import meeko
    import vina  # type: ignore[import-not-found]
    from rdkit import rdBase
    from vina import Vina

    config = input_dict["config"]
    search_box = input_dict["search_box"]
    center = [float(value) for value in search_box["center"]]
    size = [float(value) for value in search_box["size"]]
    grid_spacing = float(config["grid_spacing"])
    grid_point_count = _validated_grid_point_count(center, size, grid_spacing)
    seed = _native_int(config, "seed", 1)
    exhaustiveness = _native_int(config, "exhaustiveness", 1)
    num_poses = _native_int(config, "num_poses", 1)
    max_evaluations = _native_int(config, "max_evaluations", 0)
    cpu = _native_int(config, "cpu", 0)
    energy_range = float(config["energy_range"])
    min_rmsd = float(config["min_rmsd"])
    if not math.isfinite(energy_range) or energy_range <= 0:
        raise ValueError("vina: energy_range must be finite and greater than zero")
    if not math.isfinite(min_rmsd) or min_rmsd <= 0:
        raise ValueError("vina: min_rmsd must be finite and greater than zero")

    ligand_smiles_list = input_dict["ligand_smiles"]
    if not ligand_smiles_list:
        raise ValueError("vina: ligand_smiles must contain at least one SMILES")

    # Meeko receptor parameterization is the expensive setup step and depends only on the
    # receptor, so it is done once and reused for every ligand in the request.
    _update_status("Preparing receptor")
    receptor_pdbqt, receptor_preparation = _prepare_receptor(
        input_dict["receptor_pdb"],
        bool(config["allow_bad_residues"]),
    )
    warnings: list[str] = []
    ignored_residue_ids = receptor_preparation["ignored_residue_ids"]
    if ignored_residue_ids:
        warning = "Receptor preparation omitted residues that Meeko could not parameterize: " + ", ".join(
            ignored_residue_ids
        )
        logger.warning(warning)
        warnings.append(warning)

    results = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        receptor_path = Path(temporary_directory) / "receptor.pdbqt"
        receptor_path.write_text(receptor_pdbqt)

        for ligand_index, ligand_smiles in enumerate(ligand_smiles_list):
            # Advance the seed per ligand so duplicate ligands in one request still sample
            # independently, while staying reproducible for a given (seed, position).
            ligand_seed = (seed + ligand_index - 1) % _SIGNED_INT32_MAX + 1
            _update_status(f"Preparing ligand {ligand_index + 1}/{len(ligand_smiles_list)}")
            ligand_pdbqt, ligand_preparation = _prepare_ligand(ligand_smiles, ligand_seed)
            ligand_warnings: list[str] = []
            if ligand_preparation["optimization_method"] is None:
                warning = (
                    "Ligand conformer was not force-field minimized because RDKit found no "
                    "complete MMFF94 or UFF parameters"
                )
                logger.warning(warning)
                ligand_warnings.append(warning)
            elif not ligand_preparation["optimization_converged"]:
                warning = (
                    f"Ligand conformer did not converge after {ligand_preparation['optimization_attempts']} "
                    f"{ligand_preparation['optimization_method']} optimization attempts"
                )
                logger.warning(warning)
                ligand_warnings.append(warning)

            _update_status(f"Computing affinity maps {ligand_index + 1}/{len(ligand_smiles_list)}")
            docking = Vina(
                sf_name=config["scoring_function"],
                cpu=cpu,
                seed=ligand_seed,
                verbosity=min(int(config["verbose"]), 2),
            )
            docking.set_receptor(str(receptor_path))
            docking.set_ligand_from_string(ligand_pdbqt)
            docking.compute_vina_maps(
                center=center,
                box_size=size,
                spacing=grid_spacing,
            )

            _update_status(f"Searching docking poses {ligand_index + 1}/{len(ligand_smiles_list)}")
            docking.dock(
                exhaustiveness=exhaustiveness,
                n_poses=num_poses,
                min_rmsd=min_rmsd,
                max_evals=max_evaluations,
            )
            poses_pdbqt = docking.poses(
                n_poses=num_poses,
                energy_range=energy_range,
            )

            _update_status(f"Converting docking poses {ligand_index + 1}/{len(ligand_smiles_list)}")
            poses_sdf = _convert_poses_to_sdf(poses_pdbqt)
            pdbqt_models = _split_pdbqt_models(poses_pdbqt)
            sdf_records = _split_sdf_records(poses_sdf)
            if len(pdbqt_models) != len(sdf_records):
                raise ValueError(
                    f"vina: pose-count mismatch after conversion for ligand {ligand_index}: "
                    f"{len(pdbqt_models)} PDBQT models, {len(sdf_records)} SDF records"
                )

            poses = []
            for rank, (pdbqt_model, sdf_record) in enumerate(zip(pdbqt_models, sdf_records, strict=True), start=1):
                affinity, rmsd_lower_bound, rmsd_upper_bound = _pose_metrics(pdbqt_model)
                poses.append(
                    {
                        "rank": rank,
                        "affinity": affinity,
                        "rmsd_lower_bound": rmsd_lower_bound,
                        "rmsd_upper_bound": rmsd_upper_bound,
                        "sdf": sdf_record,
                        "pdbqt": pdbqt_model,
                    }
                )

            results.append(
                {
                    "smiles": ligand_smiles,
                    "seed": int(docking.info()["seed"]),
                    "poses": poses,
                    "poses_sdf": poses_sdf,
                    "poses_pdbqt": poses_pdbqt,
                    "ligand_preparation": ligand_preparation,
                    "warnings": ligand_warnings,
                }
            )

    return {
        "results": results,
        "seed": seed,
        "vina_version": getattr(vina, "__version__", "unknown"),
        "meeko_version": getattr(meeko, "__version__", "unknown"),
        "rdkit_version": rdBase.rdkitVersion,
        "grid_point_count": grid_point_count,
        "receptor_preparation": receptor_preparation,
        "warnings": warnings,
    }


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CPU-only Vina docking."""
    return {"success": True, "device": device, "note": "CPU-only tool"}


def get_memory_stats() -> dict[str, Any]:
    """Return the CPU-only DeviceManager memory response."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"vina: usage: {sys.argv[0]} <input_json_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as input_handle:
        inputs = json.load(input_handle)

    outputs = dispatch(inputs)

    with open(sys.argv[2], "w") as output_handle:
        json.dump(outputs, output_handle)
