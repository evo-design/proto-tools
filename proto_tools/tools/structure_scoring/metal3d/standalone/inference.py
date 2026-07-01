"""Standalone Metal3D/dEVA inference worker."""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moleculekit.molecule import Molecule  # type: ignore[import-not-found]
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors  # type: ignore[import-not-found]
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]
from standalone_helpers import get_logger
from standalone_helpers.weights import resolve_weights_dir

logger = get_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CHECKPOINT_FILES = {
    "metal3d-cat": "metal3d_cat.pth",
    "metal3d-clean": "metal3d_clean.pth",
    "metal3d-original": "metal_0.5A_v3_d0.2_16Abox.pth",
}
# Conv kernel size per checkpoint: 3 for original Metal3D, 4 for dEVA cat/clean.
CHECKPOINT_KERNEL_SIZES = {
    "metal3d-cat": 4,
    "metal3d-clean": 4,
    "metal3d-original": 3,
}
# Grid-averaging cutoff (A) per checkpoint: 0.25 for original Metal3D, 0.5 for dEVA cat/clean.
CHECKPOINT_PROBABILITY_CUTOFFS = {
    "metal3d-cat": 0.5,
    "metal3d-clean": 0.5,
    "metal3d-original": 0.25,
}
METAL_BINDING_RESNAMES = "HIS HID HIE HIP CYS CYX GLU GLH GLN ASP ASH ASN MET"


class Model(nn.Module):  # type: ignore[misc, unused-ignore]
    """Metal3D 3D-convolutional network.

    ``kernel_size`` selects the architecture variant: 3 for the original Metal3D
    weights, 4 for dEVA's retrained cat/clean checkpoints. conv5 always uses a
    large filter to aggregate features over the whole box.
    """

    def __init__(self, kernel_size: int) -> None:
        """Initialize the Metal3D convolutional layers for the given kernel size."""
        super().__init__()
        self.conv1 = nn.Conv3d(8, 32, kernel_size, padding="same")
        self.conv2 = nn.Conv3d(32, 64, kernel_size, padding="same")
        self.conv3 = nn.Conv3d(64, 80, kernel_size, padding="same")
        self.conv4 = nn.Conv3d(80, 20, kernel_size, padding="same")
        self.conv5 = nn.Conv3d(20, 20, 20, padding="same")
        self.conv6 = nn.Conv3d(20, 16, kernel_size, padding="same")
        self.conv7 = nn.Conv3d(16, 1, kernel_size, padding="same")
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a Metal3D forward pass over voxelized residue environments."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout1(x)
        x = F.relu(self.conv6(x))
        return torch.sigmoid(self.conv7(x))


class Metal3DModel:
    """Persistent Metal3D model holder."""

    def __init__(self) -> None:
        """Initialize an unloaded model holder."""
        self.model: Model | None = None
        self.device: str | None = None
        self.checkpoint: str | None = None

    def load(self, model_checkpoint: str, device: str, verbose: bool = False) -> None:
        """Load the requested checkpoint onto ``device`` if it is not already loaded."""
        if self.model is not None and self.device == device and self.checkpoint == model_checkpoint:
            return
        self.unload()
        checkpoint_path = _resolve_checkpoint_path(model_checkpoint)
        if verbose:
            logger.info("Loading Metal3D checkpoint %s from %s", model_checkpoint, checkpoint_path)
        model = Model(kernel_size=CHECKPOINT_KERNEL_SIZES[model_checkpoint]).to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        self.model = model
        self.device = device
        self.checkpoint = model_checkpoint

    def unload(self) -> None:
        """Release the loaded model and clear CUDA cache."""
        self.model = None
        self.device = None
        self.checkpoint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str) -> None:
        """Move the loaded model to a new device."""
        if self.model is None:
            self.device = device
            return
        if self.device == device:
            return
        self.model = self.model.to(device)
        self.device = device


_model = Metal3DModel()


def _resolve_checkpoint_path(model_checkpoint: str) -> Path:
    """Locate the weight file for ``model_checkpoint`` across the resolved weights directories, or raise ``FileNotFoundError``."""
    filename = CHECKPOINT_FILES[model_checkpoint]
    search_dirs: list[Path] = []
    resolved = resolve_weights_dir("metal3d")
    if resolved:
        search_dirs.append(Path(resolved))
    venv = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
    if venv:
        search_dirs.append(Path(venv) / "weights")
    for directory in search_dirs:
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"metal3d: checkpoint {filename!r} not found in {search_dirs}; run setup.sh or set PROTO_METAL3D_WEIGHTS_DIR"
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Unwrap a saved checkpoint to a plain tensor state dict.

    Handles checkpoints nested under ``state_dict``/``model_state_dict``/``model`` and
    strips any ``module.`` prefixes left by ``DataParallel``.
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"metal3d: expected checkpoint dict, got {type(checkpoint).__name__}")
    return {
        str(key).removeprefix("module."): value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)
    }


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a Metal3D worker request."""
    operation = input_dict["operation"]
    if operation != "predict":
        raise ValueError(f"metal3d: unknown operation {operation!r}; valid: ['predict']")

    device = input_dict.get("device", "cuda")
    model_checkpoint = input_dict.get("model_checkpoint", "metal3d-original")
    if model_checkpoint not in CHECKPOINT_FILES:
        raise ValueError(f"metal3d: unknown checkpoint {model_checkpoint!r}")
    verbose = bool(input_dict.get("verbose", False))

    _model.load(model_checkpoint, device, verbose=verbose)
    assert _model.model is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdb_path = Path(tmp_dir) / "input.pdb"
        pdb_path.write_text(input_dict["pdb_content"])
        return _predict(
            model=_model.model,
            pdb_path=pdb_path,
            candidate_residues=input_dict.get("candidate_residues"),
            probability_threshold=float(input_dict.get("probability_threshold", 0.2)),
            cluster_distance_threshold=float(input_dict.get("cluster_distance_threshold", 7.0)),
            probability_cutoff=CHECKPOINT_PROBABILITY_CUTOFFS[model_checkpoint],
            max_sites=int(input_dict.get("max_sites", 8)),
            device=device,
        )


def _predict(
    *,
    model: Model,
    pdb_path: Path,
    candidate_residues: dict[str, list[int]] | None,
    probability_threshold: float,
    cluster_distance_threshold: float,
    probability_cutoff: float,
    max_sites: int,
    device: str,
) -> dict[str, Any]:
    """Run Metal3D over a structure and return sites, per-residue probabilities, and an annotated PDB.

    Voxelizes each candidate residue environment, runs the model, averages per-voxel
    probabilities onto a grid, clusters high-probability points into sites, and appends the
    top site as a zinc atom when it clears ``probability_threshold``.
    """
    ids, residue_records = _candidate_ca_indices(pdb_path, candidate_residues)
    if len(ids) == 0:
        annotated_pdb = pdb_path.read_text()
        return {
            "pmetal": 0.0,
            "found": False,
            "sites": [],
            "residue_probabilities": [],
            "annotated_pdb": annotated_pdb,
        }

    record_by_id = dict(zip(ids, residue_records, strict=True))
    voxels, prot_centers, _prot_n, _prots, voxelized_ids = _process_structures(str(pdb_path), ids)

    outputs = torch.zeros([voxels.size(0), 1, 32, 32, 32], device="cpu")
    with torch.no_grad():
        for i in range(voxels.size(0)):
            outputs[i : i + 1] = model(voxels[i : i + 1].to(device)).detach().cpu()

    # Align probabilities to residues by the actually-voxelized atom id (failed residues are dropped).
    per_res_p, _ = torch.max(outputs.view(outputs.shape[0], -1), 1)
    residue_probabilities = [
        {**record_by_id[atom_id], "probability": float(probability)}
        for atom_id, probability in zip(voxelized_ids, per_res_p.numpy().tolist(), strict=True)
    ]

    prot_v = np.vstack(prot_centers)
    output_v = outputs.flatten().numpy()
    bb = _get_bb(prot_v)
    grid, _ = _create_grid_from_bb(bb)
    probability_values = _get_probability_mean(grid, prot_v, output_v, cutoff=probability_cutoff)
    raw_sites = _find_unique_sites(
        probability_values,
        grid,
        threshold=cluster_distance_threshold,
        probability_threshold=probability_threshold,
    )
    raw_sites.sort(key=lambda site: site["probability"], reverse=True)
    sites = raw_sites[:max_sites]
    pmetal = float(sites[0]["probability"]) if sites else 0.0
    annotated_pdb = _annotate_top_site(
        pdb_path.read_text(), sites[0] if sites and pmetal > probability_threshold else None
    )

    return {
        "pmetal": pmetal,
        "found": bool(sites),
        "sites": sites,
        "residue_probabilities": residue_probabilities,
        "annotated_pdb": annotated_pdb,
    }


def _candidate_ca_indices(
    pdb_path: Path,
    candidate_residues: dict[str, list[int]] | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return CA atom indices to score, with a chain/residue record for each.

    Uses ``candidate_residues`` when provided, otherwise every canonical metal-binding
    residue type in the structure.
    """
    mol = Molecule(str(pdb_path))
    mol.filter("protein and not hydrogen")
    if candidate_residues:
        ids: list[int] = []
        for chain_id, positions in candidate_residues.items():
            if not positions:
                continue
            joined = " ".join(str(p) for p in positions)
            selection = f"name CA and chain {chain_id} and resid {joined}"
            ids.extend(int(x) for x in mol.get("index", selection))
    else:
        selection = f"name CA and resname {METAL_BINDING_RESNAMES}"
        ids = [int(x) for x in mol.get("index", selection)]

    records = [_residue_record(mol, atom_id) for atom_id in ids]
    return ids, records


def _residue_record(mol: Molecule, atom_id: int) -> dict[str, Any]:
    try:
        return {
            "chain_id": _scalar_or_none(mol.get("chain", f"index {atom_id}")),
            "residue_id": _int_or_none(_scalar_or_none(mol.get("resid", f"index {atom_id}"))),
            "residue_name": _scalar_or_none(mol.get("resname", f"index {atom_id}")),
        }
    except Exception:
        return {"chain_id": None, "residue_id": None, "residue_name": None}


def _scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _voxelize_single_notcentered(env: tuple[Molecule, int]) -> tuple[torch.Tensor, np.ndarray, Any, Molecule] | None:
    """Voxelize the atomic environment around one residue's CA into the 8-channel Metal3D grid.

    Builds the eight physicochemical channels (hydrophobic, aromatic, metal-coordinating,
    H-bond acceptor/donor, positive, negative, occupancy) and returns the voxel tensor with
    its grid centers, or ``None`` if voxelization fails.
    """
    prot, atom_id = env
    center = prot.get("coords", sel=f"index {atom_id} and name CA")
    size = [16, 16, 16]
    try:
        hydrophobic = prot.atomselect("element C").reshape(-1, 1)
        aromatic = prot.atomselect(
            "resname HIS HIE HIP HID TRP TYR PHE and sidechain and not name CB and not hydrogen"
        ).reshape(-1, 1)
        metalcoordination = prot.atomselect("(name ND1 NE2 SG OE1 OE2 OD2) or (protein and name O N)").reshape(-1, 1)
        hbondacceptor = prot.atomselect(
            "(resname ASP GLU HIS HIE HIP HID SER THR MSE CYS MET and name ND2 NE2 OE1 OE2 OD1 OD2 OG OG1 SE SG) or name O"
        ).reshape(-1, 1)
        hbonddonor = prot.atomselect(
            "(resname ASN GLN ASH GLH TRP MSE SER THR MET CYS and name ND2 NE2 NE1 SG SE OG OG1) or name N"
        ).reshape(-1, 1)
        positive = prot.atomselect("resname LYS ARG HIS HIE HIP HID and name NZ NH1 NH2 ND1 NE2 NE").reshape(-1, 1)
        negative = prot.atomselect("(resname ASP GLU ASH GLH and name OD1 OD2 OE1 OE2)").reshape(-1, 1)
        occupancy = prot.atomselect("protein and not hydrogen").reshape(-1, 1)
        userchannels = np.hstack(
            [hydrophobic, aromatic, metalcoordination, hbondacceptor, hbonddonor, positive, negative, occupancy]
        )
        prot_vox, prot_centers, prot_n = getVoxelDescriptors(
            prot,
            center=center,
            userchannels=userchannels,
            boxsize=size,
            voxelsize=0.5,
            validitychecks=False,
        )
    except Exception as exc:
        logger.warning("metal3d: voxelization failed for atom index %s: %s", atom_id, exc)
        return None

    nchannels = prot_vox.shape[1]
    prot_vox_t = prot_vox.transpose().reshape([nchannels, prot_n[0], prot_n[1], prot_n[2]])
    return torch.from_numpy(prot_vox_t.copy()).float(), prot_centers, prot_n, prot.copy()


def _process_structures(
    pdb_file: str, atom_ids: list[int]
) -> tuple[torch.Tensor, tuple[np.ndarray, ...], tuple[Any, ...], tuple[Molecule, ...], list[int]]:
    """Voxelize every candidate residue and stack the results.

    Returns the batched voxel tensor, per-residue grid centers/dims/molecules, and the subset
    of ``atom_ids`` that voxelized successfully (failures are dropped).
    """
    prot = Molecule(pdb_file)
    prot.filter("protein and not hydrogen")
    results = []
    voxelized_ids: list[int] = []
    for atom_id in atom_ids:
        result = _voxelize_single_notcentered((prot.copy(), atom_id))
        if result is not None:
            results.append(result)
            voxelized_ids.append(atom_id)
    if not results:
        raise RuntimeError("metal3d: voxelization failed for every candidate residue")
    vox_env, prot_centers_list, prot_n_list, envs = zip(*results, strict=True)
    return torch.stack(vox_env, dim=0), prot_centers_list, prot_n_list, envs, voxelized_ids


def _create_grid_from_bb(
    bounding_box: list[list[float]], voxel_size: float = 1.0
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build a regular point grid spanning ``bounding_box`` at ``voxel_size`` spacing.

    Returns the flattened ``(N, 3)`` grid points and the per-axis dimensions.
    """
    xrange = np.arange(bounding_box[0][0], bounding_box[1][0] + 0.5, step=voxel_size)
    yrange = np.arange(bounding_box[0][1], bounding_box[1][1] + 0.5, step=voxel_size)
    zrange = np.arange(bounding_box[0][2], bounding_box[1][2] + 0.5, step=voxel_size)
    gridpoints = np.zeros((xrange.shape[0] * yrange.shape[0] * zrange.shape[0], 3))
    i = 0
    for x in xrange:
        for y in yrange:
            for z in zrange:
                gridpoints[i] = (x, y, z)
                i += 1
    return gridpoints, (xrange.shape[0], yrange.shape[0], zrange.shape[0])


def _get_bb(points: np.ndarray) -> list[list[float]]:
    return [
        [float(np.min(points[:, 0])), float(np.min(points[:, 1])), float(np.min(points[:, 2]))],
        [float(np.max(points[:, 0])), float(np.max(points[:, 1])), float(np.max(points[:, 2]))],
    ]


def _get_probability_mean(grid: np.ndarray, prot_centers: np.ndarray, pvalues: np.ndarray, cutoff: float) -> np.ndarray:
    """Average per-voxel probabilities onto each grid point from nearby voxel centers.

    For each grid point, averages ``pvalues`` over voxel centers within ``cutoff`` Angstroms
    (up to 20 neighbors); points with no neighbor in range get 0.
    """
    tree = KDTree(prot_centers)
    probabilities = []
    for point in grid:
        nearest_neighbors, indices = tree.query(point, k=20, distance_upper_bound=cutoff, workers=1)
        finite = nearest_neighbors != np.inf
        probabilities.append(float(np.mean(pvalues[indices[finite]])) if np.any(finite) else 0.0)
    probabilities_array: np.ndarray = np.asarray(probabilities, dtype=np.float64)
    return probabilities_array


def _find_unique_sites(
    pvalues: np.ndarray,
    grid: np.ndarray,
    *,
    threshold: float,
    probability_threshold: float,
) -> list[dict[str, float]]:
    """Cluster grid points above ``probability_threshold`` into distinct metal sites.

    Groups high-probability points with complete-linkage agglomerative clustering (merge
    distance ``threshold``) and returns each cluster's probability-weighted center and peak
    probability.
    """
    points = grid[pvalues > probability_threshold]
    point_p = pvalues[pvalues > probability_threshold]
    if len(points) == 0:
        return []
    if len(points) == 1:
        position = points[0]
        return [
            {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "probability": float(point_p[0]),
            }
        ]

    clustering = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=threshold).fit(points)
    sites = []
    for cluster_idx in range(clustering.n_clusters_):
        c_points = points[clustering.labels_ == cluster_idx]
        c_points_p = point_p[clustering.labels_ == cluster_idx]
        center = np.average(c_points, axis=0, weights=c_points_p)
        sites.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "probability": float(np.max(c_points_p)),
            }
        )
    return sites


def _annotate_top_site(pdb_content: str, site: dict[str, float] | None) -> str:
    """Append ``site`` as a zinc HETATM to the PDB text, or return it unchanged when ``site`` is None."""
    if site is None:
        return pdb_content
    lines = [line for line in pdb_content.rstrip().splitlines() if not line.startswith("END")]
    zinc_line = (
        f"HETATM99999 ZN    ZN Z   1    "
        f"{site['x']:8.3f}{site['y']:8.3f}{site['z']:8.3f}"
        f"  {site['probability']:4.2f}  0.00          ZN2+"
    )
    return "\n".join([*lines, zinc_line, "END"]) + "\n"


def to_device(device: str) -> dict[str, Any]:
    """Move the persistent model to ``device``."""
    _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Return CUDA memory statistics for the worker."""
    if not torch.cuda.is_available():
        return {"available": False, "framework": "torch", "reason": "CUDA not available"}
    return {
        "available": True,
        "framework": "torch",
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("metal3d: usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    output = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(output, f)
