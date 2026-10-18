"""proto_tools/tools/inverse_folding/caliby/standalone/inference.py.

Caliby sequence design, Potts-energy scoring, and sidechain packing inside the isolated venv.
"""

import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger, resolve_weights_dir, serialize_output, set_torch_seed

logger = get_logger(__name__)

_OPERATIONS = ("sample", "score", "pack")

# Single stable key for the one structure (or one ensemble) each dispatch carries.
_ENSEMBLE_KEY = "input"


def _configure_caliby_env(model_checkpoint: str) -> None:
    """Resolve the managed weights directory into the environment variables Caliby reads.

    AtomWorks requires ``PDB_MIRROR_PATH`` and ``CCD_MIRROR_PATH`` to be set (empty is
    fine), and ``caliby.weights.resolve_ckpt_path`` looks up bundled checkpoints under
    ``MODEL_PARAMS_DIR``.

    Args:
        model_checkpoint (str): Bundled checkpoint name or a path to a ``.ckpt`` file.

    Raises:
        FileNotFoundError: If no weights directory resolves, or the requested checkpoint
            is not present in it.
    """
    os.environ.setdefault("PDB_MIRROR_PATH", "")
    os.environ.setdefault("CCD_MIRROR_PATH", "")

    weights_dir = resolve_weights_dir("caliby")
    if not weights_dir:
        raise FileNotFoundError(
            "caliby: unable to resolve a weights directory; set PROTO_CALIBY_WEIGHTS_DIR, "
            "or configure PROTO_MODEL_CACHE"
        )
    os.environ["MODEL_PARAMS_DIR"] = weights_dir

    if model_checkpoint.endswith(".ckpt"):
        if not Path(model_checkpoint).is_file():
            raise FileNotFoundError(f"caliby: checkpoint file not found at {model_checkpoint}")
        return

    checkpoint_path = Path(weights_dir) / "caliby" / f"{model_checkpoint}.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"caliby: checkpoint '{model_checkpoint}' not found at {checkpoint_path}; re-run setup.sh to "
            "download the bundled checkpoints, or point PROTO_CALIBY_WEIGHTS_DIR at a directory holding "
            "caliby/<name>.ckpt"
        )


def _load_atom_array(structure_path: str) -> Any:
    """Read a structure file's first model into a biotite ``AtomArray``.

    Args:
        structure_path (str): Path to a PDB file.

    Returns:
        Any: The parsed ``biotite.structure.AtomArray``.
    """
    import biotite.structure.io.pdb as pdb

    return pdb.get_structure(pdb.PDBFile.read(structure_path), model=1)


def _verify_fixed_positions(structure_path: str, fixed_positions: dict[str, list[int]] | None) -> None:
    """Require every requested fixed position to exist in the structure.

    Caliby's ``parse_fixed_pos_str`` only warns and carries on when a position is absent,
    so a residue-numbering mismatch would otherwise fix nothing at all and pass silently.
    Matching is on residue number alone, as upstream's is, so a position that shares its
    number with an insertion-coded neighbour pins that neighbour too.

    Args:
        structure_path (str): Path to the primary conformer being designed.
        fixed_positions (dict[str, list[int]] | None): Per-chain residue numbers to fix.

    Raises:
        ValueError: If any requested ``(chain, residue number)`` pair is missing.
    """
    if not fixed_positions:
        return

    atom_array = _load_atom_array(structure_path)
    present = set(zip(atom_array.chain_id.tolist(), atom_array.res_id.tolist(), strict=True))
    missing = [
        f"{chain_id}{number}"
        for chain_id, numbers in fixed_positions.items()
        for number in numbers
        if (chain_id, int(number)) not in present
    ]
    if missing:
        raise ValueError(
            f"caliby: fixed position(s) {missing} are not present in {structure_path}; fixed_positions must "
            "use the input structure's residue numbering"
        )


def _build_fixed_pos_seq(
    chain_ids: list[str],
    chains_to_redesign: list[str] | None,
    fixed_positions: dict[str, list[int]] | None,
) -> str:
    """Build Caliby's comma-separated ``fixed_pos_seq`` string from the tool's selections.

    Caliby has no chain-selection concept, so chains that are not being redesigned are
    held fixed by emitting their bare chain ID (Caliby's "whole chain" token). Individual
    fixed positions are emitted as one ``<chain><residue number>`` token each.

    Args:
        chain_ids (list[str]): All polymer chain IDs, in input structure order.
        chains_to_redesign (list[str] | None): Chains whose sequence is redesigned.
        fixed_positions (dict[str, list[int]] | None): Per-chain residue numbers to fix.

    Returns:
        str: The ``fixed_pos_seq`` value, empty when nothing is constrained.

    Raises:
        ValueError: If a constrained chain ID is not a single letter, which Caliby's
            position syntax cannot express.
    """
    redesigned = set(chains_to_redesign or [])
    fixed_chain_ids = [chain_id for chain_id in chain_ids if chain_id not in redesigned]

    constrained = set(fixed_chain_ids) | set(fixed_positions or {})
    invalid = sorted(chain_id for chain_id in constrained if len(chain_id) != 1 or not chain_id.isalpha())
    if invalid:
        raise ValueError(
            f"caliby: cannot constrain chain(s) {invalid}; Caliby's position syntax only accepts "
            "single-letter chain IDs"
        )

    tokens = list(fixed_chain_ids)
    for chain_id, numbers in (fixed_positions or {}).items():
        tokens.extend(f"{chain_id}{number}" for number in numbers)
    return ",".join(tokens)


def _build_fixed_pos_override_seq(pinned_residues: dict[str, dict[int, str]] | None) -> str:
    """Build Caliby's ``fixed_pos_override_seq`` string from supplied residue identities.

    Caliby rewrites the residue type at these positions, re-derives the atom array's
    residue names, then appends the positions to ``fixed_pos_seq``, so an override is
    conditioned on without being listed as fixed separately. Caliby also refuses to fix
    sidechains at an overridden position, since the structure's sidechain atoms belong to
    the residue that was replaced.

    Args:
        pinned_residues (dict[str, dict[int, str]] | None): Per-chain maps of residue
            number to the one-letter code to place there.

    Returns:
        str: Overrides as ``"A26:A,A27:L"``, empty when nothing is pinned.
    """
    if not pinned_residues:
        return ""
    return ",".join(
        f"{chain_id}{number}:{residue}"
        for chain_id, pins in pinned_residues.items()
        for number, residue in sorted(pins.items())
    )


def _build_symmetry_pos(chain_groups: list[list[str]] | None, structure_path: str) -> str:
    """Build Caliby's ``symmetry_pos`` string, tying each group's chains by residue number.

    Caliby ties positions by residue number, so every chain in a group must carry the same
    numbering. Resolution runs against the structure Caliby loads, and a numbering
    disagreement raises rather than silently tying a subset.

    Args:
        chain_groups (list[list[str]] | None): Chain IDs to tie, one list per group.
        structure_path (str): Path to the structure the groups are resolved against.

    Returns:
        str: Groups as ``"A15,B15|A16,B16"``, empty when nothing is tied.

    Raises:
        ValueError: If a group's chains do not share identical residue numbering.
    """
    if not chain_groups:
        return ""

    atom_array = _load_atom_array(structure_path)
    numbering: dict[str, list[int]] = {}
    for chain_id in {cid for group in chain_groups for cid in group}:
        residues = atom_array.res_id[atom_array.chain_id == chain_id]
        numbering[chain_id] = sorted(dict.fromkeys(int(r) for r in residues))

    groups: list[str] = []
    for group in chain_groups:
        reference = numbering[group[0]]
        if not reference:
            raise ValueError(f"caliby: chain {group[0]!r} has no residues to tie")
        for chain_id in group[1:]:
            if numbering[chain_id] != reference:
                raise ValueError(
                    f"caliby: chains {group[0]!r} and {chain_id!r} are tied together but carry "
                    f"different residue numbering ({len(reference)} vs {len(numbering[chain_id])} "
                    "residues). Caliby ties positions by residue number, so renumber the chains to "
                    "match before tying them."
                )
        groups.extend(",".join(f"{cid}{res_num}" for cid in group) for res_num in reference)

    return "|".join(groups)


def _split_chain_sequences(sequence: str, chain_ids: list[str]) -> list[str]:
    """Split a ``":"``-joined Caliby sequence into per-chain strings in ``chain_ids`` order.

    Caliby emits one part per chain its featurizer kept, in alphabetical chain order. The
    featurizer keeps peptide polymers of at least five residues and discards everything
    else, so that set matches the input structure's polymer chains for a plain protein
    input but omits nucleic acid and short peptide chains. A disagreement is reported
    rather than silently mis-assigning sequences to chains.

    Args:
        sequence (str): A Caliby sequence with ``':'`` between chains.
        chain_ids (list[str]): All polymer chain IDs, in input structure order.

    Returns:
        list[str]: One sequence per chain, aligned with ``chain_ids``.

    Raises:
        ValueError: If the sequence does not carry one part per input polymer chain.
    """
    parts = sequence.split(":")
    if len(parts) != len(chain_ids):
        raise ValueError(
            f"caliby: the model produced {len(parts)} chain(s) but the input structure has "
            f"{len(chain_ids)} polymer chain(s) ({chain_ids}). Caliby designs peptide chains of "
            "at least five residues and discards everything else, so nucleic acid chains and "
            "shorter peptides are dropped before the model sees them. Supply protein chains only."
        )
    by_chain_id = dict(zip(sorted(chain_ids), parts, strict=True))
    return [by_chain_id[chain_id] for chain_id in chain_ids]


def _sequence_recovery(
    designed: list[str],
    native: list[str],
    chain_ids: list[str],
    chains_to_redesign: list[str] | None,
) -> float:
    """Fraction of redesigned positions whose residue matches the input structure.

    Args:
        designed (list[str]): Designed per-chain sequences, in ``chain_ids`` order.
        native (list[str]): Input per-chain sequences, in ``chain_ids`` order.
        chain_ids (list[str]): All polymer chain IDs, in input structure order.
        chains_to_redesign (list[str] | None): Chains whose sequence is redesigned.

    Returns:
        float: Recovery over the redesigned chains, 0.0 when nothing was redesigned.

    Raises:
        ValueError: If a designed chain's length differs from the input chain's length.
    """
    redesigned = set(chains_to_redesign or [])
    matches = 0
    positions = 0
    for chain_id, design_seq, native_seq in zip(chain_ids, designed, native, strict=True):
        if chain_id not in redesigned:
            continue
        if len(design_seq) != len(native_seq):
            raise ValueError(
                f"caliby: chain {chain_id} was designed at length {len(design_seq)} but the input chain has "
                f"length {len(native_seq)}"
            )
        matches += sum(designed_aa == native_aa for designed_aa, native_aa in zip(design_seq, native_seq, strict=True))
        positions += len(design_seq)
    return matches / positions if positions else 0.0


@contextlib.contextmanager
def _staged_inputs(primary_path: str, conformer_paths: list[str]) -> Iterator[tuple[Path, list[Path]]]:
    """Symlink the dispatched structures into one directory under distinct names.

    Caliby keys every example, and every positional-constraint row, on the input file's
    stem, while the dispatched structures all arrive as ``input.pdb`` in separate temp
    directories. Staging makes the stems unique so ensemble constraints resolve per
    conformer instead of colliding.

    Args:
        primary_path (str): Path to the primary conformer.
        conformer_paths (list[str]): Paths to the other ensemble members.

    Yields:
        tuple[Path, list[Path]]: The staged primary path, then the staged conformer paths.
    """
    with tempfile.TemporaryDirectory() as staging_dir:
        staged_primary = Path(staging_dir) / f"primary{Path(primary_path).suffix}"
        staged_primary.symlink_to(Path(primary_path).resolve())

        staged_conformers: list[Path] = []
        for index, conformer_path in enumerate(conformer_paths):
            staged = Path(staging_dir) / f"conformer_{index}{Path(conformer_path).suffix}"
            staged.symlink_to(Path(conformer_path).resolve())
            staged_conformers.append(staged)

        yield staged_primary, staged_conformers


class CalibyRunner:
    """A loaded Caliby model, reloaded only when the checkpoint or device changes."""

    def __init__(self) -> None:
        """Initialize with no model loaded."""
        self.model: Any = None
        self.device: str | None = None
        self._model_checkpoint: str | None = None

    def load(self, model_checkpoint: str, device: str) -> None:
        """Load a checkpoint onto ``device``, replacing any previously loaded model.

        Args:
            model_checkpoint (str): Bundled checkpoint name or a path to a ``.ckpt`` file.
            device (str): Torch device string, e.g. ``'cuda'`` or ``'cpu'``.
        """
        _configure_caliby_env(model_checkpoint)

        from caliby import load_model

        logger.debug("Loading Caliby checkpoint %s on %s", model_checkpoint, device)
        self.model = load_model(model_checkpoint, device=device)
        self.device = device
        self._model_checkpoint = model_checkpoint

    def ensure_loaded(self, model_checkpoint: str, device: str) -> None:
        """Load on first use, reload on a checkpoint change, move on a device change.

        Args:
            model_checkpoint (str): Bundled checkpoint name or a path to a ``.ckpt`` file.
            device (str): Torch device string, e.g. ``'cuda'`` or ``'cpu'``.
        """
        if self.model is None or self._model_checkpoint != model_checkpoint:
            self.load(model_checkpoint, device)
        elif self.device != device:
            self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the loaded model to a different device.

        Args:
            device (str): Torch device string to move to.

        Raises:
            ValueError: If no model is loaded yet.
        """
        from standalone_helpers import move_model_to_device

        if self.model is None:
            raise ValueError("caliby: cannot move unloaded model to device; call load() first")
        if self.device == device:
            return

        self.model.model = move_model_to_device(self.model.model, self.device, device)
        self.model.device = device
        self.device = device

    def sample(
        self,
        *,
        primary_path: str,
        conformer_paths: list[str],
        chain_ids: list[str],
        chains_to_redesign: list[str] | None,
        fixed_positions: dict[str, list[int]] | None,
        symmetric_chains: list[list[str]] | None,
        pinned_residues: dict[str, dict[int, str]] | None,
        num_sequences: int,
        batch_size: int,
        temperature: float,
        potts_sweeps: int,
        excluded_amino_acids: list[str] | None,
        use_primary_res_type: bool,
        model_checkpoint: str,
        device: str,
        verbose: bool,
    ) -> dict[str, Any]:
        """Design sequences for one structure, optionally conditioned on a conformer ensemble.

        Args:
            primary_path (str): Path to the primary conformer.
            conformer_paths (list[str]): Paths to the other ensemble members; empty runs
                single-structure design.
            chain_ids (list[str]): All polymer chain IDs, in input structure order.
            chains_to_redesign (list[str] | None): Chains whose sequence is redesigned.
            fixed_positions (dict[str, list[int]] | None): Per-chain residue numbers to fix.
            symmetric_chains (list[list[str]] | None): Chain groups sampled to one sequence.
            pinned_residues (dict[str, dict[int, str]] | None): Per-chain residue numbers mapped
                to the residue code to place and condition on.
            num_sequences (int): Sequences to design.
            batch_size (int): Conformers per forward pass; ensemble-conditioned design only.
            temperature (float): Final Potts sampling temperature.
            potts_sweeps (int): MCMC sweeps per designed sequence.
            excluded_amino_acids (list[str] | None): One-letter codes never placed.
            use_primary_res_type (bool): Take residue identities from the primary conformer.
            model_checkpoint (str): Design checkpoint name or ``.ckpt`` path.
            device (str): Torch device string.
            verbose (bool): Let Caliby print its constraint summary.

        Returns:
            dict[str, Any]: ``chain_sequences``, ``potts_energy``, ``sequence_recovery``
                and ``cif_strings``, each of length ``num_sequences``.

        Raises:
            ValueError: If Caliby returns a different number of designs than requested.
        """
        _verify_fixed_positions(primary_path, fixed_positions)
        _verify_fixed_positions(primary_path, {cid: list(pins) for cid, pins in (pinned_residues or {}).items()})
        self.ensure_loaded(model_checkpoint, device)

        from caliby import make_constraints, make_ensemble_constraints

        fixed_pos_seq = _build_fixed_pos_seq(chain_ids, chains_to_redesign, fixed_positions)
        symmetry_pos = _build_symmetry_pos(symmetric_chains, primary_path)
        override_seq = _build_fixed_pos_override_seq(pinned_residues)
        # Only the columns that carry a value; an all-empty dict must stay None.
        constraint_cols = {
            key: value
            for key, value in (
                ("fixed_pos_seq", fixed_pos_seq),
                ("symmetry_pos", symmetry_pos),
                ("fixed_pos_override_seq", override_seq),
            )
            if value
        }
        overrides = {"potts_sampling_cfg": {"potts_sweeps": potts_sweeps}}

        with (
            _staged_inputs(primary_path, conformer_paths) as (staged_primary, staged_conformers),
            tempfile.TemporaryDirectory() as out_dir,
        ):
            if staged_conformers:
                # Element 0 is the primary conformer; the rest are aggregated position-wise.
                pdb_to_conformers = {_ENSEMBLE_KEY: [str(staged_primary), *(str(path) for path in staged_conformers)]}
                constraint_df = (
                    make_ensemble_constraints({_ENSEMBLE_KEY: constraint_cols}, pdb_to_conformers)
                    if constraint_cols
                    else None
                )
                results = self.model.ensemble_sample(
                    pdb_to_conformers,
                    out_dir=out_dir,
                    num_seqs_per_pdb=num_sequences,
                    batch_size=batch_size,
                    omit_aas=excluded_amino_acids,
                    num_workers=0,
                    temperature=temperature,
                    verbose=verbose,
                    pos_constraint_df=constraint_df,
                    use_primary_res_type=use_primary_res_type,
                    sampling_overrides=overrides,
                )
            else:
                constraint_df = make_constraints({staged_primary.stem: constraint_cols}) if constraint_cols else None
                results = self.model.sample(
                    [str(staged_primary)],
                    out_dir=out_dir,
                    num_seqs_per_pdb=num_sequences,
                    batch_size=batch_size,
                    omit_aas=excluded_amino_acids,
                    num_workers=0,
                    temperature=temperature,
                    verbose=verbose,
                    pos_constraint_df=constraint_df,
                    sampling_overrides=overrides,
                )
            cif_strings = [Path(out_pdb).read_text() for out_pdb in results["out_pdb"]]

        if len(results["seq"]) != num_sequences:
            raise ValueError(
                f"caliby: requested {num_sequences} sequence(s) but the model returned {len(results['seq'])}"
            )

        chain_sequences = [_split_chain_sequences(seq, chain_ids) for seq in results["seq"]]
        recoveries = [
            _sequence_recovery(designed, _split_chain_sequences(native, chain_ids), chain_ids, chains_to_redesign)
            for designed, native in zip(chain_sequences, results["input_seq"], strict=True)
        ]

        return {
            "chain_sequences": chain_sequences,
            "potts_energy": [float(energy) for energy in results["U"]],
            "sequence_recovery": recoveries,
            "cif_strings": cif_strings,
        }

    def score(
        self,
        *,
        primary_path: str,
        conformer_paths: list[str],
        return_local_energies: bool,
        model_checkpoint: str,
        device: str,
    ) -> dict[str, Any]:
        """Score a structure's native sequence under the Potts model.

        Args:
            primary_path (str): Path to the structure whose sequence is scored.
            conformer_paths (list[str]): Paths to the other ensemble members; empty runs
                single-structure scoring.
            return_local_energies (bool): Also return the local conditional energy matrix.
            model_checkpoint (str): Design checkpoint name or ``.ckpt`` path.
            device (str): Torch device string.

        Returns:
            dict[str, Any]: ``sequence`` and ``potts_energy``, plus
                ``local_conditional_energies`` and ``potts_tokens`` when requested.

        Raises:
            ValueError: If Caliby returns other than one score for the structure.
        """
        self.ensure_loaded(model_checkpoint, device)

        with _staged_inputs(primary_path, conformer_paths) as (staged_primary, staged_conformers):
            if staged_conformers:
                pdb_to_conformers = {_ENSEMBLE_KEY: [str(staged_primary), *(str(path) for path in staged_conformers)]}
                results = self.model.score_ensemble(pdb_to_conformers, num_workers=0)
            else:
                results = self.model.score([str(staged_primary)], num_workers=0)

        if len(results["seq"]) != 1:
            raise ValueError(f"caliby: scoring returned {len(results['seq'])} results for a single structure")

        output: dict[str, Any] = {
            "sequence": results["seq"][0],
            "potts_energy": float(results["U"][0]),
        }
        if return_local_energies:
            from caliby.data import const

            # U_i is a (seq_len, len(POTTS_TOKENS)) matrix of local conditional energies.
            output["local_conditional_energies"] = results["U_i"][0].tolist()
            output["potts_tokens"] = list(const.POTTS_TOKENS)
        return output

    def pack(
        self,
        *,
        primary_path: str,
        diffusion_steps: int,
        step_scale: float,
        model_checkpoint: str,
        device: str,
    ) -> dict[str, Any]:
        """Pack sidechains onto a backbone that already carries the sequence to pack.

        Args:
            primary_path (str): Path to the structure to pack.
            diffusion_steps (int): Sidechain diffusion denoising steps.
            step_scale (float): Step scale (eta) for sidechain diffusion.
            model_checkpoint (str): Packer checkpoint name or ``.ckpt`` path.
            device (str): Torch device string.

        Returns:
            dict[str, Any]: The packed structure as ``cif_string``.

        Raises:
            ValueError: If Caliby returns other than one packed structure.
        """
        self.ensure_loaded(model_checkpoint, device)

        overrides = {"scn_packing_cfg": {"num_steps": diffusion_steps, "step_scale": step_scale}}

        with (
            _staged_inputs(primary_path, []) as (staged_primary, _),
            tempfile.TemporaryDirectory() as out_dir,
        ):
            results = self.model.sidechain_pack(
                [str(staged_primary)],
                out_dir=out_dir,
                num_workers=0,
                sampling_overrides=overrides,
            )
            if len(results["out_pdb"]) != 1:
                raise ValueError(f"caliby: packing returned {len(results['out_pdb'])} structures for a single input")
            cif_string = Path(results["out_pdb"][0]).read_text()

        return {"cif_string": cif_string}


# ============================================================================
# Dispatch
# ============================================================================
_runner: CalibyRunner | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution.

    Args:
        input_dict (dict[str, Any]): The dispatched payload, routed on ``operation``.

    Returns:
        dict[str, Any]: The operation's result payload.

    Raises:
        ValueError: If ``operation`` is not one of ``sample``, ``score`` or ``pack``.
    """
    global _runner
    if _runner is None:
        _runner = CalibyRunner()

    set_torch_seed(input_dict.get("seed"))
    operation = input_dict["operation"]

    if operation == "sample":
        return _runner.sample(
            primary_path=input_dict["primary_path"],
            conformer_paths=input_dict["conformer_paths"],
            chain_ids=input_dict["chain_ids"],
            chains_to_redesign=input_dict["chains_to_redesign"],
            fixed_positions=input_dict["fixed_positions"],
            symmetric_chains=input_dict["symmetric_chains"],
            pinned_residues=input_dict["pinned_residues"],
            num_sequences=input_dict["num_sequences"],
            batch_size=input_dict["batch_size"],
            temperature=input_dict["temperature"],
            potts_sweeps=input_dict["potts_sweeps"],
            excluded_amino_acids=input_dict["excluded_amino_acids"],
            use_primary_res_type=input_dict["use_primary_res_type"],
            model_checkpoint=input_dict["model_checkpoint"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )
    if operation == "score":
        return _runner.score(
            primary_path=input_dict["primary_path"],
            conformer_paths=input_dict["conformer_paths"],
            return_local_energies=input_dict["return_local_energies"],
            model_checkpoint=input_dict["model_checkpoint"],
            device=input_dict["device"],
        )
    if operation == "pack":
        return _runner.pack(
            primary_path=input_dict["primary_path"],
            diffusion_steps=input_dict["diffusion_steps"],
            step_scale=input_dict["step_scale"],
            model_checkpoint=input_dict["model_checkpoint"],
            device=input_dict["device"],
        )
    raise ValueError(f"caliby: unknown operation {operation!r}; valid: {list(_OPERATIONS)}")


def to_device(device: str) -> dict[str, Any]:
    """Move the loaded model to a device (called by DeviceManager).

    Args:
        device (str): Torch device string to move to.

    Returns:
        dict[str, Any]: Move status, noting when no model is loaded yet.
    """
    global _runner
    if _runner is not None and _runner.model is not None:
        _runner.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring).

    Returns:
        dict[str, Any]: PyTorch memory statistics for the model's device.
    """
    from standalone_helpers import get_pytorch_memory_stats

    global _runner
    device = _runner.device if _runner is not None and _runner.device is not None else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("caliby: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
