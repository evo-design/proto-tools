"""PyRosetta scoring standalone runner for ToolInstance venv execution."""

import json
import sys
from typing import Any


# ============================================================================
# PyRosetta Scorer
# ============================================================================
class PyRosettaScorer:
    """Lazy-initialized PyRosetta scorer supporting SAP, SASA, and energy."""

    def __init__(self) -> None:
        """Initialize scorer with lazy PyRosetta loading."""
        self._initialized = False

    def _ensure_init(self) -> None:
        """Initialize PyRosetta on first use.

        Flags:
            ``-mute all``: Suppress PyRosetta's C-level stdout/stderr output.
            ``-ignore_unrecognized_res true``: Silently drop residues PyRosetta
                cannot parse (non-standard AAs, exotic ligands, etc.) instead
                of aborting. This is the de facto community standard for
                production scoring pipelines on heterogeneous PDB input —
                flipping to ``false`` would crash on any PDB with a ligand
                lacking a ``.params`` file (ATP, NAD, HEM, user cofactors).
            ``-remember_unrecognized_res true``: Track the names of dropped
                residues in ``PDBInfo`` so we can surface them to the user
                via a warning after scoring completes. This gives us
                "warn-but-continue" semantics without hand-rolling a PDB
                parser to diff the input against the loaded pose.
        """
        if self._initialized:
            return
        import pyrosetta

        pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true -remember_unrecognized_res true")
        self._initialized = True

    def _find_dropped_residues(self, pose: Any) -> list[str]:
        """Return three-letter codes of residues PyRosetta silently dropped.

        Reads from ``pose.pdb_info()`` — Rosetta tracks dropped residues
        natively when ``-remember_unrecognized_res true`` is set, so we do
        not need to diff the input PDB against the pose. Waters are excluded
        upstream by Rosetta's separate ``-ignore_waters`` flag (default
        ``true``) and never reach this list.

        Args:
            pose (Any): PyRosetta pose object returned by ``_pdb_content_to_pose``.

        Returns:
            list[str]: Three-letter residue codes (e.g. ``["ATP", "MG", "MG"]``)
                for each dropped residue, preserving duplicates so callers
                can aggregate counts downstream.
        """
        pdb_info = pose.pdb_info()
        num = pdb_info.get_num_unrecognized_res()
        # PDBInfo uses 1-indexed accessors.
        return [pdb_info.get_unrecognized_res_name(i) for i in range(1, num + 1)]

    def _pdb_content_to_pose(self, pdb_content: str) -> Any:
        """Load PDB content string as a Rosetta pose.

        Prefers the high-level ``pyrosetta.io.pose_from_pdbstring`` convenience
        wrapper. Falls back to the lower-level C++ binding on older builds
        that predate the wrapper — e.g. the 2023.11 build that ships on
        ``conda.rosettacommons.org/linux-aarch64``, where ``pyrosetta.io``
        does not yet expose ``pose_from_pdbstring``.
        """
        import pyrosetta
        import pyrosetta.io

        if hasattr(pyrosetta.io, "pose_from_pdbstring"):
            return pyrosetta.io.pose_from_pdbstring(pdb_content)
        pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_content)
        return pose

    def _get_residue_info(self, pose: Any, res_index: int) -> tuple[str, int, str]:
        """Extract chain ID, PDB residue number, and residue name.

        Args:
            pose: PyRosetta pose object.
            res_index (int): 1-indexed Rosetta residue index.

        Returns:
            tuple: (chain_id, pdb_residue_index, residue_name_3letter)
        """
        pdb_info = pose.pdb_info()
        chain_id = pdb_info.chain(res_index)
        pdb_resnum = pdb_info.number(res_index)
        res_name = pose.residue(res_index).name3().strip()
        return chain_id, pdb_resnum, res_name

    def _build_residue_selector(
        self,
        pose: Any,
        chain_ids: list[str] | None = None,
        residue_positions: dict[str, list[int]] | None = None,
    ) -> Any:
        """Build a PyRosetta ResidueSelector from chain/residue constraints.

        Args:
            pose: PyRosetta pose object.
            chain_ids (list[str] | None): Chain IDs to include.
            residue_positions (dict[str, list[int]] | None): Chain ID to
                PDB residue numbers (1-indexed).

        Returns:
            ResidueSelector: A selector matching the specified constraints.
        """
        from pyrosetta.rosetta.core.select.residue_selector import (
            AndResidueSelector,
            ChainSelector,
            OrResidueSelector,
            ResidueIndexSelector,
            TrueResidueSelector,
        )

        if chain_ids is None and residue_positions is None:
            return TrueResidueSelector()

        # Build chain selector
        if chain_ids is not None:
            if len(chain_ids) == 1:
                chain_selector = ChainSelector(chain_ids[0])
            else:
                chain_selector = OrResidueSelector(
                    ChainSelector(chain_ids[0]),
                    ChainSelector(chain_ids[1]),
                )
                for cid in chain_ids[2:]:
                    chain_selector = OrResidueSelector(chain_selector, ChainSelector(cid))
        else:
            chain_selector = None

        # Build residue position selector
        if residue_positions is not None:
            pdb_info = pose.pdb_info()
            rosetta_indices = []
            for cid, pdb_resnums in residue_positions.items():
                for resnum in pdb_resnums:
                    rosetta_idx = pdb_info.pdb2pose(cid, resnum)
                    if rosetta_idx > 0:
                        rosetta_indices.append(rosetta_idx)
            idx_str = ",".join(str(i) for i in rosetta_indices)
            residue_selector = ResidueIndexSelector(idx_str)
        else:
            residue_selector = None

        # Combine selectors
        if chain_selector is not None and residue_selector is not None:
            return AndResidueSelector(chain_selector, residue_selector)
        if chain_selector is not None:
            return chain_selector
        return residue_selector

    def _get_selected_indices(self, pose: Any, selector: Any) -> list[int]:
        """Get 1-indexed Rosetta residue indices matching a selector.

        Args:
            pose: PyRosetta pose object.
            selector: ResidueSelector to apply.

        Returns:
            list[int]: 1-indexed Rosetta residue indices.
        """
        selection = selector.apply(pose)
        return [i for i in range(1, len(selection) + 1) if selection[i]]

    def compute_sap(
        self,
        pdb_contents: list[str],
        chain_ids_list: list[list[str] | None] | None = None,
    ) -> dict[str, Any]:
        """Compute SAP scores with per-residue contributions.

        Uses the user's chain selection for score_sel (which residues to
        report), but always uses the full structure for sap_calculate_sel
        and sasa_sel so that burial and SASA context are correct.

        Args:
            pdb_contents (list[str]): PDB format content strings.
            chain_ids_list (list[list[str] | None] | None): Per-structure chain IDs.

        Returns:
            dict: {"results": [{"sap_score": float, "per_residue": [...]}, ...]}
        """
        self._ensure_init()
        from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import (
            PerResidueSapScoreMetric,
            calculate_sap,
        )
        from pyrosetta.rosetta.core.select.residue_selector import (
            TrueResidueSelector,
        )

        all_res = TrueResidueSelector()

        results = []
        for i, pdb_content in enumerate(pdb_contents):
            pose = self._pdb_content_to_pose(pdb_content)
            dropped_residues = self._find_dropped_residues(pose)
            chain_ids = chain_ids_list[i] if chain_ids_list else None

            # score_sel = user's chain selection; context selectors = full structure
            score_sel = self._build_residue_selector(pose, chain_ids)
            sap_score = float(calculate_sap(pose, score_sel, all_res, all_res))

            # Per-residue SAP contributions
            metric = PerResidueSapScoreMetric()
            metric.set_score_selector(score_sel)
            metric.set_sap_calculate_selector(all_res)
            metric.set_sasa_selector(all_res)
            per_res_map = metric.calculate(pose)

            selected = self._get_selected_indices(pose, score_sel)
            per_residue = []
            for idx in selected:
                chain_id, pdb_resnum, res_name = self._get_residue_info(pose, idx)
                sap_val = float(per_res_map[idx]) if idx in per_res_map else 0.0
                per_residue.append(
                    {
                        "chain_id": chain_id,
                        "residue_index": pdb_resnum,
                        "residue_name": res_name,
                        "sap_score": sap_val,
                    }
                )

            results.append(
                {
                    "sap_score": sap_score,
                    "per_residue": per_residue,
                    "dropped_residues": dropped_residues,
                }
            )

        return {"results": results}

    def compute_sasa(
        self,
        pdb_contents: list[str],
        probe_radius: float = 1.4,
        chain_ids_list: list[list[str] | None] | None = None,
    ) -> dict[str, Any]:
        """Compute SASA for a list of PDB content strings.

        SASA is always calculated on the full pose: PyRosetta's ``SasaCalc``
        scores every residue in context, including the effect of buried
        surfaces between chains. Chain selection only filters which residues
        get reported and summed; it does not change the geometry PyRosetta sees.

        When ``chain_ids_list`` is active, the reported ``total_sasa`` is the
        sum over the **selected residues only**, not the whole-pose SASA. This
        differs from ``compute_energy``, whose ``total_energy`` is always the
        whole-pose total regardless of chain selection. The asymmetry is
        intentional: SASA is a per-residue additive quantity, so a chain-subset
        sum is a meaningful number, whereas energy has pair terms that make a
        chain-subset sum physically incoherent. Per-residue output is filtered
        by chain selection in both tools.

        Args:
            pdb_contents (list[str]): PDB format content strings.
            probe_radius (float): Solvent probe radius in Angstroms.
            chain_ids_list (list[list[str] | None] | None): Per-structure chain IDs.

        Returns:
            dict: {"results": [{"total_sasa": float, "per_residue": [...]}, ...]}
        """
        self._ensure_init()
        from pyrosetta.rosetta.core.scoring.sasa import SasaCalc

        results = []
        for i, pdb_content in enumerate(pdb_contents):
            pose = self._pdb_content_to_pose(pdb_content)
            dropped_residues = self._find_dropped_residues(pose)
            chain_ids = chain_ids_list[i] if chain_ids_list else None

            calc = SasaCalc()
            calc.set_probe_radius(probe_radius)
            calc.calculate(pose)

            residue_sasa = calc.get_residue_sasa()

            # Build selector and get selected indices
            selector = self._build_residue_selector(pose, chain_ids)
            selected = self._get_selected_indices(pose, selector)

            per_residue = []
            total_sasa = 0.0
            for idx in selected:
                chain_id, pdb_resnum, res_name = self._get_residue_info(pose, idx)
                sasa_val = float(residue_sasa[idx])
                total_sasa += sasa_val
                per_residue.append(
                    {
                        "chain_id": chain_id,
                        "residue_index": pdb_resnum,
                        "residue_name": res_name,
                        "sasa": sasa_val,
                    }
                )

            results.append(
                {
                    "total_sasa": total_sasa,
                    "per_residue": per_residue,
                    "dropped_residues": dropped_residues,
                }
            )

        return {"results": results}

    def compute_energy(
        self,
        pdb_contents: list[str],
        scorefxn_name: str = "ref2015",
        relax: bool = True,
        relax_cycles: int = 5,
        constrain_to_start: bool = True,
        chain_ids_list: list[list[str] | None] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Compute energy scores for a list of PDB content strings.

        Chain selection (``chain_ids_list``) filters which residues appear in the
        per-residue breakdown, but does not change how energies are computed.
        The full pose is always scored, so per-residue energies for selected
        chains reflect each residue's contribution within the full complex
        (including pair interactions with the un-selected chains), not the
        energy of the chain scored in isolation. The reported ``total_energy``
        is always the whole-pose total and is independent of chain selection.
        To score a chain as if it were isolated, extract it into its own
        structure first and score it separately.

        Args:
            pdb_contents (list[str]): PDB format content strings.
            scorefxn_name (str): Rosetta score function name.
            relax (bool): Whether to run FastRelax before scoring.
            relax_cycles (int): Number of FastRelax cycles.
            constrain_to_start (bool): Constrain relaxation to starting coords.
            chain_ids_list (list[list[str] | None] | None): Per-structure chain IDs.
            seed (int | None): Random seed for FastRelax reproducibility.

        Returns:
            dict: {"results": [{"total_energy": float, "energy_terms": {...}, ...}, ...]}
        """
        self._ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.core.scoring.methods import (
            EnergyMethodOptions,
        )

        # Seed PyRosetta's internal C++ RNG for reproducible FastRelax.
        if seed is not None:
            pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

        sfxn = pyrosetta.create_score_function(scorefxn_name)

        # Decompose backbone H-bonds into per-residue pair energies so
        # that per-residue totals sum to the whole-pose total.
        emopts = EnergyMethodOptions(sfxn.energy_method_options())
        emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
        sfxn.set_energy_method_options(emopts)

        results = []
        for i, pdb_content in enumerate(pdb_contents):
            pose = self._pdb_content_to_pose(pdb_content)
            dropped_residues = self._find_dropped_residues(pose)
            chain_ids = chain_ids_list[i] if chain_ids_list else None

            if relax:
                from pyrosetta.rosetta.core.scoring import ScoreType
                from pyrosetta.rosetta.protocols.relax import FastRelax

                relax_sfxn = sfxn.clone()
                if constrain_to_start:
                    relax_sfxn.set_weight(ScoreType.coordinate_constraint, 1.0)

                fast_relax = FastRelax(relax_sfxn, relax_cycles)
                fast_relax.constrain_relax_to_start_coords(constrain_to_start)
                fast_relax.apply(pose)

            # Full-structure energy (physics requires all residues)
            total_energy = float(sfxn(pose))

            # Extract per-term energy breakdown
            energies = pose.energies()
            energy_terms = {}
            for st in sfxn.get_nonzero_weighted_scoretypes():
                term_name = str(st).split(".")[-1]
                energy_terms[term_name] = float(energies.total_energies()[st])

            # Per-residue breakdown (filtered by chain selection)
            selector = self._build_residue_selector(pose, chain_ids)
            selected = self._get_selected_indices(pose, selector)

            per_residue = []
            for idx in selected:
                chain_id, pdb_resnum, res_name = self._get_residue_info(pose, idx)
                per_residue.append(
                    {
                        "chain_id": chain_id,
                        "residue_index": pdb_resnum,
                        "residue_name": res_name,
                        "total_energy": float(energies.residue_total_energy(idx)),
                    }
                )

            results.append(
                {
                    "total_energy": total_energy,
                    "energy_terms": energy_terms,
                    "per_residue": per_residue,
                    "relaxed": relax,
                    "dropped_residues": dropped_residues,
                }
            )

        return {"results": results}


# ============================================================================
# Dispatch Entry Point
# ============================================================================
_scorer = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution.

    Args:
        input_dict (dict): Input with "operation" key routing to sap/sasa/energy.

    Returns:
        dict: Operation-specific results.
    """
    global _scorer
    if _scorer is None:
        _scorer = PyRosettaScorer()

    operation = input_dict["operation"]
    pdb_contents = input_dict["pdb_contents"]
    chain_ids_list = input_dict.get("chain_ids_list")

    if operation == "sap":
        return _scorer.compute_sap(
            pdb_contents,
            chain_ids_list=chain_ids_list,
        )
    if operation == "sasa":
        return _scorer.compute_sasa(
            pdb_contents,
            probe_radius=input_dict.get("probe_radius", 1.4),
            chain_ids_list=chain_ids_list,
        )
    if operation == "energy":
        return _scorer.compute_energy(
            pdb_contents,
            scorefxn_name=input_dict.get("scorefxn", "ref2015"),
            relax=input_dict.get("relax", True),
            relax_cycles=input_dict.get("relax_cycles", 5),
            constrain_to_start=input_dict.get("constrain_to_start", True),
            chain_ids_list=chain_ids_list,
            seed=input_dict.get("seed"),
        )
    raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Device Management Protocol
# ============================================================================
def to_device(device: str) -> dict[str, Any]:
    """Passthrough — PyRosetta scoring is CPU-only."""
    return {"success": True, "device": device, "note": "CPU-only tool"}


def get_memory_stats() -> dict[str, Any]:
    """CPU-only tool — no GPU memory stats."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


# ============================================================================
# One-shot Entry Point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
