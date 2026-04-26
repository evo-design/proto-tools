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
        chain_ids_list: list[list[str] | None] | None = None,
    ) -> dict[str, Any]:
        """Compute energy scores for a list of PDB content strings.

        Energy scoring is deterministic. No FastRelax happens here. To score a
        relaxed pose, either run ``pyrosetta-relax`` explicitly first, or enable
        the optional ``pre_relax_structures`` preprocessing step (which runs
        ``pyrosetta-relax`` on each input before scoring).

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
            chain_ids_list (list[list[str] | None] | None): Per-structure chain IDs.

        Returns:
            dict: {"results": [{"total_energy": float, "energy_terms": {...}, ...}, ...]}
        """
        self._ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.core.scoring.methods import (
            EnergyMethodOptions,
        )

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
                    "dropped_residues": dropped_residues,
                }
            )

        return {"results": results}

    def compute_relax(
        self,
        pdb_contents: list[str],
        scorefxn_name: str = "ref2015",
        relax_cycles: int = 1,
        constrain_to_start: bool = True,
        seed: int | None = None,
        max_iter: int | None = None,
    ) -> dict[str, Any]:
        """Run FastRelax on each input pose and return the relaxed PDB + total score.

        Args:
            pdb_contents (list[str]): PDB-format content strings.
            scorefxn_name (str): Rosetta score function name.
            relax_cycles (int): Number of FastRelax repeats.
            constrain_to_start (bool): If True, add a coordinate-constraint
                term to the relax score function and call
                ``constrain_relax_to_start_coords(True)`` on the FastRelax
                mover so atoms stay near their input positions.
            seed (int | None): Seed for PyRosetta's C++ RNG, applied via
                ``rg().set_seed(seed)`` before FastRelax runs. When ``None``,
                the RNG is not reseeded; because PyRosetta's RNG is
                process-global state, output becomes dependent on prior
                calls within the same persistent worker.
            max_iter (int | None): Maximum minimizer iterations per relax
                cycle. When ``None``, PyRosetta's default (2500) is used.

        Returns:
            dict: ``{"results": [{"relaxed_pdb": str, "total_score": float,
                "relax_cycles": int, "dropped_residues": [...]}]}`` — one entry
                per input, in input order.
        """
        self._ensure_init()
        import os
        import tempfile

        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreType
        from pyrosetta.rosetta.protocols.relax import FastRelax

        if seed is not None:
            pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

        sfxn = pyrosetta.create_score_function(scorefxn_name)
        relax_sfxn = sfxn.clone()
        if constrain_to_start:
            relax_sfxn.set_weight(ScoreType.coordinate_constraint, 1.0)

        results = []
        for pdb_content in pdb_contents:
            pose = self._pdb_content_to_pose(pdb_content)
            dropped_residues = self._find_dropped_residues(pose)

            fast_relax = FastRelax(relax_sfxn, relax_cycles)
            if max_iter is not None:
                fast_relax.max_iter(max_iter)
            fast_relax.constrain_relax_to_start_coords(constrain_to_start)
            fast_relax.apply(pose)

            # PyRosetta's dump_pdb is file-based; round-trip through tempfile
            # so we can return the relaxed coordinates as a string.
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pose.dump_pdb(tmp_path)
                with open(tmp_path) as f:
                    relaxed_pdb = f.read()
            finally:
                os.unlink(tmp_path)

            results.append(
                {
                    "relaxed_pdb": relaxed_pdb,
                    "total_score": float(sfxn(pose)),
                    "relax_cycles": relax_cycles,
                    "dropped_residues": dropped_residues,
                }
            )
        return {"results": results}

    def _hotspot_residues_from_pose(
        self,
        pose: Any,
        binder_chain: str,
        target_chain: str,
        cutoff: float = 4.0,
    ) -> dict[int, str]:
        """Identify binder interface residues by all-atom proximity to the target.

        Returns ``{pdb_resnum: one_letter_aa}`` for every binder-chain residue
        that has at least one atom within ``cutoff`` angstroms of any
        target-chain atom. Uses ``scipy.spatial.cKDTree`` for the neighbor
        search and skips residues whose three-letter name is outside the
        standard 20 amino acids.

        Args:
            pose (Any): PyRosetta pose object.
            binder_chain (str): Binder chain label (single character).
            target_chain (str): Target chain label (single character).
            cutoff (float): Atom-atom distance cutoff in angstroms.

        Returns:
            dict[int, str]: Mapping from binder PDB residue number to the
                residue's one-letter amino-acid code.
        """
        import numpy as np
        from scipy.spatial import cKDTree

        # PDBInfo translates Rosetta's 1-indexed residues back to original
        # PDB chain letters and residue numbers.
        pdb_info = pose.pdb_info()

        # Binder atoms carry residue metadata for later AA counting; target
        # atoms only need coordinates (they are pure KDTree lookup targets).
        binder_atoms: list[tuple[int, str, float, float, float]] = []
        target_coords: list[list[float]] = []

        # Single pass over the pose, partitioning atoms by chain.
        for res_idx in range(1, pose.total_residue() + 1):
            chain = pdb_info.chain(res_idx)
            if chain == binder_chain:
                resnum = pdb_info.number(res_idx)
                resname = pose.residue(res_idx).name3().strip()
                for atom_idx in range(1, pose.residue(res_idx).natoms() + 1):
                    xyz = pose.residue(res_idx).atom(atom_idx).xyz()
                    binder_atoms.append((resnum, resname, xyz.x, xyz.y, xyz.z))
            elif chain == target_chain:
                for atom_idx in range(1, pose.residue(res_idx).natoms() + 1):
                    xyz = pose.residue(res_idx).atom(atom_idx).xyz()
                    target_coords.append([xyz.x, xyz.y, xyz.z])

        # Nothing to score if either side is absent.
        if not binder_atoms or not target_coords:
            return {}

        # Build KDTrees for both chains; `query_ball_tree` returns, for each
        # binder atom, the list of target-atom indices within `cutoff` angstroms.
        binder_coords = np.array([(a[2], a[3], a[4]) for a in binder_atoms])
        target_arr = np.array(target_coords)
        tgt_tree = cKDTree(target_arr)
        bnd_tree = cKDTree(binder_coords)
        pairs = bnd_tree.query_ball_tree(tgt_tree, cutoff)

        three_to_one = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }
        result: dict[int, str] = {}
        for bnd_idx, close_indices in enumerate(pairs):
            if close_indices:
                resnum, resname, *_ = binder_atoms[bnd_idx]
                if resname in three_to_one:
                    result[resnum] = three_to_one[resname]
        return result

    def _surface_hydrophobicity(self, pose: Any, binder_chain: str) -> float:
        """Fraction of binder-chain surface residues that are apolar or aromatic.

        Splits the binder chain into its own sub-pose, marks surface residues
        with ``LayerSelector(pick_surface=True)``, and counts each surface
        residue that is either ``residue.is_apolar()`` or named ``PHE`` /
        ``TRP`` / ``TYR``. The fraction is ``apolar_count / total_surface``.

        Args:
            pose (Any): PyRosetta pose object.
            binder_chain (str): Binder chain label (single character).

        Returns:
            float: Surface hydrophobicity fraction in [0, 1], or 0.0 if the
                binder chain has no surface residues.
        """
        from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

        pdb_info = pose.pdb_info()
        binder_chain_num = None
        for res_idx in range(1, pose.total_residue() + 1):
            if pdb_info.chain(res_idx) == binder_chain:
                binder_chain_num = pose.chain(res_idx)
                break
        if binder_chain_num is None:
            return 0.0
        sub_poses = pose.split_by_chain()
        binder_pose = sub_poses[binder_chain_num]
        layer_sel = LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_vec = layer_sel.apply(binder_pose)
        exp_apol_count = 0
        total_count = 0
        for i in range(1, len(surface_vec) + 1):
            if surface_vec[i]:
                res = binder_pose.residue(i)
                if res.is_apolar() or res.name3().strip() in ("PHE", "TRP", "TYR"):
                    exp_apol_count += 1
                total_count += 1
        return (exp_apol_count / total_count) if total_count else 0.0

    def compute_interface_analyzer(
        self,
        pdb_contents: list[str],
        binder_chains: list[str],
        target_chains: list[str],
        scorefxn_name: str = "ref2015",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Compute interface-analysis metrics for a list of two-chain complexes.

        Runs InterfaceAnalyzerMover for shape complementarity, H-bonds, ΔG,
        dSASA, and packstat; computes interface_hydrophobicity from
        hotspot-residue AA composition (4.0 Å atom-atom cutoff, apolar set
        'ACFILMPVWY'); computes surface_hydrophobicity from
        LayerSelector(pick_surface=True) on the binder sub-pose.

        Args:
            pdb_contents (list[str]): PDB-format content strings.
            binder_chains (list[str]): Per-input binder chain labels
                (PDB-shortened, single character).
            target_chains (list[str]): Per-input target chain labels
                (PDB-shortened, single character).
            scorefxn_name (str): Rosetta score function name.
            seed (int | None): Seed for PyRosetta's C++ RNG, applied via
                ``rg().set_seed(seed)`` before ``InterfaceAnalyzerMover.apply``.
                Required for reproducible ``interface_dG`` / ``interface_packstat``
                because ``set_pack_separated(True)`` drives a stochastic
                simulated-annealing repack off the global RNG. When ``None``,
                the RNG is not reseeded; output becomes dependent on prior
                calls within the same persistent worker.

        Returns:
            dict: ``{"results": [{"interface_sc": float, "interface_hbonds": int,
                "interface_dG": float, "interface_dSASA": float,
                "interface_packstat": float, "interface_hydrophobicity": float,
                "surface_hydrophobicity": float,
                "dropped_residues": [...]}, ...]}``
        """
        self._ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        if seed is not None:
            pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

        results = []
        for i, pdb_content in enumerate(pdb_contents):
            pose = self._pdb_content_to_pose(pdb_content)
            dropped_residues = self._find_dropped_residues(pose)
            binder_chain = binder_chains[i]
            target_chain = target_chains[i]

            iam = InterfaceAnalyzerMover()
            iam.set_interface(f"{target_chain}_{binder_chain}")
            sfxn = pyrosetta.create_score_function(scorefxn_name)
            iam.set_scorefunction(sfxn)
            iam.set_compute_packstat(True)
            iam.set_compute_interface_energy(True)
            iam.set_calc_dSASA(True)
            iam.set_calc_hbond_sasaE(True)
            iam.set_compute_interface_sc(True)
            iam.set_pack_separated(True)
            iam.apply(pose)
            data = iam.get_all_data()
            interface_sc = float(data.sc_value)
            interface_hbonds = int(data.interface_hbonds)
            interface_dG = float(iam.get_interface_dG())
            interface_dSASA = float(iam.get_interface_delta_sasa())
            interface_packstat = float(iam.get_interface_packstat())

            interface_residues = self._hotspot_residues_from_pose(pose, binder_chain, target_chain)
            interface_nres = len(interface_residues)
            apolar_aa = set("ACFILMPVWY")
            hydrophobic_count = sum(1 for aa in interface_residues.values() if aa in apolar_aa)
            interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100 if interface_nres else 0.0

            surface_hydrophobicity = self._surface_hydrophobicity(pose, binder_chain)

            results.append(
                {
                    "interface_sc": interface_sc,
                    "interface_hbonds": interface_hbonds,
                    "interface_dG": interface_dG,
                    "interface_dSASA": interface_dSASA,
                    "interface_packstat": interface_packstat,
                    "interface_hydrophobicity": interface_hydrophobicity,
                    "surface_hydrophobicity": surface_hydrophobicity,
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
        input_dict (dict): Input with "operation" key routing to
            sap/sasa/energy/relax/interface_analyzer.

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
            probe_radius=input_dict["probe_radius"],
            chain_ids_list=chain_ids_list,
        )
    if operation == "energy":
        return _scorer.compute_energy(
            pdb_contents,
            scorefxn_name=input_dict["scorefxn"],
            chain_ids_list=chain_ids_list,
        )
    if operation == "relax":
        return _scorer.compute_relax(
            pdb_contents,
            scorefxn_name=input_dict["scorefxn"],
            relax_cycles=input_dict["relax_cycles"],
            constrain_to_start=input_dict["constrain_to_start"],
            seed=input_dict["seed"],
            max_iter=input_dict.get("max_iter"),
        )
    if operation == "interface_analyzer":
        return _scorer.compute_interface_analyzer(
            pdb_contents,
            binder_chains=input_dict["binder_chains"],
            target_chains=input_dict["target_chains"],
            scorefxn_name=input_dict["scorefxn"],
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
