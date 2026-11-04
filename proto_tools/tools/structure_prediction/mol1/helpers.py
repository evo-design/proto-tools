"""Convert complexes to Mol-1 inference jobs."""

from typing import Any

from proto_tools.entities.complex import Chain
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.structure_prediction.shared_data_models import resolve_chain_ids


def complex_to_mol1_json(chains: list[Chain | Fragment], name: str) -> dict[str, Any]:
    """Build a native Mol-1 job preserving chain IDs and residue modifications.

    Args:
        chains (list[Chain | Fragment]): Polymer chains and ligand fragments.
        name (str): Internal prediction job label.

    Returns:
        dict[str, Any]: Native Mol-1 job dictionary.
    """
    sequences = []
    for chain, chain_id in zip(chains, resolve_chain_ids(chains), strict=True):
        entry: dict[str, Any] = {"count": 1, "id": [chain_id]}
        if isinstance(chain, Fragment):
            entry["ligand"] = f"CCD_{chain.ccd_code}" if chain.ccd_code else chain.smiles
            sequences.append({"ligand": entry})
            continue
        kind = chain.entity_type or ""
        key = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}[kind]
        entry["sequence"] = chain.sequence
        if chain.modifications:
            code_key, position_key = (
                ("ptmType", "ptmPosition") if kind == "protein" else ("modificationType", "basePosition")
            )
            entry["modifications"] = [
                {
                    code_key: f"CCD_{mod.modification_code.upper()}",
                    position_key: mod.position,
                }
                for mod in chain.modifications
            ]
        sequences.append({key: entry})
    return {"name": name, "sequences": sequences}
