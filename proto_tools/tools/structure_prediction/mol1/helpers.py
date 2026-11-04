"""Convert plain chain dictionaries to Mol-1 inference jobs."""

from typing import Any


def complex_to_mol1_json(chains: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Build a native Mol-1 job preserving chain IDs and residue modifications.

    Args:
        chains (list[dict[str, Any]]): Serialized chains with resolved unique IDs.
        name (str): Internal prediction job label.

    Returns:
        dict[str, Any]: Native Mol-1 job dictionary.
    """
    sequences = []
    for chain in chains:
        entry: dict[str, Any] = {"count": 1, "id": [chain["id"]]}
        kind = chain["entity_type"]
        if kind == "ligand":
            ccd = chain.get("ccd_code")
            entry["ligand"] = f"CCD_{ccd.upper().removeprefix('CCD_')}" if ccd else chain["smiles"]
            sequences.append({"ligand": entry})
            continue
        key = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}[kind]
        entry["sequence"] = chain["sequence"]
        modifications = chain.get("modifications", [])
        if modifications:
            code_key, position_key = (
                ("ptmType", "ptmPosition") if kind == "protein" else ("modificationType", "basePosition")
            )
            entry["modifications"] = [
                {
                    code_key: f"CCD_{mod['modification_code'].upper().removeprefix('CCD_')}",
                    position_key: mod["position"],
                }
                for mod in modifications
            ]
        sequences.append({key: entry})
    return {"name": name, "sequences": sequences}
