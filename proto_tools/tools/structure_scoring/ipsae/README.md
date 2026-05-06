<a href="https://bio-pro.mintlify.app/tools/structure-scoring/ipsae"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# IPSAE

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

IPSAE (Dunbrack 2025) is a comprehensive interface-quality scoring tool for cofolded protein complexes. Given a predicted structure with its PAE (Predicted Aligned Error) matrix, per-residue pLDDT, and CA coordinates, it computes five complementary interface metrics: **ipSAE**, **pDockQ2** (Zhu 2023), **LIS** (Kim 2024), **pDockQ** (Bryant 2022), and **ipTM**. The primary metric, ipSAE, uses an adaptive d0 normalization derived from the PAE matrix to better distinguish correct from incorrect interfaces compared to earlier PAE-based scores.

Supports structures from AlphaFold2, AlphaFold3, Chai-1, and Boltz.

Tool registry key: `ipsae-scoring`. Category: `structure_scoring`.

## Background

**What does this tool measure?**
Protein structure predictors emit per-residue pLDDT and a pairwise PAE matrix as confidence signals. Different scoring methods extract interface quality from these signals in different ways. IPSAE computes all of them in a single pass:

- **ipSAE** (interface predicted Structural Alignment Error): Analogous to TM-score but computed from the PAE matrix rather than superposed coordinates. Uses an adaptive d0 that accounts for interface geometry, making it more discriminative than ipTM for distinguishing near-native from incorrect interfaces.
- **ipTM** (interface predicted TM-score): TM-score computed from PAE with d0 based on chain lengths.
- **pDockQ2**: Combines interface pLDDT and normalized PAE through a sigmoid fit to DockQ ground truth (Zhu 2023).
- **pDockQ**: Earlier version using interface pLDDT and log(number of contacts) (Bryant 2022).
- **LIS** (Local Interaction Score): Fraction of interface residues with mean cross-chain PAE below a threshold (Kim 2024).

**When to use this tool:**
- You have a cofolded complex and want a comprehensive set of interface-quality metrics in one call.
- You want ipSAE as a more discriminative alternative to ipTM for ranking predictions.
- You are building a binder-design filter cascade and want multiple complementary scores.

**When NOT to use:**
- The structure lacks an interchain PAE matrix (e.g. single-chain predictions without PAE).
- You need physics-based interface scoring -- use `pyrosetta-sap`, `pyrosetta-sasa`, or `pyrosetta-energy`.

## Tools

### IPSAE Interface Scoring (`ipsae-scoring`)

Compute IPSAE interface metrics for a cofolded protein complex.

Dispatches to the vendored DunbrackLab/IPSAE script via ToolInstance,
writing temporary PDB + PAE JSON files and parsing the output.

## How It Works

1. Write the cofolded `Structure` as a temporary PDB and the PAE matrix as a JSON file.
2. Dispatch to the vendored DunbrackLab/IPSAE script via `ToolInstance`, passing the PDB, PAE, and cutoff parameters.
3. The IPSAE script computes all five metrics for every chain pair (both asymmetric/directional and symmetric/max variants).
4. Extract the `max`-type (symmetric) scores for the binder-target chain pair as the top-level metrics.
5. Return all per-chain-pair results in `chain_pair_results` for detailed inspection.

## Execution Modes

- **Hardware:** CPU only; runs in a standalone micromamba environment. No GPU required.
- **Runtime:** sub-second per complex for typical sizes (<1000 residues).
- **Scalability:** quadratic in total residue count due to pairwise distance and PAE computation.

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure` | `Structure` | *Required* | Cofolded complex. `b_factor_type` must be `PLDDT` or `NORMALIZED_PLDDT`. PAE matrix must be attached at `structure.metrics["pae_matrix"]` as a square `list[list[float]]` whose dimension matches the total residue count. |
| `binder_chain` | `str` | *Required* | Single-character chain ID of the binder. |
| `target_chains` | `list[str]` | *Required* | Target chain ID(s). Single-character entries; comma-separated strings are accepted and normalized. |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pae_cutoff` | `float` | `10.0` | PAE cutoff in angstroms for interface residue detection. |
| `distance_cutoff` | `float` | `10.0` | CA-CA distance cutoff in angstroms for contact detection. |

## Output Specification

```python
# Return type: IPSAEScoringOutput
{
    "metrics": {
        "ipsae": float,          # primary metric: ipSAE for binder-target interface [0, 1]
        "pdockq2": float,        # pDockQ2 score [0, ~1.5]
        "lis": float,            # Local Interaction Score [0, 1]
        "pdockq": float,         # pDockQ score [0, 1]
        "iptm_d0chn": float,     # ipTM from PAE with chain-length d0 [0, 1]
        "chain_pair_results": [  # full per-chain-pair breakdown
            {
                "chain1": str,
                "chain2": str,
                "pair_type": str,      # "asym" or "max"
                "ipsae": float,
                "ipsae_d0chn": float,
                "ipsae_d0dom": float,
                "iptm_af": float,      # -1.0 if unavailable
                "iptm_d0chn": float,
                "pdockq": float,
                "pdockq2": float,
                "lis": float,
            },
            ...
        ],
    }
}
```

## Interpreting Results

- `ipsae > 0.4`: generally indicates a reliably predicted interface (Dunbrack 2025).
- `ipsae` vs `iptm_d0chn`: ipSAE uses an adaptive d0 normalization that better separates correct from incorrect interfaces compared to ipTM. Prefer ipSAE as the primary ranking metric.
- `pdockq2 > 0.23`: conventional "acceptable" threshold from Zhu 2023's benchmark.
- `lis` close to 1.0: most interface residues have low cross-chain PAE, indicating high confidence in the interface geometry.
- `chain_pair_results` with `pair_type="max"` gives the symmetric (direction-independent) score for each chain pair. `pair_type="asym"` gives the directional scores (useful for asymmetric complexes).
- If no `max`-type pair is found for the binder-target combination, all top-level metrics default to 0.0 and a warning is logged.

## Quick Start Examples

```python
from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_scoring.ipsae import (
    IPSAEScoringConfig,
    IPSAEScoringInput,
    run_ipsae_scoring,
)

# Load a cofolded complex; attach PAE from your structure predictor.
structure = Structure.from_file(
    "complex.pdb",
    b_factor_type=BFactorType.PLDDT,
    metrics={"pae_matrix": my_pae_matrix},  # list[list[float]], N x N
)

inputs = IPSAEScoringInput(
    structure=structure,
    binder_chain="A",
    target_chains=["B"],
)
result = run_ipsae_scoring(inputs, IPSAEScoringConfig())

# Top-level metrics
print(f"ipSAE: {result.metrics.ipsae:.3f}")
print(f"pDockQ2: {result.metrics.pdockq2:.3f}")
print(f"LIS: {result.metrics.lis:.3f}")
print(f"pDockQ: {result.metrics.pdockq:.3f}")
print(f"ipTM: {result.metrics.iptm_d0chn:.3f}")
```

Adjust cutoffs for stricter interface definitions:

```python
config = IPSAEScoringConfig(pae_cutoff=8.0, distance_cutoff=8.0)
result = run_ipsae_scoring(inputs, config)
```

Inspect the per-chain-pair breakdown:

```python
for pair in result.metrics.chain_pair_results:
    print(f"{pair.chain1}-{pair.chain2} ({pair.pair_type}): "
          f"ipSAE={pair.ipsae:.3f}, pDockQ2={pair.pdockq2:.3f}, LIS={pair.lis:.3f}")
```

## Best Practices & Gotchas

- **Residue ordering invariant**: the PAE matrix must be indexed in the same residue order as the structure's residue iteration order. If your predictor emits PAE in a different order, remap it before attaching.
- **pLDDT scale**: upstream predictors differ (AF2/AF3 emit 0-100, others may emit 0-1). Set `b_factor_type` correctly on the `Structure` so internal rescaling works.
- **`binder_chain` must be single-character**: multi-character mmCIF chain labels need to be shortened via `Structure.to_pdb_with_chain_mapping()` before calling this tool.
- **Empty interface**: if the binder and target chains have no contacts within the cutoffs, all scores will be 0.0. Verify chain IDs and cutoff values.
- **Multiple target chains**: the tool finds the first `max`-type pair matching any binder-target combination. For multi-chain targets, inspect `chain_pair_results` for the full breakdown.

## References

- Dunbrack R.L. "Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it." *bioRxiv* (2025). [doi:10.1101/2025.02.10.637595](https://doi.org/10.1101/2025.02.10.637595)
- Zhu W., Shenoy A., Kundrotas P., Elofsson A. "Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes." *Bioinformatics* 39(7):btad424 (2023). [doi:10.1093/bioinformatics/btad424](https://doi.org/10.1093/bioinformatics/btad424)
- Bryant P., Pozzati G., Elofsson A. "Improved prediction of protein-protein interactions using AlphaFold2." *Nature Communications* 13:1265 (2022). [doi:10.1038/s41467-022-28865-w](https://doi.org/10.1038/s41467-022-28865-w)
- Kim A.-R., Hu Y., Comjean A., Rodiger J., Mohr S.E., Perrimon N. "Enhanced Protein-Protein Interaction Discovery via AlphaFold-Multimer." *bioRxiv* (2024). [doi:10.1101/2024.02.19.580970](https://doi.org/10.1101/2024.02.19.580970)

## Related Tools

- `pdockq2` -- standalone pDockQ2 scoring (if you only need pDockQ2 without the full IPSAE suite).
- `pyrosetta-sap` / `pyrosetta-sasa` / `pyrosetta-energy` -- physics-based interface scoring (complementary to IPSAE).
- `structure-metrics` -- per-structure secondary-structure and compactness metrics.
- Structure predictors that produce valid inputs: `alphafold3`, `chai1`, `boltz2`, `protenix`.
