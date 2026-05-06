<a href="https://bio-pro.mintlify.app/tools/structure-scoring/pdockq2"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# pDockQ2

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

pDockQ2 (Zhu et al. 2023) scores the interface quality of a predicted protein complex from AlphaFold-Multimer / AlphaFold3 / Chai-1 / Boltz-2 / Protenix outputs. It combines per-residue pLDDT and the PAE (Predicted Aligned Error) matrix into a single scalar in `[0, 1]`, where higher values indicate more reliably predicted interfaces. Commonly used as a filter gate in binder-design pipelines; a value above roughly `0.23` is typically treated as "acceptable" in published benchmarks.

Tool registry key: `pdockq2`. Category: `structure_scoring`.

## Background

**What does this tool measure?**
Protein-protein interface predictions can look geometrically reasonable while being unreliable. AlphaFold-Multimer (and successors) emit two orthogonal confidence signals: per-residue pLDDT (local-structure confidence) and PAE (pairwise residue-position confidence). pDockQ2 reduces both into one scalar per chain using two terms:

- Mean pLDDT over binder-side interface residues, penalizing flexible/unstructured interfaces.
- Mean of `1 / (1 + (PAE / 10)^2)` over CA-CA residue pairs within the interface cutoff, penalizing interfaces with poor inter-chain geometric constraints.

These feed a sigmoid whose parameters were fit by Zhu et al. 2023 against ground-truth DockQ values on the AlphaFold-Multimer benchmark.

**When to use this tool:**
- You have a cofolded complex and need a fast scalar indicator of interface-prediction reliability.
- You are building a binder-design or complex-prediction filter cascade that combines pLDDT and PAE into one gate.

**When NOT to use:**
- The structure lacks an interchain PAE matrix (e.g. ESMFold single-chain prediction without PAE).
- You need a physics-based interface score — use `pyrosetta-sap`, `pyrosetta-sasa`, or `pyrosetta-energy`.

## Tools

### pDockQ2 Interface Quality (`pdockq2`)

Compute pDockQ2 (Zhu 2023) for a cofolded protein complex.

Returns the mean per-chain `pmidockq` over chains in `target_chains`
that contact `binder_chain`, plus per-chain debug rows.

## How It Works

1. Extract per-residue CA coordinates and per-residue pLDDT (0-1) from the cofolded `Structure`. The pLDDT is rescaled to 0-100 internally so the sigmoid parameters match the published fit.
2. Read the full PAE matrix from `structure.metrics["pae_matrix"]`. The tool asserts the matrix is square and matches the total residue count emitted by `Structure.per_residue_plddt`.
3. For each chain, find CA-CA contact residue pairs against every other chain within `distance_cutoff` (default 10.0 Å). Collect the per-chain interface pLDDT and the normalized-PAE average over those pairs.
4. Apply the Zhu-2023 sigmoid to `if_plddt * norm_pae` for each chain to get `pmidockq`.
5. Aggregate: average `pmidockq` over chains in `inputs.target_chains` that contact `inputs.binder_chain`.

## Execution Modes

- **Hardware:** CPU only; pure numpy math. No GPU, no standalone environment.
- **Runtime:** sub-second per complex for typical sizes (<1000 residues).
- **Scalability:** quadratic in total residue count due to pairwise distance computation.

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure` | `Structure` | *Required* | Cofolded complex. `b_factor_type` must be `PLDDT` or `NORMALIZED_PLDDT`. PAE matrix must be attached at `structure.metrics["pae_matrix"]` as a square `list[list[float]]` whose dimension matches the total residue count. |
| `binder_chain` | `str` | *Required* | Single-character chain ID of the binder. |
| `target_chains` | `list[str]` | *Required* | Target chain ID(s). Single-character entries; comma-separated strings are accepted and normalized. |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `distance_cutoff` | `float` | `10.0` | CA-CA distance cutoff in Å for interface residue detection (Zhu 2023). |

## Output Specification

```python
# Return type: PDockQ2Output
{
    "metrics": {
        "pdockq2": float,                  # overall score in [0, 1]
        "avg_interface_plddt": float,      # mean target-chain interface pLDDT (0-100)
        "avg_interface_pae": float,        # mean normalized PAE in [0, 1]
        "num_interface_contacts": int,     # # binder x target residue pairs in contact
        "interfaces": [
            {"chain_id": str, "neighbor_chains": str,
             "if_plddt": float, "norm_pae": float, "pmidockq": float},
            ...
        ],
    }
}
```

## Interpreting Results

- `pdockq2 > 0.23`: conventional "acceptable" threshold from Zhu 2023's AlphaFold-Multimer benchmark; values above this gate are consistent with reliably predicted interfaces.
- `pdockq2 < 0.1`: low-confidence interface; usually reflects either low interface pLDDT or high PAE between the binder and target.
- `num_interface_contacts == 0`: binder and target chains are not in contact within `distance_cutoff` — score is set to 0.0 and a warning is logged. Verify `binder_chain` and `target_chains` correctness before interpreting.
- `interfaces` rows expose the per-chain breakdown (useful when debugging multi-chain targets).

## Quick Start Examples

```python
from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools import PDockQ2Config, PDockQ2Input, run_pdockq2

# Load a cofolded complex; attach PAE emitted by your structure predictor.
structure = Structure.from_file(
    "complex.pdb",
    b_factor_type=BFactorType.PLDDT,
    metrics={"pae_matrix": my_pae_matrix},  # list[list[float]], N x N over all residues
)

inputs = PDockQ2Input(structure=structure, binder_chain="A", target_chains=["B"])
result = run_pdockq2(inputs, PDockQ2Config())
print(result.metrics.pdockq2, result.metrics.num_interface_contacts)
```

Tighten the cutoff for stricter interface definitions:

```python
result = run_pdockq2(inputs, PDockQ2Config(distance_cutoff=8.0))
```

Inspect the per-chain breakdown (useful for multi-chain targets):

```python
for row in result.metrics.interfaces:
    print(row.chain_id, row.neighbor_chains, row.if_plddt, row.norm_pae, row.pmidockq)
```

## Best Practices & Gotchas

- **Residue ordering invariant**: the PAE matrix must be indexed in the same residue order as `Structure.per_residue_plddt` (i.e. gemmi model → chain → residue iteration order). If your structure predictor emits PAE in a different order, remap it before attaching.
- **pLDDT scale**: upstream predictors differ (AF2/AF3 emit 0-100, ESMFold emits 0-1). `Structure.b_factor_type` captures this; the tool uses `Structure.per_residue_plddt` which normalizes to 0-1 and rescales internally to the 0-100 scale the sigmoid was fit on.
- **`binder_chain` must be single-character**: multi-character mmCIF chain labels need to be shortened via `Structure.to_pdb_with_chain_mapping()` before calling this tool.
- **Empty contact set**: a zero score with zero contacts typically means the caller passed `binder_chain`/`target_chains` that do not interact. Check structure orientation and cutoff.
- **Contact detection uses CA atoms only**: every residue is treated by its CA position, without fallback to CB. This is appropriate for model-predicted structures where every modeled residue has CA; experimental structures with disordered backbones may skip residues with missing CA.
- **Interface pLDDT is contact-pair weighted** (Zhu 2023). A residue that contacts `k` cross-chain partners contributes its pLDDT `k` times to the interface mean — not once per unique residue.
- **Chains with no cross-chain contacts score exactly 0.0**. A caller comparing raw `pmidockq` values across predictions should be aware this short-circuits past the sigmoid.

## References

- Zhu W., Shenoy A., Kundrotas P., Elofsson A. "Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes." *Bioinformatics* 39(7):btad424 (2023). [doi:10.1093/bioinformatics/btad424](https://doi.org/10.1093/bioinformatics/btad424)
- Bryant P., Pozzati G., Elofsson A. "Improved prediction of protein-protein interactions using AlphaFold2." *Nature Communications* 13:1265 (2022). [doi:10.1038/s41467-022-28865-w](https://doi.org/10.1038/s41467-022-28865-w)

## Related Tools

- `pyrosetta-sap` / `pyrosetta-sasa` / `pyrosetta-energy` — physics-based interface scoring (complementary to pDockQ2).
- `structure-metrics` — per-structure secondary-structure and compactness metrics.
- Structure predictors that produce valid inputs: `alphafold3`, `chai1`, `boltz2`, `protenix`.
