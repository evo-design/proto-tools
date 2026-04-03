<a href="https://bio-pro.mintlify.app/tools/structure-alignment/usalign"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# USalign

## Overview

USalign (Universal Structure alignment) extends TMalign to support monomers, multimers, and nucleic acid structures (`usalign-alignment`). It aligns two macromolecular structures using multimer-aware mode (`-mm 1 -ter 1`) and returns [TM-scores](https://en.wikipedia.org/wiki/Template_modeling_score) normalized by each structure's length. USalign is a compiled C++ binary that runs on CPU with no external dependencies.

## Background

**What does this tool measure/predict?**
USalign computes the TM-score for pairs of macromolecular structures, generalizing the TMalign algorithm to handle:
- **Monomeric proteins**: Standard pairwise alignment (equivalent to TMalign)
- **Multimeric complexes**: Joint alignment considering all chains, with optimal chain permutation for symmetric assemblies
- **Nucleic acid structures**: RNA and DNA structure comparison using nucleotide-aware scoring
- **Mixed complexes**: Protein-nucleic acid complexes aligned as unified structures

**Why is this important?**
- Complex structure validation: verify predicted multimer assemblies match experimental structures
- Protein design: evaluate how well designed complexes match target architectures
- RNA structure comparison: compare predicted RNA 3D structures to known folds
- Structural genomics: classify and compare protein complexes at the fold level

**Scientific foundation:**
USalign unifies multiple structure alignment methods (TMalign, MMalign, RNAalign) into a single universal framework. For multimers, it uses an iterative chain mapping algorithm that finds the optimal correspondence between chains in the two structures, then computes TM-score over all aligned chains jointly. The TM-score threshold of 0.5 for fold similarity applies equally to monomers and multimers. For nucleic acids, backbone phosphorus atoms replace C-alpha atoms as alignment anchors.

## How It Works

**Method overview:**
1. USalign reads two PDB structures and identifies chains and molecular types (protein, RNA, DNA)
2. For multimers: an optimal chain-to-chain mapping is determined using iterative alignment
3. C-alpha (protein) or phosphorus (nucleic acid) coordinates are extracted
4. [Structural alignment](https://en.wikipedia.org/wiki/Structural_alignment) is iteratively refined to maximize TM-score
5. TM-scores are reported normalized by each structure's total length

The tool runs with flags `-mm 1` (multimer mode) and `-ter 1` (treat each chain as a separate entity), which is the recommended mode for complex structure comparison.

**Key assumptions:**
- Input structures are in PDB format (text content, not file paths)
- Multi-chain structures have proper chain identifiers
- Structures should represent folded, stable conformations for meaningful comparison

**Limitations:**
- Requires C-alpha atoms (proteins) or phosphorus atoms (nucleic acids) in PDB format
- Very short structures (<20 residues/nucleotides) may produce unreliable TM-scores
- Chain mapping heuristics may not find optimal alignment for highly asymmetric complexes
- Ligands and small molecules are ignored

**Computational requirements:**
- **Hardware:** CPU only; no GPU required
- **Runtime:** <1 second for most alignments; up to a few seconds for very large multimeric complexes
- **Scalability:** Fast enough for batch comparisons of hundreds of complexes

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_text_1` | `str` | *required* | PDB content of structure 1 (query). Full PDB text, not a file path. |
| `pdb_text_2` | `str` | *required* | PDB content of structure 2 (reference). Full PDB text, not a file path. |

## Configuration

USalign uses default configuration inherited from `BaseConfig`. No tool-specific configuration parameters.

## Output Specification

```python
# Return type: USalignOutput
USalignOutput(
    tm_score_structure_1: float,  # TM-score normalized by Structure 1 length
    tm_score_structure_2: float,  # TM-score normalized by Structure 2 length
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `tm_score_structure_1` | `float` | `0.0 - 1.0` | TM-score normalized by query structure length |
| `tm_score_structure_2` | `float` | `0.0 - 1.0` | TM-score normalized by reference structure length. Use when evaluating a candidate against a fixed target. |

**Supported export formats:** `json`

## Interpreting Results

**Thresholds & decision boundaries:**
- **Same fold/architecture:** `TM-score > 0.5`: Structures share the same fold topology or complex architecture. Validated threshold for both monomers and multimers.
- **Similar topology:** `0.3 < TM-score <= 0.5`: Structures may share a similar fold; relationship is not definitive. Inspect visually.
- **Different fold:** `TM-score <= 0.3`: Structures are topologically unrelated.
- **Near-identical:** `TM-score > 0.9`: Nearly identical structures; differences are in loops or minor conformational changes.

**Interpreting edge cases:**
- For multimers, the alignment considers all chains jointly. Poor chain mapping (e.g., for symmetric complexes with ambiguous correspondence) may lower the score.
- The two TM-scores differ because they use different normalization lengths. For evaluating a predicted complex against a reference, use the score normalized by the reference length.
- Mixed protein-nucleic acid complexes: the TM-score reflects similarity of the overall assembly, with protein and nucleic acid regions contributing jointly.

## Quick Start Examples

**Example 1: Compare two multimeric complexes**
```python
from proto_tools.tools.structure_alignment.usalign import (
    USalignInput, USalignConfig, run_usalign,
)

with open("predicted_dimer.pdb") as f:
    query_pdb = f.read()
with open("reference_dimer.pdb") as f:
    reference_pdb = f.read()

inputs = USalignInput(pdb_text_1=query_pdb, pdb_text_2=reference_pdb)
result = run_usalign(inputs, USalignConfig())

print(f"TM-score (norm by query):     {result.tm_score_structure_1:.3f}")
print(f"TM-score (norm by reference): {result.tm_score_structure_2:.3f}")

if result.tm_score_structure_2 > 0.5:
    print("Complexes share the same architecture")
```

**Example 2: Validate predicted complex against experimental structure**
```python
from proto_tools.tools.structure_alignment.usalign import (
    USalignInput, USalignConfig, run_usalign,
)

# Compare Boltz2 prediction to experimental crystal structure
with open("boltz2_prediction.pdb") as f:
    predicted = f.read()
with open("experimental_4oo8.pdb") as f:
    experimental = f.read()

result = run_usalign(
    USalignInput(pdb_text_1=predicted, pdb_text_2=experimental),
    USalignConfig(),
)

# Score normalized by experimental structure length
tm_score = result.tm_score_structure_2
print(f"TM-score vs experimental: {tm_score:.3f}")
if tm_score > 0.7:
    print("Good structural agreement with experimental data")
```

**Example 3: Batch comparison of predicted complexes**
```python
from proto_tools.tools.structure_alignment.usalign import (
    USalignInput, USalignConfig, run_usalign,
)

with open("target_complex.pdb") as f:
    target_pdb = f.read()

candidates = ["design_1.pdb", "design_2.pdb", "design_3.pdb"]
for path in candidates:
    with open(path) as f:
        cand_pdb = f.read()
    result = run_usalign(
        USalignInput(pdb_text_1=cand_pdb, pdb_text_2=target_pdb),
        USalignConfig(),
    )
    status = "PASS" if result.tm_score_structure_2 > 0.5 else "FAIL"
    print(f"{path}: TM={result.tm_score_structure_2:.3f} [{status}]")
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Passing file paths instead of PDB content:** USalign expects PDB text content as strings, not file paths. Read the file first with `open(path).read()`.
2. **Using USalign for simple monomer comparison:** TMalign is faster and equivalent for monomeric proteins. Use USalign only when inputs may be multimeric or contain nucleic acids.
3. **Ignoring chain identifiers:** USalign uses chain IDs for its chain mapping algorithm. Ensure multi-chain PDB files have proper, distinct chain identifiers.
4. **Comparing structures of very different composition:** Comparing a monomer to a multimer, or a protein to a protein-RNA complex, may produce low TM-scores even if the protein components are similar.

**Tips for optimal results:**
- For protein complex design validation, TM-score > 0.5 is the minimum; > 0.7 indicates strong structural agreement
- Use USalign as the default choice when you aren't sure whether inputs are monomeric or multimeric
- For symmetric complexes, the chain mapping may find non-obvious but valid correspondences

**Edge cases to watch for:**
- Symmetric complexes (e.g., homodimers): chain mapping may be ambiguous but the TM-score should be consistent
- Structures with missing chains or partial occupancy may produce lower TM-scores
- Very large complexes (>5000 residues total) may take a few seconds instead of sub-second

## References

**Primary publication:**
- Zhang, C., Shine, M., Pyle, A.M. & Zhang, Y. (2022). "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." *Nature Methods*, 19(9), 1109-1115. [DOI: 10.1038/s41592-022-01585-1](https://doi.org/10.1038/s41592-022-01585-1)
- Summary: Introduces USalign as a universal structure alignment method that unifies TMalign, MMalign, and RNAalign into a single framework supporting all macromolecular types.

**Implementation:**
- GitHub: [https://github.com/pylelab/USalign](https://github.com/pylelab/USalign)

## Related Tools

**Tools often used together:**
- **`boltz2-prediction`** / **`alphafold3-prediction`**: Predict multi-chain complexes, then validate with USalign
- **`structure-tmscore` constraint**: Optimization constraint that uses TM-score tools internally

**Alternative tools (similar function):**
- **`tmalign-alignment`**: Faster monomer-only structural alignment. Use when inputs are guaranteed to be single-chain proteins.
