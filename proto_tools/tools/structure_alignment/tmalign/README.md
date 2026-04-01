<a href="https://bio-pro.mintlify.app/tools/structure-alignment/tmalign"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# TMalign

## Overview

TMalign performs pairwise protein structure alignment using the [TM-score](https://en.wikipedia.org/wiki/Template_modeling_score) metric (`tmalign-alignment`). It aligns two monomeric protein structures and returns TM-scores normalized by the length of each chain, providing a length-independent measure of structural similarity. TMalign is a compiled C++ binary that runs on CPU with no external dependencies.

## When to Use This Tool

**Primary use cases:**
- Compare two monomeric protein structures for topological similarity
- Quantify structural similarity with a length-independent metric (TM-score)
- Evaluate how well a predicted structure matches a reference/target (e.g., after ESMFold or AlphaFold2 prediction)
- Quality control in protein design pipelines: compare designed structure to intended fold
- Benchmark structure prediction methods by comparing predicted vs experimental structures

**When NOT to use this tool:**
- Compare multimeric complexes or protein-nucleic acid structures: use USalign (`usalign-alignment`) instead, which handles all macromolecular types
- Compute RMSD without alignment: use structure prediction tools' built-in RMSD or the `structure-rmsd` constraint
- Predict structures from sequence: use ESMFold, AlphaFold2, Boltz2, etc.
- Align sequences (not structures): use MAFFT (`mafft-alignment`) or BLAST (`blast-search`)

**Comparison with alternatives:**
- **TMalign vs USalign:** TMalign is faster and simpler for monomeric protein-protein comparisons. USalign extends the same algorithm to multimers, nucleic acids, and mixed complexes. For simple monomer comparisons, TMalign is preferred.
- **TMalign vs RMSD:** TM-score is length-independent (always 0-1), while RMSD grows with protein size. TM-score is better for comparing proteins of different lengths.

## Biological Background

**What does this tool measure/predict?**
TMalign computes the TM-score (Template Modeling score), which measures the topological similarity of two protein structures. It performs a [structural superposition](https://en.wikipedia.org/wiki/Structural_alignment) that maximizes the TM-score, then reports scores normalized by each chain's length.

**Why is this important?**
- Fold classification: TM-score > 0.5 reliably indicates proteins share the same fold topology, regardless of sequence similarity
- Structure prediction validation: compare predicted structures to experimental references
- Protein design: verify that designed proteins adopt the intended fold
- Evolutionary analysis: detect structural homologs even when sequence similarity is low (<20% identity)

**Scientific foundation:**
TM-score uses a length-dependent distance weighting scheme that emphasizes well-aligned residues and penalizes outliers less harshly than [RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions). The score is normalized by protein length, making it comparable across proteins of different sizes. The alignment algorithm uses an iterative heuristic that directly optimizes TM-score rather than RMSD, finding the structural superposition that maximizes topological similarity. Key properties:
- **Range:** (0, 1], where 1.0 = identical structures
- **Length-independent:** Unlike RMSD, TM-score does not increase with protein size
- **Fold discrimination:** TM-score > 0.5 reliably indicates the same fold (Zhang & Skolnick, 2004)
- **Random baseline:** Expected TM-score for randomly related proteins is ~0.17

## How It Works

**Method overview:**
1. TMalign reads two PDB structures and extracts C-alpha coordinates
2. An initial alignment is generated using secondary structure and sequence order heuristics
3. The alignment is iteratively refined to maximize TM-score using [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) with TM-score-based distance matrices
4. The final structural superposition is computed and TM-scores are reported normalized by each chain's length

**Key assumptions:**
- Both input structures are monomeric proteins with C-alpha coordinates
- Structures are in PDB format (text content, not file paths)
- Both structures should represent folded, globular proteins for meaningful comparison

**Limitations:**
- Monomeric proteins only: cannot handle multi-chain complexes (use USalign)
- Protein-only: no nucleic acid or ligand support
- Requires C-alpha atoms in PDB format
- Very short proteins (<20 residues) may produce unreliable TM-scores

**Computational requirements:**
- **Hardware:** CPU only; no GPU required
- **Runtime:** <1 second per alignment, even for large proteins
- **Scalability:** Extremely fast; suitable for all-vs-all comparisons of thousands of structures

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_text_1` | `str` | *required* | PDB content of structure 1 (query). Full PDB text, not a file path. |
| `pdb_text_2` | `str` | *required* | PDB content of structure 2 (reference). Full PDB text, not a file path. |

## Configuration

TMalign uses default configuration inherited from `BaseConfig`. No tool-specific configuration parameters.

## Output Specification

```python
# Return type: TMalignOutput
TMalignOutput(
    tm_score_chain_1: float,  # TM-score normalized by length of Chain 1 (query)
    tm_score_chain_2: float,  # TM-score normalized by length of Chain 2 (reference)
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `tm_score_chain_1` | `float` | `0.0 - 1.0` | TM-score normalized by query length. Use when evaluating query against a reference. |
| `tm_score_chain_2` | `float` | `0.0 - 1.0` | TM-score normalized by reference length. Use when evaluating how well a candidate matches a fixed target. |

**Supported export formats:** `json`

## Interpreting Results

**Thresholds & decision boundaries:**
- **Same fold:** `TM-score > 0.5`: Structures share the same fold topology with high confidence. This threshold is statistically validated and widely used in structural biology.
- **Similar fold:** `0.3 < TM-score <= 0.5`: Structures may share a similar fold or superfold, but the relationship is not definitive. Inspect visually.
- **Different fold:** `TM-score <= 0.3`: Structures are structurally unrelated or share only local structural motifs.
- **Near-identical:** `TM-score > 0.9`: Structures are nearly identical in topology; differences are in loop regions or minor conformational changes.
- **Random baseline:** `TM-score ~ 0.17`: Expected for randomly related proteins of typical length.

**Interpreting edge cases:**
- The two TM-scores (normalized by Chain 1 vs Chain 2) differ when chains have different lengths. The score normalized by the shorter chain is always higher.
- For comparing a designed protein to a target, use the score normalized by the **target** length (`tm_score_chain_2` if the target is structure 2).
- Very short proteins (<30 residues) may have artificially high TM-scores because the length normalization amplifies small alignments.
- TM-score measures **topological** similarity, not RMSD. Two structures can have high TM-score but moderate RMSD if the core is well-aligned with divergent loops.

## Quick Start Examples

**Example 1: Compare two structures**
```python
from proto_tools.tools.structure_alignment.tmalign import (
    TMalignInput, TMalignConfig, run_tmalign,
)

# Load PDB content (from files, predictions, or databases)
with open("predicted.pdb") as f:
    query_pdb = f.read()
with open("reference.pdb") as f:
    reference_pdb = f.read()

inputs = TMalignInput(pdb_text_1=query_pdb, pdb_text_2=reference_pdb)
result = run_tmalign(inputs, TMalignConfig())

print(f"TM-score (norm by query):     {result.tm_score_chain_1:.3f}")
print(f"TM-score (norm by reference): {result.tm_score_chain_2:.3f}")

if result.tm_score_chain_2 > 0.5:
    print("Structures share the same fold")
```

**Example 2: Validate designed protein against target fold**
```python
from proto_tools.tools.structure_alignment.tmalign import (
    TMalignInput, TMalignConfig, run_tmalign,
)
from proto_tools.tools.structure_prediction.esmfold import (
    run_esmfold_prediction, ESMFoldPredictionInput, ESMFoldPredictionConfig,
)

# Predict structure of designed sequence
designed_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDI"
pred_result = run_esmfold_prediction(
    ESMFoldPredictionInput(complexes=[designed_seq]),
    ESMFoldPredictionConfig(),
)
predicted_pdb = pred_result.structures[0].to_pdb_string()

# Compare to target structure
with open("target_fold.pdb") as f:
    target_pdb = f.read()

tm_result = run_tmalign(
    TMalignInput(pdb_text_1=predicted_pdb, pdb_text_2=target_pdb),
    TMalignConfig(),
)

# Use TM-score normalized by target length
print(f"TM-score vs target: {tm_result.tm_score_chain_2:.3f}")
```

**Example 3: Batch comparison of multiple candidates**
```python
from proto_tools.tools.structure_alignment.tmalign import (
    TMalignInput, TMalignConfig, run_tmalign,
)

with open("target.pdb") as f:
    target_pdb = f.read()

candidate_pdbs = ["cand1.pdb", "cand2.pdb", "cand3.pdb"]
results = []

for pdb_path in candidate_pdbs:
    with open(pdb_path) as f:
        cand_pdb = f.read()
    result = run_tmalign(
        TMalignInput(pdb_text_1=cand_pdb, pdb_text_2=target_pdb),
        TMalignConfig(),
    )
    results.append((pdb_path, result.tm_score_chain_2))

# Rank by TM-score (higher = better match to target)
for path, score in sorted(results, key=lambda x: -x[1]):
    status = "PASS" if score > 0.5 else "FAIL"
    print(f"{path}: TM-score={score:.3f} [{status}]")
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Passing file paths instead of PDB content:** TMalign expects PDB text content as strings, not file paths. Read the file first with `open(path).read()`.
2. **Using TMalign for multimers:** TMalign only handles monomeric proteins. For multi-chain complexes, use USalign (`usalign-alignment`).
3. **Comparing structures of vastly different sizes:** TM-score is length-independent, but aligning a 50-residue domain against a 500-residue protein may not be meaningful. Consider extracting the relevant domain first.
4. **Ignoring which normalization to use:** The two TM-scores can differ significantly. Choose the one normalized by the length that matters for your application (usually the target/reference).

**Tips for optimal results:**
- For protein design validation, use TM-score > 0.5 as a minimum threshold and > 0.7 as a strong match
- Combine with pLDDT filtering: first check that the predicted structure is well-folded (pLDDT > 0.8), then compare topology with TMalign
- For all-vs-all comparisons, TMalign is fast enough to run thousands of pairwise alignments

**Edge cases to watch for:**
- Very short proteins (<20 residues): TM-score normalization may produce misleadingly high scores
- Proteins with large disordered regions: the disordered tails will reduce TM-score even if structured domains match well
- NMR ensembles: only the first model in the PDB will be used

## References

**Primary publication:**
- Zhang, Y. & Skolnick, J. (2005). "TM-align: a protein structure alignment algorithm based on the TM-score." *Nucleic Acids Research*, 33(7), 2302-2309. [DOI: 10.1093/nar/gki524](https://doi.org/10.1093/nar/gki524)
- Summary: Introduces TMalign, a structure alignment algorithm that directly optimizes TM-score rather than RMSD, providing length-independent structural comparison of proteins.

**Implementation:**
- GitHub: [https://github.com/pylelab/USalign](https://github.com/pylelab/USalign) (TMalign is included in the USalign package)

**Additional resources:**
- Zhang, Y. & Skolnick, J. (2004). "Scoring function for automated assessment of protein structure template quality." *Proteins*, 57(4), 702-710. [DOI: 10.1002/prot.20264](https://doi.org/10.1002/prot.20264): Original TM-score paper.

## Related Tools

**Tools often used together:**
- **`esmfold-prediction`**: Predict structures from designed sequences, then compare to targets with TMalign
- **`alphafold2-prediction`**: Higher-accuracy structure prediction for important targets
- **`structure-tmscore` constraint**: Optimization constraint that uses TMalign internally for automated TM-score evaluation

**Alternative tools (similar function):**
- **`usalign-alignment`**: Universal structure alignment supporting multimers, nucleic acids, and mixed complexes. Use USalign when TMalign's monomer limitation is a problem.
