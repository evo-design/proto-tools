# Random Nucleotide Sampling

## Overview

Random Nucleotide Sampling fills masked positions in DNA or RNA sequences with random bases drawn from an [IUPAC ambiguity code](https://en.wikipedia.org/wiki/Nucleic_acid_notation#IUPAC_notation) substitution pool. It supports configurable masking strategies, automatic DNA/RNA detection, and all 15 IUPAC degenerate base codes for controlling the substitution alphabet.

- **Tool key**: `random-nucleotide-sample`
- **Input**: DNA or RNA sequences (with optional `_` masks at positions to mutate)
- **Output**: Sequences with masked positions filled by random bases
- **Execution**: CPU only, no external dependencies

## When to Use This Tool

**Primary use cases:**
- Generating random nucleotide variants for directed evolution libraries
- Creating diverse sequence pools for experimental screening
- Introducing controlled randomness at specific positions in synthetic constructs
- Testing sequence robustness by random perturbation
- Building degenerate primer libraries with IUPAC-controlled diversity

**When NOT to use this tool:**
- For intelligent, model-guided mutations: use Evo1/Evo2 generators
- For codon-level protein mutagenesis: use `random-protein-sample` instead
- For structure-aware sequence design: use inverse folding tools (ProteinMPNN, LigandMPNN)
- For targeted single-nucleotide variants: manually specify mutations

## Biological Background

**What does this tool do?**
This tool performs random [mutagenesis](https://en.wikipedia.org/wiki/Mutagenesis) at the nucleotide level. Given a DNA or RNA sequence, it identifies positions to mutate (either pre-marked with `_` or selected by a masking strategy) and replaces them with random bases drawn from a specified substitution pool.

**Why is this important?**
Random mutagenesis is a foundational technique in:
- **[Directed evolution](https://en.wikipedia.org/wiki/Directed_evolution)**: Creating libraries of sequence variants for functional screening
- **Combinatorial library design**: Generating diversity at specific positions in promoters, UTRs, or coding regions
- **Robustness testing**: Evaluating how sensitive a designed sequence is to random perturbations
- **Degenerate codon libraries**: Using IUPAC codes to control which bases appear at each position

**IUPAC substitution schemes:**
The substitution scheme controls which bases can appear at masked positions:

| Code | Bases | Description |
|------|-------|-------------|
| `N` | A, C, G, T | Any base (maximum diversity) |
| `R` | A, G | Purines only |
| `Y` | C, T | Pyrimidines only |
| `S` | G, C | Strong (3 hydrogen bonds) |
| `W` | A, T | Weak (2 hydrogen bonds) |
| `K` | G, T | Keto |
| `M` | A, C | Amino |
| `B` | C, G, T | Not A |
| `D` | A, G, T | Not C |
| `H` | A, C, T | Not G |
| `V` | A, C, G | Not T |

## How It Works

1. **Masking**: Positions to mutate are identified either from pre-existing `_` characters in the input or by applying a `MaskingStrategy` (random selection, entropy-based, or max-logit)
2. **Substitution**: Each masked position is independently replaced with a random base drawn uniformly from the IUPAC substitution pool
3. **RNA handling**: If the input contains `U` (or `sequence_type="rna"`), output bases are automatically converted from T to U

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | DNA or RNA sequences. Use `_` to mark positions for mutation. Accepts a single string (auto-wrapped) or a list. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `masking_strategy` | `MaskingStrategy` | Random 30% | Controls which positions to mask for sampling. Ignored if sequences already contain `_` masks. |
| `substitution_scheme` | `str` | `"N"` | IUPAC ambiguity code defining the substitution pool (see table above). |
| `sequence_type` | `Literal["auto", "dna", "rna"]` | `"auto"` | Sequence type. `"auto"` detects RNA by presence of U. |
| `seed` | `Optional[int]` | `None` | Random seed for reproducible sampling. |

### Sweep Priorities

1. **`substitution_scheme`**: Most impactful. Controls diversity of the generated library (e.g., `N` for full diversity, `R` for purine-only transitions).
2. **`masking_strategy`**: Controls how many and which positions are mutated.

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Sequences with masked positions filled by random bases |

Export formats: `fasta`, `txt`, `json`

## Quick Start Examples

**Example 1: Pre-masked positions**
```python
from proto_tools.tools.mutagenesis import (
    RandomNucleotideSampleInput, RandomNucleotideSampleConfig,
    run_random_nucleotide_sample,
)

# Underscore marks positions to randomize
inputs = RandomNucleotideSampleInput(sequences=["ACGT_CGT_A"])
result = run_random_nucleotide_sample(inputs)
print(result.sequences[0])  # e.g., "ACGTGCGTCA"
```

**Example 2: Purine-only substitutions**
```python
from proto_tools.tools.mutagenesis import (
    RandomNucleotideSampleInput, RandomNucleotideSampleConfig,
    run_random_nucleotide_sample,
)

inputs = RandomNucleotideSampleInput(sequences=["ACGT_CGT_A"])
config = RandomNucleotideSampleConfig(substitution_scheme="R")  # A or G only
result = run_random_nucleotide_sample(inputs, config)
print(result.sequences[0])  # e.g., "ACGTACGTGA"
```

**Example 3: Auto-masking with MaskingStrategy**
```python
from proto_tools.tools.mutagenesis import (
    RandomNucleotideSampleInput, RandomNucleotideSampleConfig,
    run_random_nucleotide_sample,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy

config = RandomNucleotideSampleConfig(
    masking_strategy=MaskingStrategy(num_mutations=3),
    substitution_scheme="N",
    seed=42,
)
inputs = RandomNucleotideSampleInput(sequences=["ACGTACGTAC"])
result = run_random_nucleotide_sample(inputs, config)
print(result.sequences[0])  # 3 random positions mutated
```

**Example 4: RNA sequences**
```python
from proto_tools.tools.mutagenesis import (
    RandomNucleotideSampleInput, run_random_nucleotide_sample,
)

# U in input triggers automatic RNA mode
inputs = RandomNucleotideSampleInput(sequences=["ACGU_CGU_A"])
result = run_random_nucleotide_sample(inputs)
print(result.sequences[0])  # Output uses U instead of T
```

## Best Practices & Gotchas

- **Pre-masked vs auto-masked**: If your sequences already contain `_`, the masking strategy is ignored. Remove `_` characters if you want the strategy to select positions.
- **IUPAC codes are case-insensitive** in the config but the tool normalizes them internally.
- **RNA auto-detection**: The tool detects RNA by the presence of `U` in the input. Use `sequence_type="rna"` to force RNA mode if your input contains only `_` masks.
- **Uniform sampling within pool**: Each base in the IUPAC pool is equally likely. There is no frequency weighting within a substitution scheme.
- **Reproducibility**: Set `seed` for deterministic output across runs.

## References

- IUPAC nucleotide codes: [https://www.bioinformatics.org/sms/iupac.html](https://www.bioinformatics.org/sms/iupac.html)

## Related Tools

**Tools often used together:**
- `evo2-sample`: Model-guided DNA sequence generation (smarter than random)
- `random-protein-sample`: Random mutagenesis at the protein/codon level

**Alternative tools:**
- Evo1/Evo2 generators: Use learned sequence distributions instead of uniform random sampling
