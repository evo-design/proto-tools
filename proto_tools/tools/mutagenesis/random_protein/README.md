<a href="https://bio-pro.mintlify.app/tools/mutagenesis/random-protein"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Random Protein Sampling

## Overview

Random Protein Sampling fills masked positions in protein sequences with random amino acids drawn from a configurable [codon scheme](https://en.wikipedia.org/wiki/Genetic_code). Different codon schemes produce different amino acid frequency distributions, mirroring the biases of real degenerate codon libraries used in experimental directed evolution.

- **Tool key**: `random-protein-sample`
- **Input**: Protein sequences (with optional `_` masks at positions to mutate)
- **Output**: Sequences with masked positions filled by random amino acids
- **Execution**: CPU only, no external dependencies

## Background

**What does this tool do?**
This tool performs random [mutagenesis](https://en.wikipedia.org/wiki/Mutagenesis) at the protein level. Given a protein sequence, it identifies positions to mutate and replaces them with random amino acids sampled according to a codon scheme's frequency distribution.

**Why is this important?**
Random protein mutagenesis is central to [protein engineering](https://en.wikipedia.org/wiki/Protein_engineering):
- **Directed evolution libraries**: Experimental mutagenesis uses degenerate codons (e.g., NNK) that encode amino acids with non-uniform frequencies. Simulating these distributions computationally helps predict library diversity.
- **Baseline comparisons**: Random mutagenesis provides a null model for evaluating whether model-guided designs (ESM2, ProteinMPNN) outperform chance.
- **Combinatorial screening**: Randomizing a few key positions generates focused libraries for functional assays.

**Codon schemes:**
Each scheme represents a degenerate codon pattern used in synthetic biology. The scheme determines which amino acids can appear and their relative frequencies:

| Scheme | Codons | Amino Acids | Use Case |
|--------|--------|-------------|----------|
| `UNIFORM` | N/A | All 20, equal weight | Unbiased random sampling |
| `NNK` | 32 | All 20 | Standard library design, reduced stop codons |
| `NNS` | 32 | All 20 | Alternative to NNK, same coverage |
| `NDT` | 12 | 12 AAs | Zero-redundancy, small focused libraries |
| `DBK` | 18 | 18 AAs | Specialized scheme |
| `NRT` | 8 | 8 AAs | Purine/pyrimidine combination |

In NNK/NNS schemes, amino acids encoded by more codons (e.g., Leu, Ser, Arg) appear more frequently than those with fewer codons (e.g., Met, Trp), matching real experimental library distributions.

## How It Works

1. **Masking**: Positions to mutate are identified either from pre-existing `_` characters in the input or by applying a `MaskingStrategy` (random selection, entropy-based, or max-logit)
2. **Codon expansion**: The codon scheme is expanded to compute amino acid sampling weights proportional to codon count
3. **Substitution**: Each masked position is independently replaced with a random amino acid drawn from the scheme's probability distribution

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences (20 standard amino acids + X for unknown). Use `_` to mark positions for mutation. Accepts a single string (auto-wrapped) or a list. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `masking_strategy` | `MaskingStrategy` | Random 30% | Controls which positions to mask for sampling. Ignored if sequences already contain `_` masks. |
| `codon_scheme` | `str` | `"UNIFORM"` | Codon scheme for amino acid sampling. One of: `UNIFORM`, `NNK`, `NNS`, `NDT`, `DBK`, `NRT`. |
| `seed` | `Optional[int]` | `None` | Random seed for reproducible sampling. |

### Sweep Priorities

1. **`codon_scheme`**: Most impactful. Determines amino acid diversity and frequency distribution of the library.
2. **`masking_strategy`**: Controls how many and which positions are mutated.

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Sequences with masked positions filled by random amino acids |

Export formats: `fasta`, `txt`, `json`

## Quick Start Examples

**Example 1: Pre-masked positions with uniform sampling**
```python
from proto_tools.tools.mutagenesis import (
    RandomProteinSampleInput, RandomProteinSampleConfig,
    run_random_protein_sample,
)

inputs = RandomProteinSampleInput(sequences=["MK_AY_AKQR"])
result = run_random_protein_sample(inputs)
print(result.sequences[0])  # e.g., "MKTAYFAKQR"
```

**Example 2: NNK codon scheme**
```python
from proto_tools.tools.mutagenesis import (
    RandomProteinSampleInput, RandomProteinSampleConfig,
    run_random_protein_sample,
)

# NNK mimics real degenerate codon library frequencies
inputs = RandomProteinSampleInput(sequences=["MK_AY_AKQR"])
config = RandomProteinSampleConfig(codon_scheme="NNK")
result = run_random_protein_sample(inputs, config)
print(result.sequences[0])  # Leu, Ser, Arg more likely than Met, Trp
```

**Example 3: Auto-masking with MaskingStrategy**
```python
from proto_tools.tools.mutagenesis import (
    RandomProteinSampleInput, RandomProteinSampleConfig,
    run_random_protein_sample,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy

config = RandomProteinSampleConfig(
    masking_strategy=MaskingStrategy(num_mutations=3),
    codon_scheme="UNIFORM",
    seed=42,
)
inputs = RandomProteinSampleInput(sequences=["MKTAYIAKQR"])
result = run_random_protein_sample(inputs, config)
print(result.sequences[0])  # 3 random positions mutated
```

**Example 4: Compare codon scheme diversity**
```python
from proto_tools.tools.mutagenesis import (
    RandomProteinSampleInput, RandomProteinSampleConfig,
    run_random_protein_sample,
)

for scheme in ["UNIFORM", "NNK", "NDT"]:
    config = RandomProteinSampleConfig(codon_scheme=scheme, seed=42)
    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["____"]),
        config,
    )
    print(f"{scheme}: {result.sequences[0]}")
```

## Best Practices & Gotchas

- **Pre-masked vs auto-masked**: If your sequences already contain `_`, the masking strategy is ignored. Remove `_` characters if you want the strategy to select positions.
- **UNIFORM vs codon-based**: Use `UNIFORM` for unbiased random sampling. Use `NNK`/`NNS` to simulate real experimental library distributions where codon degeneracy biases amino acid frequencies.
- **NDT for small libraries**: NDT encodes only 12 amino acids with zero redundancy, giving the most even distribution among those 12. Ideal for small combinatorial libraries.
- **Reproducibility**: Set `seed` for deterministic output across runs.
- **Stop codons**: `NNK` and `NNS` encode 1 stop codon (TAG or TGA) out of 32 codons. The tool samples amino acids only, so stop codons are excluded from the sampling distribution.

## References

- Reetz, M.T. & Carballeira, J.D. (2007). "Iterative saturation mutagenesis (ISM) for rapid directed evolution of functional enzymes." *Nature Protocols*, 2(4), 891-903. [DOI: 10.1038/nprot.2007.72](https://doi.org/10.1038/nprot.2007.72)
- Nov, Y. (2012). "When second best is good enough: another probabilistic look at saturation mutagenesis." *Applied and Environmental Microbiology*, 78(1), 258-262. [DOI: 10.1128/AEM.06265-11](https://doi.org/10.1128/AEM.06265-11)

## Related Tools

**Tools often used together:**
- `random-nucleotide-sample`: Random mutagenesis at the DNA level
- `proteinmpnn-sample`: Structure-aware sequence design (smarter than random)
- `esm2-sample`: Language model-guided protein mutations

**Alternative tools:**
- ESM2/ESM3 sampling: Use learned protein distributions instead of uniform/codon-based random sampling
- ProteinMPNN: Structure-conditioned inverse folding for informed sequence design
