# Evo1 Sample

## Overview
Evo1 is a 7-billion parameter DNA language model built on the StripedHyena architecture, trained on 2.7 million prokaryotic and phage genomes from the OpenGenome dataset. This tool performs autoregressive DNA sequence generation from prompts and optionally scores generated sequences by mean log-probability.

## Biological Background

**What does this tool measure/predict?**
Evo1 generates novel DNA sequences by learning the statistical patterns of prokaryotic and phage genomes. It can also score sequences by log-likelihood, providing a measure of how "natural" a generated sequence appears relative to the training distribution.

**Why is this important?**
Generative DNA models enable de novo design of biological sequences — genes, operons, CRISPR systems, and mobile genetic elements — without requiring template sequences. By learning the grammar of genomes at single-nucleotide resolution, Evo1 can propose functional DNA that respects codon usage, regulatory signals, and structural constraints.

**Scientific foundation:**
Evo1 uses the StripedHyena architecture, a hybrid state-space/attention model that processes DNA at single-nucleotide resolution. Unlike transformer-only models, StripedHyena supports efficient long-range sequence modeling up to 131k tokens. The model is trained with a standard autoregressive (next-token prediction) objective on raw genomic DNA.

## When to Use This Tool

**Primary use cases:**
- De novo generation of prokaryotic genes, operons, or CRISPR loci from DNA prompts
- Extending partial DNA sequences with biologically plausible continuations
- Scoring candidate DNA sequences by log-likelihood
- Generating diverse sequence variants via temperature and top-k sampling

**When NOT to use this tool:**
- For protein sequences: use ESM2/ESM3 or ProGen2
- For eukaryotic genome-scale tasks: use Evo2 (trained on eukaryotic genomes)
- For transcriptional activity prediction: use Enformer or Borzoi
- For short oligo/primer design: simpler tools may suffice

**Comparison with alternatives:**
- **Evo2**: Successor model with larger training set including eukaryotic genomes and context windows up to 1M tokens. Use Evo2 for eukaryotic sequences or when longer context is needed.
- **Evo1**: Better suited for prokaryotic/phage sequence generation, especially with the CRISPR and transposon fine-tuned checkpoints.

## How It Works

**Method overview:**
Evo1 performs autoregressive generation: given a prompt DNA sequence, it predicts one nucleotide at a time, sampling from the predicted distribution. The model uses StripedHyena layers that combine gated state-space layers with attention, enabling efficient processing of long genomic sequences. Generation uses top-k and nucleus (top-p) sampling with configurable temperature to control diversity.

**Key assumptions:**
- Input prompts are valid DNA strings (characters A, T, C, G)
- The model is most reliable for prokaryotic and phage-like sequences (its training domain)
- Generated sequences require downstream validation (e.g., ORF finding, structure prediction)

**Limitations:**
- Context window is 8,192 tokens for base model, 131,072 for the 131k variant
- Single-nucleotide tokenization means generation is slower per base than codon-level models
- Fine-tuned checkpoints (CRISPR, transposon) are specialized and may perform poorly outside their domain

**Computational requirements:**
- **Hardware:** NVIDIA GPU with ≥24 GB VRAM (A100 recommended); CPU mode available but very slow
- **Runtime:** ~1-5 minutes for 1,000 tokens on a single A100 GPU, depending on model variant
- **Scalability:** Supports batched generation via `batch_size` parameter

## Model Variants

| Checkpoint | Context Length | Description |
|------------|--------------|-------------|
| `evo-1-8k-base` | 8,192 | General-purpose prokaryotic/phage DNA model |
| `evo-1-131k-base` | 131,072 | Long-context variant for multi-gene regions |
| `evo-1-8k-crispr` | 8,192 | Fine-tuned on CRISPR loci for CRISPR system generation |
| `evo-1-8k-transposon` | 8,192 | Fine-tuned on transposable elements |

## Important Parameters

**Input parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `List[str]` | *Required* | Prompt DNA sequence(s) for generation |

**Configuration parameters:**

| Parameter | Type | Default | Sweep Range | Description |
|-----------|------|---------|-------------|-------------|
| `model_name` | `str` | `"evo-1-8k-base"` | N/A | Evo1 model checkpoint to use (see Model Variants) |
| `num_tokens` | `int` | `100` | `100 - 10000` | Number of tokens to generate per prompt |
| `temperature` | `float` | `1.0` | `0.5 - 1.5` | Sampling temperature; lower = more conservative |
| `top_k` | `int` | `4` | `1 - 20` | Top-k sampling; limits to k most likely tokens |
| `top_p` | `float` | `1.0` | `0.8 - 1.0` | Nucleus sampling threshold |
| `prepend_prompt` | `bool` | `False` | N/A | Whether to prepend prompt to generated sequence |
| `batch_size` | `Optional[int]` | `128` | N/A | Max prompts per GPU batch |
| `device` | `str` | `"cuda"` | N/A | Device to run on |

**Parameters to prioritize for sweeps:**
1. **`temperature`**: Controls diversity of generated sequences. Lower values (0.5-0.8) produce conservative, high-likelihood sequences; higher values (1.0-1.5) increase diversity but may reduce quality.
2. **`top_k`**: With `top_k=4` (DNA alphabet size), generation stays close to the most likely nucleotide at each position. Increasing top_k allows more exploratory generation.
3. **`num_tokens`**: Determines output length. For gene-length sequences, use 1,000-5,000; for full operons, 5,000-10,000+.

---

**Output specification:**

```python
# Return type: Evo1SampleOutput
{
    "sequences": List[str],           # Generated DNA sequences
    "scores": Optional[List[float]],  # Mean log-probability per sequence (if computed)
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `sequences` | `List[str]` | Variable length | Generated DNA sequences, one per input prompt |
| `scores` | `Optional[List[float]]` | Negative floats | Mean log-probability; higher (less negative) = more likely under model |

**Supported export formats:** `fasta`, `txt`, `json`

## Best Practices & Gotchas

**Parameter tuning:**
- **`temperature`**:
  - Low values (0.5-0.8): More repetitive but higher per-token likelihood; good for generating sequences close to training distribution
  - High values (1.0-1.5): More diverse; useful for exploring sequence space but may produce less realistic sequences

**Common mistakes:**
1. **Using the wrong checkpoint for your task**: The `evo-1-8k-crispr` checkpoint is fine-tuned specifically for CRISPR loci — it will not perform well for general gene generation. Use `evo-1-8k-base` for general purposes.
2. **Expecting eukaryotic-quality generation**: Evo1 was trained on prokaryotic and phage genomes. For eukaryotic sequences, use Evo2.
3. **Ignoring prompt length limits**: The 8k model has an 8,192-token context window. Prompt + generated tokens must fit within this limit.

**Tips for optimal results:**
- Use biologically meaningful prompts (e.g., start codons, promoter regions) rather than random DNA
- For CRISPR system generation, use the `evo-1-8k-crispr` fine-tuned checkpoint
- Generated sequences should be validated with downstream tools (Prodigal for ORFs, ESMFold for structure)

**Edge cases to watch for:**
- Very short prompts (<10 nt) may produce less coherent output
- Homopolymer runs in prompts can cause repetitive generation
- The model may generate sequences longer than typical genes; use downstream ORF detection to identify coding regions

## References

**Primary publication:**
- Nguyen, E., Poli, M., Durrant, M.G., et al. (2024). "Sequence modeling and design from molecular to genome scale with Evo." *Science* 386(6723). [DOI: 10.1126/science.ado9336](https://doi.org/10.1126/science.ado9336)
- Summary: Introduces the Evo model family for DNA sequence modeling, demonstrating generation of functional CRISPR systems and transposable elements using a StripedHyena architecture trained on prokaryotic and phage genomes.

**Implementation:**
- GitHub: [https://github.com/evo-design/evo](https://github.com/evo-design/evo)
- Model weights: Available via Hugging Face (`togethercomputer/evo-1-8k-base`, etc.)

## Quick Start Examples

**Example 1: Basic DNA generation**
```python
from bio_programming_tools.tools.causal_models.evo1 import (
    run_evo1_sample, Evo1SampleInput, Evo1SampleConfig
)

inputs = Evo1SampleInput(prompts=["ATGCGTAAACGATTGCAGT"])
config = Evo1SampleConfig(
    num_tokens=500,
    temperature=0.8,
    top_k=4,
)

result = run_evo1_sample(inputs, config)
print(f"Generated: {result.sequences[0][:100]}...")
print(f"Total length: {len(result.sequences[0])}")
```

**Example 2: CRISPR-specific generation**
```python
from bio_programming_tools.tools.causal_models.evo1 import (
    run_evo1_sample, Evo1SampleInput, Evo1SampleConfig
)

inputs = Evo1SampleInput(prompts=["ATGCGTAAACGATTGCAGT"])
config = Evo1SampleConfig(
    model_name="evo-1-8k-crispr",
    num_tokens=3000,
    temperature=1.0,
)

result = run_evo1_sample(inputs, config)
print(f"Generated CRISPR locus: {len(result.sequences[0])} bp")
```

**Example 3: Batched generation with scoring**
```python
from bio_programming_tools.tools.causal_models.evo1 import (
    run_evo1_sample, Evo1SampleInput, Evo1SampleConfig
)

inputs = Evo1SampleInput(prompts=[
    "ATGAAACGT",
    "ATGCCCGTT",
    "ATGGGGAAA",
])
config = Evo1SampleConfig(
    num_tokens=1000,
    temperature=1.0,
    batch_size=3,
)

result = run_evo1_sample(inputs, config)
for i, seq in enumerate(result.sequences):
    score = result.scores[i] if result.scores else "N/A"
    print(f"Sequence {i+1}: {len(seq)} bp, score={score}")
```

## Related Tools

**Tools often used together:**
- **`evo2-sample`**: Successor model for eukaryotic and longer-context generation
- **`prodigal`**: Find ORFs in generated DNA sequences
- **`minced-crispr`**: Detect CRISPR arrays in generated sequences
- **`local-blast`**: Search generated sequences against reference databases

**Alternative tools (similar function):**
- **`evo2-sample`**: Use for eukaryotic genomes or when longer context is needed

---
**Maintenance notes:**
- Last updated: 2025-06-01
