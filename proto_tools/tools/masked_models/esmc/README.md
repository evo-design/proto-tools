<a href="https://bio-pro.mintlify.app/tools/masked-models/esmc"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESM C (Cambrian)

## Overview
ESM C ("Cambrian") is EvolutionaryScale's [embedding-focused protein language model](https://www.evolutionaryscale.ai/blog/esm-cambrian). It ships in the same `esm` Python package as ESM3 and exposes only an embedding/logits interface — there is no sample/score interface. Two open-weights variants are wrapped here: `esmc_300m` (open commercial license) and `esmc_600m` (non-commercial only).

## Background

ESM C is trained with the masked language modeling (MLM) objective: positions are randomly masked during training and the model learns to predict the original amino acid. The model exposes per-position logits over the 20 amino acids alongside its embeddings.

Unlike ESM3, ESM C drops the structure-token track and generative head; only the embedding/logits interface is provided.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esmc-embedding` | Extract mean-pooled embeddings (and optional per-position logits) | Embeddings, attention masks, optional logits |

## Model Variants

| Checkpoint | Hidden dim | License | Default |
|------------|------------|---------|---------|
| `esmc_300m` | 960  | [Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement) — commercial OK | Yes |
| `esmc_600m` | 1152 | [Cambrian Non-Commercial License](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement) — research/internal only | No |

The 6B-parameter ESM C variant is API-only (Forge) and is not exposed by this wrapper.

## Execution Modes

- **Local GPU/CPU**: Loads the model on-demand. Use `device="cuda"`, `"cpu"`, or `"mps"`.
- **Shared env**: ESM C and ESM3 share a single on-disk micromamba env (`evolutionaryscale_esm`). Installing one tool installs the env for both.

## How It Works

ESM C uses a transformer encoder trained with the MLM objective on a large protein-sequence corpus. For each sequence:

1. Tokenize with the ESM tokenizer (BOS/EOS added).
2. Run a single forward pass to produce per-position hidden states.
3. Strip BOS/EOS, mean-pool over the attention mask to yield a fixed-length embedding.
4. (Optional) Return per-position logits restricted to the 20 standard amino acids.

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences (amino acid strings) |

## Configuration

### `ESMCEmbeddingsConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `Literal["esmc_300m", "esmc_600m"]` | `esmc_300m` | Model variant |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `cuda` | `cuda`, `cpu`, or `mps` |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits over the 20 amino acids |

## Output Specification

### `ESMCEmbeddingsOutput`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[SequenceEmbedding]` | Per-sequence embedding results |

**`SequenceEmbedding` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mean_embedding` | `List[float]` | Mean-pooled embedding (length 960 for 300M, 1152 for 600M) |
| `attention_mask` | `List[int]` | Binary mask: 1 = valid residue, 0 = padding |
| `logits` | `Optional[List[List[float]]]` | Per-position logits `(seq_len, 20)`. Only present if `return_logits=True` |

## Quick Start Examples

**Example 1: Extract embeddings (300M, default)**
```python
from proto_tools.tools.masked_models.esmc import (
    ESMCEmbeddingsInput, ESMCEmbeddingsConfig, run_esmc_embeddings,
)

inputs = ESMCEmbeddingsInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESMCEmbeddingsConfig()  # defaults to esmc_300m

result = run_esmc_embeddings(inputs, config)
print(f"Processed {len(result.results)} sequences")
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")  # 960
```

**Example 2: Use the 600M variant with logits**
```python
inputs = ESMCEmbeddingsInput(sequences=["MVLSPADKTNVKAAW"])
config = ESMCEmbeddingsConfig(model_checkpoint="esmc_600m", return_logits=True)

result = run_esmc_embeddings(inputs, config)
emb = result.results[0]
print(f"Embedding dim: {len(emb.mean_embedding)}")              # 1152
print(f"Logits shape: ({len(emb.logits)}, {len(emb.logits[0])})")  # (seq_len, 20)
```

**Example 3: Batch processing**
```python
sequences = ["MVLSPADKTNVKAAW"] * 100

inputs = ESMCEmbeddingsInput(sequences=sequences)
config = ESMCEmbeddingsConfig(batch_size=16, verbose=True)

result = run_esmc_embeddings(inputs, config)
print(f"Processed {len(result.results)} sequences")
```

## Best Practices & Gotchas

- Default `return_logits=False`. Logits are large (`seq_len × 20` floats per sequence); only enable when you need them.
- Pick `esmc_300m` for any commercial workload — `esmc_600m` is non-commercial only.
- For multi-GPU or large-batch workloads, prefer `ToolPool`; for repeated single-batch calls, use `ToolInstance.persist_tool("esmc")` to keep the model warm.
- The on-disk env is shared with ESM3 — installing either installs both.

## References

**Primary publication:**
- EvolutionaryScale Team (2024). *ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning.* Blog post: https://www.evolutionaryscale.ai/blog/esm-cambrian

**Implementation:**
- GitHub: [https://github.com/evolutionaryscale/esm](https://github.com/evolutionaryscale/esm)
- HuggingFace: [https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12)
- EvolutionaryScale: [https://www.evolutionaryscale.ai/](https://www.evolutionaryscale.ai/)

## Related Tools

**Tools often used together:**
- `esm3`: Same family; adds generative and scoring interfaces. Shares the `evolutionaryscale_esm` env on disk.
- `inverse_folding/proteinmpnn`: Structure-conditioned sequence design.

**Alternative tools:**
- `esm2`: Earlier embeddings-only ESM family.
- `ablang`: Antibody-specialized embedding model.
