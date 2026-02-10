# AlphaGenome

## Overview
AlphaGenome predicts diverse genomic signals from long genomic context windows using open-source model weights.

This implementation uses **local inference with Hugging Face weights** through an isolated standalone venv runtime.

- Tool keys:
  - `alphagenome-predict-interval`
  - `alphagenome-predict-variant`
  - `alphagenome-predict-sequence`
  - `alphagenome-score-variant`
  - `alphagenome-score-interval`
  - `alphagenome-score-ism-variants`
- Runtime path: `standalone/inference.py`
- Execution model: `EnvManager("alphagenome")`

## Module Structure
This tool follows the split pattern used by modernized tool families.

- `shared_data_models.py`: shared base types (`AlphaGenomeInput`, `AlphaGenomePredictOutput`, `AlphaGenomePredictConfig`, `AlphaGenomeScoreOutput`)
- `alphagenome_predict_interval.py`: interval prediction tool (`AlphaGenomePredictIntervalInput/Config/Output`)
- `alphagenome_predict_variant.py`: variant-effect prediction tool (`AlphaGenomePredictVariantInput/Config/Output`)
- `alphagenome_predict_sequence.py`: raw sequence prediction tool (`AlphaGenomePredictSequenceInput/Config/Output`)
- `alphagenome_score_variant.py`: variant scoring tool (`AlphaGenomeScoreVariantInput/Config/Output`)
- `alphagenome_score_interval.py`: interval scoring tool (`AlphaGenomeScoreIntervalInput/Config/Output`)
- `alphagenome_score_ism_variants.py`: in-silico mutagenesis tool (`AlphaGenomeScoreISMInput/Config/Output`)
- `standalone/inference.py`: isolated venv execution path
- `examples/example.ipynb`: end-to-end usage notebook

## Python API
```python
from bio_programming.bio_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictIntervalInput,
    AlphaGenomePredictVariantInput,
    AlphaGenomePredictSequenceInput,
    AlphaGenomeScoreVariantInput,
    AlphaGenomeScoreISMInput,
    AlphaGenomeScoreIntervalInput,
    AlphaGenomePredictIntervalConfig,
    AlphaGenomePredictVariantConfig,
    AlphaGenomePredictSequenceConfig,
    AlphaGenomeScoreVariantConfig,
    AlphaGenomeScoreIntervalConfig,
    AlphaGenomeScoreISMConfig,
    run_alphagenome_predict_interval,
    run_alphagenome_predict_variant,
    run_alphagenome_predict_sequence,
    run_alphagenome_score_variant,
    run_alphagenome_score_interval,
    run_alphagenome_score_ism_variants,
)
```

## Inputs
### Interval (`AlphaGenomePredictIntervalInput`)
| Field | Type | Description |
|---|---|---|
| `chromosome` | `str` | Chromosome identifier (e.g. `"chr1"`) |
| `interval_start` | `int` | Interval start (0-based, inclusive) |
| `interval_end` | `int` | Interval end (0-based, exclusive) |

### Variant (`AlphaGenomePredictVariantInput` / `AlphaGenomeScoreVariantInput`)
Extends `AlphaGenomeInput` with:

| Field | Type | Description |
|---|---|---|
| `variant_position` | `int` | Variant position (0-based) |
| `reference_bases` | `str` | Reference allele (A/C/G/T/N string) |
| `alternate_bases` | `str` | Alternate allele (A/C/G/T/N string) |

### Sequence (`AlphaGenomePredictSequenceInput`)
| Field | Type | Description |
|---|---|---|
| `sequence` | `str` | Raw DNA sequence string |

### ISM (`AlphaGenomeScoreISMInput`)
Extends `AlphaGenomeInput` with:

| Field | Type | Description |
|---|---|---|
| `ism_interval_start` | `int` | ISM sub-interval start (0-based, inclusive) |
| `ism_interval_end` | `int` | ISM sub-interval end (0-based, exclusive) |
| `variant_position` | `Optional[int]` | Optional existing variant position (0-based) |
| `reference_bases` | `Optional[str]` | Optional existing variant ref allele |
| `alternate_bases` | `Optional[str]` | Optional existing variant alt allele |

## Configs

### `AlphaGenomePredictIntervalConfig` / `AlphaGenomePredictVariantConfig` / `AlphaGenomePredictSequenceConfig` (prediction tools)
| Field | Type | Default | Description |
|---|---|---|---|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `requested_outputs` | `List[str]` | required | Output type names |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `ontology_terms` | `Optional[List[str]]` | `None` | Optional ontology filters |
| `device` | `str` | `"cuda"` | Device for standalone inference |

### `AlphaGenomeScoreVariantConfig` / `AlphaGenomeScoreISMConfig` (variant & ISM scoring)
| Field | Type | Default | Description |
|---|---|---|---|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `variant_scorers` | `Optional[List[str]]` | `None` | Scorer names (`None` = all recommended) |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `device` | `str` | `"cuda"` | Device for standalone inference |

### `AlphaGenomeScoreIntervalConfig` (interval scoring)
| Field | Type | Default | Description |
|---|---|---|---|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `interval_scorers` | `Optional[List[str]]` | `None` | Scorer names (`None` = all recommended) |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `device` | `str` | `"cuda"` | Device for standalone inference |

## Outputs

### `AlphaGenomePredictIntervalOutput` / `AlphaGenomePredictVariantOutput` / `AlphaGenomePredictSequenceOutput` (prediction tools)
| Field | Type | Description |
|---|---|---|
| `chromosome` | `str` | Chromosome identifier |
| `interval_start` | `int` | Interval start (0-based) |
| `interval_end` | `int` | Interval end (0-based, exclusive) |
| `requested_outputs` | `List[str]` | Requested output type names |
| `result` | `Dict[str, Any]` | Serialized prediction payload |
| `variant` | `Optional[Dict[str, Any]]` | Variant metadata (variant tool only) |

Supported export formats: `json`, `npy`.

### `AlphaGenomeScoreVariantOutput` / `AlphaGenomeScoreIntervalOutput` / `AlphaGenomeScoreISMOutput` (scoring tools)
| Field | Type | Description |
|---|---|---|
| `scores` | `List[Dict[str, Any]]` | Tidy score records (one per scorer-track-gene combination) |

Supported export formats: `json`, `csv`.

## Quick Start
```python
from bio_programming.bio_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictIntervalConfig,
    AlphaGenomePredictIntervalInput,
    run_alphagenome_predict_interval,
)

inputs = AlphaGenomePredictIntervalInput(
    chromosome="chr19",
    interval_start=10_587_331,
    interval_end=11_635_907,
)

config = AlphaGenomePredictIntervalConfig(
    requested_outputs=["RNA_SEQ", "ATAC"],
    organism="human",
)

result = run_alphagenome_predict_interval(inputs, config)
print(result.requested_outputs)
```

## Notes
- This implementation does **not** use API keys.
- Inference runs in an isolated standalone environment under `.venvs/alphagenome_env`.
- First run may download model weights and can take significantly longer.

## References
- Repository: https://github.com/google-deepmind/alphagenome_research

## Related Tools
- `enformer`
- `borzoi`
- `splice_transformer`
