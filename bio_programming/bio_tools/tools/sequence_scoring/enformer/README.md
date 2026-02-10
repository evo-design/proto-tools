# Enformer

## Overview
Enformer predicts gene expression and regulatory activity directly from a fixed-length DNA sequence.

- Model context: `196,608` bp
- Output bins: `896`
- Species heads: `human`, `mouse`
- Tool key: `enformer-prediction`

## Module Structure
The Enformer tool now follows the same split used by newer tool families.

- `enformer_prediction.py`: input/config/output models, tool implementation, helpers
- `standalone/inference.py`: standalone inference entrypoint used by `EnvManager("enformer")`
- `examples/example.ipynb`: end-to-end usage notebook
- `examples/example_output/`: example export target directory

## Python API
```python
from bio_programming.bio_tools.tools.sequence_scoring.enformer import (
    EnformerInput,
    EnformerConfig,
    run_enformer,
)
```

### Input
| Field | Type | Description |
|---|---|---|
| `sequence` | `str` | DNA sequence, exactly `196,608` bp |

### Config
| Field | Type | Default | Description |
|---|---|---|---|
| `output_tracks` | `List[int]` | required | Track indices to extract |
| `species` | `"human" \| "mouse"` | `"human"` | Species output head |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Verbose logging |

### Output
| Field | Type | Description |
|---|---|---|
| `sequence` | `str` | Input sequence |
| `sequence_length` | `int` | Always `196,608` |
| `prediction` | `List[List[float]]` | Shape `[896, num_tracks]` |
| `output_tracks` | `List[int]` | Extracted track indices |
| `species` | `str` | Species used |

## Quick Start
```python
from bio_programming.bio_tools.tools.sequence_scoring.enformer import (
    EnformerInput,
    EnformerConfig,
    run_enformer,
)

sequence = "ATCG" * 49152  # 196,608 bp

inputs = EnformerInput(sequence=sequence)
config = EnformerConfig(
    output_tracks=[0, 1, 2],
    species="human",
)

result = run_enformer(inputs, config)
print(len(result.prediction), len(result.prediction[0]))  # 896 x 3
```

## Example Notebook
Use:

- `bio_programming/bio_tools/tools/sequence_scoring/enformer/examples/example.ipynb`

The notebook includes:

1. A basic Enformer prediction run
2. A compact variant-effect style comparison workflow
3. Export to `examples/example_output/`

## Notes
- Enformer requires exact sequence length (`196,608`).
- Put your region of interest near the center of the window.
- GPU inference is strongly recommended for throughput.

## References
- Paper: https://doi.org/10.1038/s41592-021-01252-x
- Repository: https://github.com/google-deepmind/deepmind-research/tree/master/enformer

## Related Tools
- `borzoi`
- `alphagenome`
- `splice_transformer`
