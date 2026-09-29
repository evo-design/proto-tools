<a href="https://bio-pro.mintlify.app/tools/masked-models/minerva"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Minerva

> [!NOTE]
> **License:** Minerva is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/garykbrixi/minerva/blob/main/LICENSE) for full terms.

## Overview

[Minerva](https://github.com/garykbrixi/minerva) is a genome language model initialized from gLM2 and trained on mixed protein/DNA sequences. Alongside embeddings, masked scoring, sampling, and gradients, this toolkit exposes the released base-pairing, protein, and repeat interaction heads as labeled token-by-token probability maps.

## Background

Minerva-MLM extends the mixed genomic representation used by gLM2: translated coding regions use uppercase amino acids, intergenic regions use lowercase nucleotides, and atomic strand markers retain orientation. The released 650M-parameter checkpoints support contexts of 4096 or 8192 tokens. The upstream [model card](https://huggingface.co/gbrixi/minerva-mlm) describes the model and its interaction heads; no paper citation is supplied there, so this toolkit cites the software repository.

The interaction heads turn internal model features into separate dense predictions for RNA base pairing, protein contacts, and repeats. These are learned interaction signals; a high matrix value alone does not establish a physical contact or a validated RNA secondary structure.

## Tools

### Minerva Embeddings (`minerva-embedding`)

Returns one mean-pooled embedding per prepared locus, with optional logits over the 24 canonical protein/DNA tokens and explicit token labels.

#### Applications

Use the pooled vector as a feature for clustering, retrieval, or a downstream supervised model. Keep the checkpoint and representation layer fixed when comparing embeddings.

#### Usage Tips

- **`repr_layer=-1` selects the last transformer output.** Layer 0 selects the input embedding table; positive indices select transformer outputs. Pooling includes every unpadded token, including strand markers.
- **`return_logits=True` adds an `(L, 24)` matrix.** Columns follow `ACDEFGHIKLMNPQRSTVWYacgt`; these are raw logits, not normalized probabilities. Both the matrix and `tokens` include strand-marker rows.
- **CSV, NPY, and PT exports contain pooled vectors.** JSON also preserves `tokens`, `vocab`, and optional `logits` for position-level analysis.

### Minerva Scoring (`minerva-score`)

Computes masked pseudo-log-likelihood by masking each canonical protein or DNA position in turn and predicting its original token from the remaining context. Returns summed and mean log-likelihood, perplexity, and the positions included in the score.

#### Applications

Rank related sequence variants with a contextual sequence prior. Scores reflect compatibility with the learned distribution and do not establish biological function or binding.

#### Usage Tips

- **`scored_positions` uses 1-indexed model-token positions.** Strand markers and accepted ambiguous protein symbols provide context but are excluded from score targets and the mean's denominator.
- **`batch_size` controls masked variants per forward pass.** Scoring requires work proportional to the number of target positions; start with the default 1 when memory is limited.
- **Optional scoring logits come from masked passes.** Their `(L, 24)` rows align with `tokens`; unscored marker and ambiguous-context rows are zero. These differ from the unmasked logits returned by embeddings.
- **PLL uses the full upstream vocabulary of 37 tokens.** Exported logits contain only the 24 biological tokens, so applying softmax to those columns does not reproduce PLL.
- **Compare similar contexts.** Higher mean log-likelihood and lower perplexity mean the model predicts the sequence more readily. Summed log-likelihood also depends on the number of scored positions.

### Minerva Sampling (`minerva-sample`)

Refills selected protein and DNA positions while preserving their original modality, sequence length, strand markers, and noneditable ambiguous protein symbols. Each input produces one sampled locus.

#### Applications

Propose local sequence edits for a design or variant-screening loop while keeping the prepared locus layout intact.

#### Usage Tips

- **Automatic masking uses `MaskingStrategy`.** By default it selects about 30% of eligible canonical biological positions, with at least one when any are eligible. Set either `num_mutations` or `mask_fraction`; `fixed_positions` counts all model tokens, including markers. Entropy and max-logit selection use the same checkpoint as sampling.
- **Premasked inputs require `mask_modalities`.** Supply one mapping per sequence that labels every `_` position as `"protein"` or `"dna"`. For example, `<+>M_<+>a_` needs `mask_modalities=[{3: "protein", 6: "dna"}]`. A marker counts once, and explicit masks bypass automatic selection for the batch.
- **The allowed alphabet is preserved per site.** Protein positions sample among the 20 canonical amino acids; DNA positions sample among lowercase `acgt`. A selected position can draw its original token again, so the number of masks is not a guaranteed number of substitutions.
- **`sampling_method` selects the fill algorithm.** `single_pass` fills all masks together. `iterative_refinement` progressively commits predictions and uses `num_steps`, `schedule`, `strategy`, `top_p`, and `temperature_annealing`. `temperature` controls sampling diversity and `seed` makes proposals reproducible.
- **Optional sampling logits describe the completed locus.** They come from a final unmasked forward pass. Export prepared mixed strings as JSON or text; they are not nucleotide or protein FASTA records.

### Minerva Gradient (`minerva-gradient`)

Returns the gradient of mean masked negative log-likelihood with respect to a relaxed `(L, 24)` sequence state. A fully specified `sequence` template fixes each position's modality and the locations of strand markers and ambiguous protein context.

#### Applications

Use the masked-language-model objective as a differentiable prior in sequence optimization while retaining the original protein/DNA layout.

#### Usage Tips

- **The matrix axis includes every model token.** Columns are `ACDEFGHIKLMNPQRSTVWYacgt`. Strand-marker and ambiguous-context rows must be zero; their returned gradients are also zero. Opposite-modality columns do not influence a position.
- **`temperature=1.0` treats the input as logits.** The worker applies a softmax within the position's modality. With `temperature=None`, provide a nonnegative probability distribution within that modality and zeros elsewhere. `one_hot_mixed_logits()` builds a valid starting state from a template.
- **The objective masks canonical sites one at a time.** The target at each site is its current discrete argmax token. `use_ste=True` uses hard forward tokens with soft derivatives; `compute_gradient=False` returns the objective with `gradient=None`.

### Minerva Interactions (`minerva-interactions`)

Returns the selected `base_pairing`, `protein`, and `repeat` heads as separate dense probability matrices. Every map has a `tokens` axis and an `(L, L)` `values` matrix, with one result bundle per input locus.

#### Applications

Inspect candidate protein contacts, RNA base-pairing signals, and repeat patterns in a prepared locus, or pass the labeled matrices to a downstream analysis.

#### Usage Tips

- **`heads` selects the returned channels.** All three are returned by default. Select fewer to reduce output size; this does not change the meaning of an individual channel.
- **`interaction_layers` chooses a released head variant.** The default 2 uses the last two transformer layers; 6 selects heads trained on the last six layers.
- **The axes include strand markers and ambiguous context.** Padding is removed. Use `tokens` to select biological subregions; matrix indices are not genome coordinates. The wrapper returns the upstream probabilities without symmetrizing, thresholding, removing the diagonal, or combining strands.
- **Dense output grows quadratically with token count.** Doubling the locus length quadruples each matrix's element count. The default export is compressed NPZ, with keys such as `0_protein` and `0_protein_tokens`; arrays load with `numpy.load(..., allow_pickle=False)`. JSON is also supported, and the public Python values remain nested lists.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Minerva tool in this toolkit.

- **Inputs are prepared, case-sensitive strings.** Uppercase letters denote protein; lowercase `acgt` denotes DNA. The uppercase protein symbols `X`, `B`, `U`, `Z`, and `O` are accepted as fixed context. Lowercase ambiguous nucleotides, whitespace, literal `<mask>`, and other control tokens are rejected. Sampling accepts `_` only with explicit modality metadata.
- **Strand markers are atomic tokens.** `<+>` and `<->` each occupy one token. Token positions are 1-indexed and refer to the prepared string, not genomic nucleotide coordinates. No case conversion, strand insertion, reverse complementation, translation, or gene calling is performed.
- **Prepare biologically appropriate orientation yourself.** Upstream examples use `<+>` for intergenic DNA and strand markers for translated coding regions. The wrapper preserves the submitted orientation and does not combine strands.
- **Model setup and weights are managed automatically.** The isolated environment is built on first use and public Hugging Face checkpoints are downloaded into the shared model cache. A Hugging Face token is not required for these public checkpoints.
- **Execution defaults to CUDA.** Reduce `batch_size` when memory is limited. Repeated calls with one checkpoint can reuse a persistent worker through the standard `ToolInstance.persist()` API.
- **The default checkpoint is `gbrixi/minerva-mlm`.** It accepts up to 4096 model tokens; `gbrixi/minerva-mlm-8k` accepts up to 8192. These counts include strand markers. Overlength inputs raise an error rather than truncating.
- **RNA base-pairing input uses the DNA alphabet.** Submit lowercase `acgt` as in upstream examples; the wrapper does not convert `u` to `t` or call an RNA structure from the map.
- **Scope is prepared-string inference.** Upstream gene calling, GenBank ingestion, fine-tuning, categorical Jacobians, and strand-ensemble workflows are not exposed by these tools.
