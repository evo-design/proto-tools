<a href="https://bio-pro.mintlify.app/tools/masked-models/ablang"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# AbLang

![AbLang](https://proto-bio.github.io/proto-assets/images/tool/ablang/hero.png)

> [!NOTE]
> **License:** AbLang is open source and free for academic and commercial use under a BSD-3-Clause license. Please refer to [the license](https://github.com/oxpig/AbLang2/blob/main/LICENSE) for full terms.

## Overview

[AbLang](https://github.com/oxpig/AbLang) is a family of antibody-specific masked language models from the [Oxford Protein Informatics Group (OPIG)](https://opig.stats.ox.ac.uk/). The models are trained on antibody variable-domain sequences from the Observed Antibody Space (OAS) and capture antibody-specific patterns including CDR variability, framework conservation, and heavy-light chain pairing. This toolkit exposes four tools that use the AbLang heavy-chain, light-chain, and paired heavy-plus-light models for embedding extraction, masked-position sampling, pseudo-log-likelihood scoring, and relaxed-sequence gradient computation.

## Background

AbLang ([Olsen, Moal, and Deane, 2022](https://doi.org/10.1093/bioadv/vbac046)) is a BERT-style masked language model trained exclusively on antibody variable-domain sequences from the OAS database. The published work demonstrates that AbLang restores residues missing from antibody sequence reads more accurately than germline-based imputation or the general-purpose ESM-1b protein language model, and runs approximately seven times faster than ESM-1b. Two single-chain checkpoints are provided, `ablang1-heavy` and `ablang1-light`, each with a 768-dimensional hidden representation.

AbLang-2 ([Olsen, Moal, and Deane, 2024](https://doi.org/10.1093/bioinformatics/btae618)) is trained on both unpaired and paired antibody sequence data and addresses a germline-residue bias observed in earlier antibody language models that overweighted germline positions during training. The published analysis shows that AbLang-2 suggests a diverse set of valid mutations with high cumulative probability and provides paired-chain context for antibody design. The `ablang2-paired` checkpoint exposed by this toolkit has a 480-dimensional hidden representation.

### Learning Resources

- [oxpig/AbLang](https://github.com/oxpig/AbLang) (OPIG, University of Oxford). Official AbLang repository, source code, and reference implementation of the heavy- and light-chain checkpoints.
- [oxpig/AbLang2](https://github.com/oxpig/AbLang2) (OPIG, University of Oxford). Official AbLang-2 repository for the paired heavy-plus-light checkpoint.
- [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OPIG). Public antibody sequence database used to train the AbLang models.

## Tools

### AbLang Embeddings (`ablang-embedding`)

Computes per-sequence AbLang embeddings for a list of `Antibody` inputs. Each `Antibody` carries an optional heavy chain and an optional light chain, and the tool routes to `ablang1-heavy`, `ablang1-light`, or `ablang2-paired` based on which chains are present. The output is a list of mean-pooled embeddings (768-dimensional for the single-chain checkpoints, 480-dimensional for the paired checkpoint) together with attention masks that mark valid sequence positions.

#### Applications

This tool is appropriate for any antibody-sequence analysis that benefits from a learned representation. Representative applications include clustering antibody repertoires by sequence similarity in embedding space, ranking humanization candidates by distance to a known humanised lead, identifying paired heavy-plus-light combinations with similar predicted binding behaviour, and providing input features to downstream classifiers for property prediction.

#### Usage Tips

- **Provide both chains when available to get the paired representation.** Setting both `heavy_chain` and `light_chain` on the `Antibody` input routes to `ablang2-paired`, which captures inter-chain co-evolutionary signals that the single-chain checkpoints cannot. Provide only one chain to use the corresponding single-chain model.
- **Use the returned attention mask when pooling or comparing positions.** Variable-length sequences in a batch are padded to the longest input, and the attention mask flags which positions are real (1) versus padding (0). Downstream per-position analyses should respect the mask.

### AbLang Sampling (`ablang-sample`)

Restores masked positions in antibody sequences using the AbLang masked-language-model head. Positions to be restored are marked with an underscore (`_`) in the input sequence, and the tool samples a replacement amino acid at each masked position from the model's predicted distribution. The sampling temperature is configurable, and greedy argmax decoding is selected by setting `temperature=0`.

#### Applications

This tool is appropriate for completing antibody sequences with missing residues, a common need when working with B-cell receptor sequencing reads that drop the first several N-terminal residues. Representative applications include filling sequencing-dropout positions before downstream structural prediction, exploring single-position substitutions in CDR or framework regions, and generating antibody-context-aware variants for humanisation or affinity-maturation campaigns.

#### Usage Tips

- **Use the underscore (`_`) as the mask character.** Other placeholders such as `*`, `X`, or `<mask>` are not recognised. Each underscore in the input sequence is replaced with a sample drawn from the model distribution at that position.
- **`temperature` controls the sampling stochasticity.** The default of `1.0` samples from the unscaled model distribution, producing different sequences across repeated calls. Set `temperature=0` for greedy argmax decoding, which matches AbLang's native `restore` mode and produces deterministic output. Lower positive values sharpen toward the top prediction, higher values flatten toward uniform. Use `seed` to make stochastic runs reproducible.
- **Set `align=True` to extend unknown-length termini.** When the input sequence is shorter than expected, enabling ANARCI-based alignment lets AbLang restore residues at the N or C terminus as well as in the middle of the sequence. Setting `align=True` forces greedy decoding regardless of the `temperature` setting, since the ANARCI alignment is incompatible with stochastic sampling.
- **Set `return_logits=True` to recover the per-position amino-acid distribution.** When enabled, the output carries a per-position logit matrix of shape `(num_sequences, seq_len, 20)` alongside the sampled sequence, which is useful for downstream re-ranking or post-hoc analysis. The default omits the logits to keep the response small.

### AbLang Scoring (`ablang-score`)

Computes per-sequence scores under the AbLang masked-language-model head. The `scoring_mode` configuration field selects between pseudo-log-likelihood (`"pseudo_log_likelihood"`) and confidence (`"confidence"`) scoring.

#### Applications

This tool is appropriate for ranking antibody sequences by how "natural" they look under the model. Representative applications include selecting humanisation candidates closer to natural human antibody repertoires, flagging candidate sequences with low predicted naturalness for redesign, and ranking ProteinMPNN- or design-pipeline-generated sequences by pseudo-log-likelihood before more expensive downstream analyses.

#### Usage Tips

- **Pseudo-log-likelihood scores from different checkpoints sit on different scales and are not directly comparable.** Each of `ablang1-heavy`, `ablang1-light`, and `ablang2-paired` was trained independently and produces scores on its own scale, so heavy-chain scores cannot be compared against light-chain scores and single-chain scores cannot be compared against paired-chain scores. Only compare antibodies that were scored with the same model variant.
- **Higher pseudo-log-likelihood corresponds to a more probable sequence under AbLang.** Use scores comparatively across variants of the same antibody rather than as an absolute developability or affinity score. A high score reflects sequence likeness to the training distribution, not predicted experimental performance.

### AbLang Gradient (`ablang-gradient`)

Computes the gradient of the AbLang masked pseudo-log-likelihood objective with respect to a relaxed antibody-logit input. The tool accepts an `AntibodyLogits` object whose `heavy_chain` and `light_chain` fields are per-position logit or probability matrices, masks each amino-acid position in turn, scores the bidirectional-context prediction with cross-entropy against the input distribution, and returns the gradient matrix together with the loss value and auxiliary metrics.

#### Applications

This tool is appropriate for differentiable antibody-design pipelines that update a continuous sequence representation by gradient descent. Representative applications include relaxed-logit hallucination for antibody design, joint optimisation of AbLang likelihood together with structure-based losses such as AlphaFold2 hallucination, and incorporating an antibody-specific naturalness term into broader binder-design objectives.

#### Usage Tips

- **Input logits use the canonical protein order `ACDEFGHIKLMNPQRSTVWY`.** The tool implementation internally maps to AbLang's vocabulary order before the forward pass and returns the gradient in the same canonical order, so the user does not need to handle the AbLang-specific token order separately.
- **Set `temperature` to apply a softmax before scoring.** When `temperature` is set, the tool implementation applies `softmax(input / temperature)` to the input logits before the forward pass. Leave `temperature=None` (the default) when the user already provides a normalised probability distribution.
- **Use the Straight-Through Estimator option for discrete-token gradients.** Setting `use_ste=True` substitutes hard one-hot tokens in the forward pass while allowing gradients to flow through the soft probabilities, which can produce sharper update directions for some discrete-design loops. The default (`use_ste=False`) uses soft blended embeddings.
- **Set `compute_gradient=False` for forward-only scoring.** This skips the backward pass and returns `gradient=None` together with the loss value, which is useful for ranking candidates from a Monte Carlo proposal without paying the backward-pass cost.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every AbLang tool in this toolkit (`ablang-embedding`, `ablang-gradient`, `ablang-sample`, `ablang-score`).

- **All four tools route automatically among the three AbLang checkpoints based on the chains provided.** Providing only a heavy chain selects `ablang1-heavy`, providing only a light chain selects `ablang1-light`, and providing both selects the paired `ablang2-paired` checkpoint. At least one chain must be set on each input.
- **Every antibody in a batched call must use the same chain configuration.** The embedding, scoring, and sampling tools accept a list of antibodies in a single call, and every antibody in that list must provide the same combination of heavy and light chains so that all entries route to the same checkpoint. Mixed lists are rejected at input construction with a clear error.
- **AbLang is appropriate for antibody variable-domain sequences only.** Non-antibody proteins should be analysed with a general-purpose protein language model such as ESM2 rather than AbLang, which was trained exclusively on antibody sequences and produces unreliable scores or embeddings outside that distribution.
