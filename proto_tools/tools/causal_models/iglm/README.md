<a href="https://bio-pro.mintlify.app/tools/causal-models/iglm"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# IgLM

![IgLM](https://proto-bio.github.io/proto-assets/images/tool/iglm/hero.png)

> [!NOTE]
> **License:** IgLM is licensed under Custom (JHU Academic Software License Agreement) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://github.com/Graylab/IgLM/blob/main/LICENSE.md) for full terms.

## Overview

[IgLM](https://github.com/Graylab/IgLM) is a generative language model developed by the Gray Lab at Johns Hopkins University for antibody sequence design. Trained on 558 million antibody variable-domain sequences, IgLM can generate complete antibody sequences, redesign selected regions, and evaluate sequence likelihood. Generation is conditioned on antibody chain type and species of origin, allowing the model to produce sequences with different repertoire characteristics.

## Background

IgLM ([Shuai et al., 2023](https://doi.org/10.1016/j.cels.2023.10.001)) is a transformer-based language model trained on antibody sequences from the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OAS) database. Its architecture is based on [GPT-2](https://en.wikipedia.org/wiki/GPT-2), with two released checkpoints: `IgLM` (13 million parameters) and `IgLM-S` (1.4 million parameters).

The model's principal innovation is its sequence-infilling objective. During training, a randomly selected span of 10-20 amino acids is masked and appended after the remaining sequence. This allows an autoregressive model to generate replacement residues using information from both sides of the masked region.

This approach is particularly useful for antibody design, where the [complementarity-determining regions](https://en.wikipedia.org/wiki/Complementarity-determining_region) (CDRs) can be modified while retaining the surrounding framework. The authors demonstrated this by generating CDR H3 libraries for 49 therapeutic antibodies. Compared with their parent sequences, the generated libraries showed improvements in computational estimates of solubility, aggregation propensity, and sequence similarity to human antibody repertoires. These results support IgLM's use for antibody library design, although experimental characterization is still required to establish binding, stability, and developability.

IgLM also assigns likelihoods to antibody sequences, providing a quantitative measure of how well they conform to the sequence distributions learned during training. The authors demonstrated that these scores can distinguish human from non-human antibodies, achieving a reported ROC AUC of 0.96 on their evaluation dataset.

### Learning Resources

- [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) - Database of antibody sequences used to train IgLM.
- [OAS95 dataset](https://data.graylab.jhu.edu/OAS95.tar.gz) - Sequence dataset and training/test splits associated with the original publication.

## Tools

### IgLM Sequence Generation (`iglm-sample`)

Generates antibody variable-domain sequences conditioned on chain type and species of origin. An optional N-terminal sequence can be provided as a starting prompt. Each input produces `num_sequences_per_input` distinct sequences.

#### Applications

Use `iglm-sample` to generate synthetic antibody sequence libraries or extend an existing N-terminal sequence into a complete variable domain. The original publication demonstrates the generation of human heavy chains beginning with `EVQ` and camelid heavy-chain sequences beginning with `QVQ`.

Generated sequences can be evaluated with `iglm-score` or other methods for assessing structural properties and developability.

#### Usage Tips

- **Chain and species conditioning:** `chains` accepts `heavy` or `light`, while `species` supports `human`, `mouse`, `rat`, `rabbit`, `rhesus`, and `camel`. A single value can be applied to an entire batch, or separate values can be provided for individual sequences.
- **Sequence diversity:** `temperature` and `top_p` control sampling diversity. Lower values favor more probable residues, while higher values allow a broader range of sequences. The paper explores temperatures of 0.8-1.2.
- **Number of sequences:** `num_sequences_per_input` specifies the number of distinct sequences requested for each input. If the implementation cannot generate enough unique, valid sequences within its sampling budget, it raises an error rather than returning an incomplete collection.
- **Maximum length:** `max_length` sets the total token budget, including conditioning tokens and the input prompt. The default is 150, and both released models support a context length of up to 512 tokens. Sequences that fail to terminate within the available budget are discarded.
- **Prompt handling:** By default, `prepend_prompt=True` includes the original prompt in the returned sequence. Set it to `False` to return only the generated continuation.

### IgLM Span Redesign (`iglm-infill`)

Replaces a specified region of an antibody sequence with newly generated residues while preserving the surrounding sequence. The model uses both flanking regions, together with chain and species conditioning, to generate a replacement of variable length.

Each result includes the redesigned full sequence, the newly generated residues, and their positions within the resulting sequence.

#### Applications

The primary application is diversification of antibody CDR loops while retaining the parent framework. This follows the approach used in the original study, where IgLM generated alternative CDR H3 sequences for therapeutic antibodies.

The resulting libraries can be evaluated using IgLM likelihood scores, developability predictors, and downstream structural or experimental assays.

#### Usage Tips

- **Span coordinates:** Residue positions are 1-indexed and inclusive. For example, `IgLMSpan(start=99, end=106)` selects eight residues. The implementation converts these coordinates to the 0-indexed, end-exclusive format used by upstream IgLM.
- **Variable-length generation:** IgLM determines the length of each replacement span, which may differ from the original. The generated residues are available through `infilled_residues`, and their updated positions are recorded in `spans`.
- **Scoring generated regions:** The returned `spans` can be passed directly to `iglm-score` to calculate the conditional likelihood of each redesigned region without manually adjusting coordinates.
- **Span length:** The model was trained with masked regions of 10-20 residues. It can generate replacements of other lengths, but performance outside the training distribution is less well characterized.
- **Maximum length:** `max_length` must accommodate the masked parent sequence, its conditioning tokens, and the generated replacement. Increase the default of 150 when working with longer sequences or expecting longer replacements.

### IgLM Scoring (`iglm-score`)

Calculates the likelihood of antibody sequences under IgLM, conditioned on chain type and species of origin. The tool returns total log-likelihood, average log-likelihood, and perplexity, with optional per-position logits.

When a span is provided, the model instead evaluates the conditional likelihood of that region given the remainder of the sequence.

#### Applications

Whole-sequence likelihood can be used to assess how closely a sequence resembles the antibody repertoire represented in IgLM's training data. This is useful for comparing candidate antibodies or studying differences between antibody sequence distributions.

Conditional likelihood is particularly useful for evaluating regions generated by `iglm-infill`, allowing candidate CDR loops to be compared in the context of their surrounding framework.

These scores reflect the model's learned sequence distribution and should not be interpreted as direct predictions of binding affinity, stability, or experimental performance.

#### Usage Tips

- **Span coordinates:** Spans use the same 1-indexed, inclusive convention as `iglm-infill`. Returned infilling spans can be supplied directly to the scoring tool.
- **Whole-sequence and conditional scoring:** Omit `spans` to evaluate complete sequences. Providing a span scores that region using the remaining sequence as context.
- **Comparing scores:** `log_likelihood` is summed over the scored tokens and depends on sequence length. Use `avg_log_likelihood` or `perplexity` when comparing sequences of different lengths. Lower perplexity indicates greater likelihood under the model.
- **Conditioning consistency:** Compare scores obtained using the same checkpoint, chain type, species, and scoring procedure. Scores from different conditioning settings should not be treated as directly equivalent.
- **Scoring convention:** Following the upstream implementation, likelihood calculations include the terminal token. `avg_log_likelihood` is the mean per-token log-likelihood, and `perplexity` is calculated as `exp(-avg_log_likelihood)`.
- **Per-position logits:** Set `return_logits=True` to obtain token logits for each scored position. These are returned with the vocabulary needed to identify the corresponding amino acid and special tokens.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

The Proto guides for tool persistence, device management, parallel execution, and cloud inference apply to all three IgLM tools.

- **Model checkpoints:** Two pretrained checkpoints are available: `IgLM` (13M parameters) and `IgLM-S` (1.4M parameters). Both support the same conditioning vocabulary. The larger model is used for the principal evaluations in the original publication.
- **Model weights:** Both checkpoints are distributed with the upstream Python package, so no separate model download or authentication is required.
- **Licensing:** IgLM is distributed under the JHU Academic Software License Agreement. Use is restricted to eligible non-commercial, non-profit purposes unless a separate commercial license is obtained.
- **Supported sequences:** IgLM was trained on antibody heavy- and light-chain variable domains. Constant regions, linked antibody constructs, and unrelated proteins fall outside its intended training distribution.
- **Conditioning:** The model supports two chain types and six species of origin. These categories are fixed by the pretrained vocabulary and cannot be extended through inference settings alone.
- **Batch processing:** `batch_size` controls the number of sequences processed together. Larger batches can improve generation throughput, subject to available device memory.
- **Sampling:** Generation is stochastic, and multiple candidates can be produced for each input. Sampling temperature and nucleus probability provide control over sequence diversity.
- **Output organization:** Generated sequences are grouped by input, preserving the association between each prompt or parent antibody and its candidate designs.
- **Evaluation:** IgLM likelihood provides a measure of sequence plausibility under its training distribution, not a calibrated estimate of antibody function or developability. Additional computational and experimental evaluation is recommended.
