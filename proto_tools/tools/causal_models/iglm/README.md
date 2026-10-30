<a href="https://bio-pro.mintlify.app/tools/causal-models/iglm"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# IgLM

> [!NOTE]
> **License:** IgLM is licensed under Custom (JHU Academic Software License Agreement) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://github.com/Graylab/IgLM/blob/main/LICENSE.md) for full terms.

## Overview

[IgLM](https://github.com/Graylab/IgLM) is an antibody-specific generative language model from the Gray Lab at Johns Hopkins University, trained on 558 million antibody variable-domain sequences from the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/). Its infilling objective lets it re-design a variable-length span inside an antibody using the residues on both sides as context, and every generation is conditioned on chain type and species of origin. This toolkit exposes full-length generation, span re-design, and likelihood scoring.

## Background

IgLM ([Shuai et al., 2023](https://doi.org/10.1016/j.cels.2023.10.001)) is a [GPT-2](https://en.wikipedia.org/wiki/GPT-2) decoder trained on 558 million antibody heavy- and light-chain variable sequences: the 809 million unique sequences then available in the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OAS) were clustered at 95% sequence identity down to 588 million non-redundant sequences, a random 5% of which was held out for testing, leaving 558 million for training. Every training sequence is prefixed with two conditioning tags naming its chain type (heavy or light) and its species of origin (human, mouse, rat, rabbit, rhesus, or camel), so the same checkpoint generates and scores across chains and species. Two checkpoints are released: `IgLM`, the model used throughout the paper, with 4 layers, 512-dimensional embeddings, 8 attention heads, and 12,889,600 parameters; and `IgLM-S`, a smaller variant with 3 layers, 192-dimensional embeddings, 6 attention heads, and 1,439,616 parameters. Both released checkpoints share a 33-token vocabulary (the 20 amino acids, the two chain tags, the six species tags, and five special tokens) and a 512-token context window.

What distinguishes IgLM from a left-to-right protein language model is its training objective, which the paper formulates as text infilling. For each training sequence a span of length drawn uniformly from 10 to 20 residues is replaced in place by a mask token and then re-appended after a separator token, so the model learns to emit the span conditioned on the residues that both precede *and* follow it while remaining an ordinary autoregressive decoder. That matters because the [complementarity-determining regions](https://en.wikipedia.org/wiki/Complementarity-determining_region) that dominate antigen binding sit in the middle of the variable domain, between conserved framework regions: re-designing one means generating into a gap with known flanks, not continuing a prefix. On a 30-million-sequence held-out test set, IgLM with bidirectional context achieved lower infilling perplexity on CDR loops than the same model restricted to preceding context and than ProGen2-base or ProGen2-OAS, which the authors read as evidence that aligning the pre-training objective with the design task is what buys the improvement.

The paper's headline application is CDR loop library design. Infilling the CDR H3 loop of each of 49 therapeutic antibodies across sampling temperatures of 0.8, 1.0, and 1.2 and nucleus probabilities of 0.5, 0.75, and 1.0 produced libraries whose members showed reduced predicted aggregation propensity, increased predicted solubility (CamSol Intrinsic), and higher OASis identity - the fraction of a sequence's 9-mers seen in human repertoires - than their parent antibodies. The same model likelihoods also carry signal on their own: used to classify sequences from the IMGT mAb DB as human versus non-human, IgLM reached a ROC AUC of 0.96, ahead of ProGen2-OAS at 0.94, which is the basis for treating IgLM likelihood as a naturalness score for antibody sequences.

### Learning Resources

- [Graylab/IgLM](https://github.com/Graylab/IgLM) (Gray Lab, Johns Hopkins University) - the official repository, with command-line and Python examples for generation, infilling, and likelihood evaluation that map one-to-one onto the three tools here.
- [IgLM in Google Colab](https://colab.research.google.com/github/Graylab/IgLM/blob/main/IgLM.ipynb) (Gray Lab) - a hosted notebook for trying antibody sequence generation without a local install.
- [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OPIG, University of Oxford) - the antibody repertoire database IgLM was trained on, and the origin of its chain and species conditioning vocabulary.
- [OAS95 dataset](https://data.graylab.jhu.edu/OAS95.tar.gz) (Gray Lab) - the exact training and test split used in the paper, clustered at 95% sequence identity.

## Tools

### IgLM Sequence Generation (`iglm-sample`)

Generates full-length antibody variable sequences autoregressively from the chain and species conditioning tokens plus an optional N-terminal prompt, returning `config.num_sequences_per_input` distinct sequences for each prompt, bundled per input item in `results`.

#### Applications

Use this to build synthetic antibody or nanobody libraries from scratch, or to extend a fixed N-terminal motif into complete variable domains. Upstream's own worked examples are the two canonical cases: human heavy chains grown from the prompt `EVQ`, and nanobodies grown from `QVQ` under camel conditioning. Generated sequences are candidates, so pair them with structure prediction, developability scoring, or `iglm-score` before drawing conclusions.

#### Usage Tips

- **`chains` and `species` are per-item and broadcastable, and are named in plain words.** `chains` accepts `"heavy"` or `"light"`; `species` accepts `"human"`, `"mouse"`, `"rat"`, `"rabbit"`, `"rhesus"`, or `"camel"`. Both are case-insensitive, so `"HUMAN"` and `"Heavy"` are accepted and normalized to lower case. Pass a single value to condition every item the same way, or one value per item to mix chains and species in one call. Camel conditioning is only meaningful with `chains=["heavy"]` - OAS's camel repertoires are heavy-chain only, which is why the paper sampled camel heavy chains alone.
- **An over-ambitious `num_sequences_per_input` can exhaust the attempt budget.** The tool resamples until it holds that many *distinct* sequences per item, discarding malformed generations the way upstream does. On a sharply peaked distribution (low `temperature`, low `top_p`) the distinct sequences run out and the tool raises rather than returning a short list. The default is `1`; upstream's own default is 1000, so raise it deliberately and keep `temperature` near the paper's 0.8-1.2 range when asking for large counts.
- **`max_length` (default 150, upstream's cap) counts the two conditioning tokens and the prompt.** It is a token budget for the whole generation, not a count of new residues, so a long prompt eats into the sequence that follows. Both checkpoints accept up to 512.
- **Running out of `max_length` shows up as an error, not as short sequences.** IgLM's format spends five tokens on the two conditioning tags, the mask, the separator, and the terminator, and a candidate that reaches the budget before completing that format fails upstream's well-formedness check and is discarded, spending a draw from the attempt budget. So a budget that is too tight surfaces as `yielded only N of M requested unique sequences`, and the fix is to raise `max_length` (up to 512) rather than to filter the output. Short sequences that *are* returned are IgLM's own choice of stopping point, not a truncation.
- **`prepend_prompt=True` (the default) returns the prompt joined to its continuation.** Set it `False` to receive only the newly generated residues.

### IgLM Span Re-design (`iglm-infill`)

Replaces the residues of `spans[i]` in `sequences[i]` with a mask token and has IgLM write a new segment conditioned on the flanking residues plus the chain and species tokens - the infilling objective the model was trained on. Each bundle in `results` holds the full re-designed sequences and, aligned with them, the residues IgLM wrote into the span in `infilled_residues` and where those residues landed in `spans`.

#### Applications

This is the CDR loop diversification workflow from the paper: hold a parent framework fixed, re-design the loop, and screen the resulting library. Infilling CDR H3 across 49 therapeutic antibodies produced libraries with reduced predicted aggregation propensity (SAP score), increased predicted solubility (CamSol Intrinsic), and higher OASis identity than the parents, so the typical pipeline is infill a loop, then filter the designs on developability predictors and `iglm-score`.

#### Usage Tips

- **Span coordinates are 1-indexed and inclusive.** This is the easiest thing to get wrong: upstream's 0-indexed, end-exclusive `(98, 106)` for the 1JPT CDR H3 is `IgLMSpan(start=99, end=106)` here. Supply exactly one span per sequence; a span whose `end` runs past its sequence, or whose `end` precedes its `start`, is rejected at input construction.
- **IgLM chooses the span length, which need not match the original.** The new segment is returned per design in `results[i].infilled_residues` and is free to be shorter or longer than the residues it replaced. The paper found infilled CDR H3 length distributions varying considerably across the 49 parents, including occasional loops of fewer than five residues, so check the returned lengths before committing designs downstream.
- **Hand `results[i].spans` straight back to `iglm-score`.** Because each design's new span has its own length, the bundle also reports where it landed as `IgLMSpan` objects, so ranking a library by its span-conditional likelihood needs no coordinate arithmetic: `run_iglm_score(IgLMScoringInput(sequences=designs.sequences, spans=designs.spans), IgLMScoringConfig())`.
- **Spans of roughly 10-20 residues match the training distribution.** Masks were drawn uniformly from that range during training, so much shorter or much longer spans are extrapolation.
- **`max_length` has to cover the masked parent as well as the new span.** The whole masked sequence is the prompt here, so the 150-token default leaves a full variable domain roughly 30 tokens of headroom for the new span; raise it (up to 512) for long chains or long spans.

### IgLM Scoring (`iglm-score`)

Computes the autoregressive likelihood of each sequence under its chain and species conditioning, returning `log_likelihood`, `avg_log_likelihood`, and `perplexity` per sequence, plus optional per-position logits over the 33-token vocabulary. Supplying a span switches to IgLM's infilling likelihood, scoring that span given the rest of the chain.

#### Applications

Use the whole-sequence score as a naturalness measure for antibody sequences - the paper's human versus non-human classification on the IMGT mAb DB (ROC AUC 0.96) rests on exactly this quantity. Use the span-conditional score to rank re-designed loops from `iglm-infill` against one another, since every design shares the same parent framework and span and so is scored under identical context.

#### Usage Tips

- **`spans` is optional and broadcastable; omit it (or pass `None`) to score the whole chain.** One span broadcasts across every sequence, which is the convenient form for ranking a library of designs that share a parent framework and loop position.
- **Compare length-normalized scores under identical conditioning.** `log_likelihood` is a sum and grows with sequence length, so use `avg_log_likelihood` or `perplexity` across sequences of different lengths. Lower perplexity means IgLM finds the sequence more antibody-like under the conditioning it was given; scores obtained under different `chains`, `species`, or `model_checkpoint` values sit on different scales and are not comparable.
- **`avg_log_likelihood` is the value upstream's `IgLM.log_likelihood` returns.** Upstream reports the mean per-token log-likelihood and derives perplexity as `exp(-avg_log_likelihood)`, which is what `perplexity` holds here.
- **The scored region ends with IgLM's terminal token.** Following upstream, a sequence therefore contributes one more scored token than it has residues, which slightly shifts both the sum and the mean relative to a residues-only score.
- **`return_logits` defaults to `False`.** The logits array is `(scored_len, 33)` per sequence; enable it only when you need the per-position distributions, and read the column order from the returned `vocab`.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every IgLM tool in this toolkit (`iglm-sample`, `iglm-infill`, `iglm-score`).

- **The weights ship inside the pip package, so there is no download step and no token.** Both checkpoints live under `iglm/trained_models/` in the `iglm` distribution (about 59 MB together), so setting up the toolkit never fetches weights from a model hub and nothing has to be requested or authorized. The JHU license still limits use to non-commercial, non-profit purposes, and commercial use is licensed separately through [Johns Hopkins Technology Ventures](https://jhtv.e-lucid.com/product/generative-language-model-for-antibody-sequence-design-iglm).
- **`model_checkpoint` is shared by all three tools and reloads the worker when it changes.** `IgLM` (12.9M parameters) is the model the paper evaluates and the default; `IgLM-S` (1.4M parameters) is the smaller variant the paper compares against. Both carry the same conditioning vocabulary and the same 512-token context, so inputs transfer between them, but scores do not.
- **The models are small; throughput, not memory, is the constraint.** Neither checkpoint is large enough for GPU memory to be the limiting factor. For the two generation tools, cost is dominated by rejection sampling - `batch_size` (default 8) sets how many candidate sequences are drawn per forward pass, and raising it is the main lever on wall-clock time when `num_sequences_per_input` is large.
- **The conditioning vocabulary is closed.** `chains` and `species` accept only the two chain types and six species IgLM was conditioned on; anything else is rejected at input construction rather than silently mapped. The species set mirrors the OAS repertoires the model was trained on.
- **IgLM models antibody variable domains only.** It was trained exclusively on heavy- and light-chain variable sequences, so constant domains, single-chain constructs with linkers, and non-antibody proteins fall outside its distribution - use a general-purpose protein language model such as ProGen2 or ESM2 for those.
