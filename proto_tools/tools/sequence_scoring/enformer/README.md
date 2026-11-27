<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/enformer"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Enformer

![Enformer](https://proto-bio.github.io/proto-assets/images/tool/enformer/hero.png)

> [!NOTE]
> **License:** Enformer uses Apache-2.0 for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/google-deepmind/deepmind-research/blob/master/LICENSE) and [model weights license](https://github.com/google-deepmind/deepmind-research/blob/master/enformer/README.md) for full terms.

## Overview

[Enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer) is a deep learning model that predicts [gene expression](https://en.wikipedia.org/wiki/Gene_expression) and chromatin activity directly from DNA sequence, developed by Google DeepMind and Calico Life Sciences. It reads a fixed 196,608 base-pair input window and predicts activity across thousands of genomic measurement tracks, summarized over 896 output bins of 128 base pairs each. This toolkit exposes Enformer through a single registered tool that returns the predicted track activity for one or more sequences.

## Background

Enformer ([Avsec et al., 2021](https://doi.org/10.1038/s41592-021-01252-x)) is a neural network that predicts [gene expression](https://en.wikipedia.org/wiki/Gene_expression) and chromatin state from genomic DNA sequence. Its architecture combines convolutional layers with [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) self-attention, which allows the model to integrate the influence of distal [cis-regulatory elements](https://en.wikipedia.org/wiki/Cis-regulatory_element) such as [enhancers](https://en.wikipedia.org/wiki/Enhancer_(genetics)) located up to 100 kilobases away from a [promoter](https://en.wikipedia.org/wiki/Promoter_(genetics)). The published work reports that this long-range modeling substantially improves gene expression prediction accuracy relative to earlier convolutional models, and that it yields more accurate predictions of the effect of genetic variants on expression for both natural variants and saturation mutagenesis measured by reporter assays.

Enformer predicts activity across 896 output bins, each summarizing 128 base pairs, for a large panel of functional genomics assays. The human output head covers 5,313 tracks spanning [chromatin accessibility](https://en.wikipedia.org/wiki/DNase_I_hypersensitive_site), [transcription factor](https://en.wikipedia.org/wiki/Transcription_factor) binding, [histone modifications](https://en.wikipedia.org/wiki/Histone), and [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_gene_expression) expression measurements, and the mouse output head covers 1,643 tracks. Because the model maps sequence directly to these signals, it can be used to compare the predicted regulatory activity of alternative alleles at the same locus, which is the basis for its use in interpreting noncoding genetic variation. The published analysis additionally shows that Enformer learns enhancer-promoter relationships directly from the input sequence.

### Learning Resources

- [Predicting gene expression with AI](https://deepmind.google/blog/predicting-gene-expression-with-ai/) (Google DeepMind). The announcement blog post introducing Enformer, its long-range modeling, and its use in interpreting genetic variants.
- [Enformer model repository](https://github.com/google-deepmind/deepmind-research/tree/master/enformer) (Google DeepMind). Official Enformer code, model card, and usage guidance.
- [enformer-pytorch](https://github.com/lucidrains/enformer-pytorch) (Phil Wang). The PyTorch implementation and pretrained weight loader that this toolkit uses to run the model.

## Tools

### Enformer Prediction (`enformer-prediction`)

Predicts regulatory track activity for one or more DNA sequences. The tool accepts input in two forms. In exact-window form, each sequence is exactly 196,608 base pairs, the full Enformer model context, and is passed to the model directly. In target-range form, each sequence is longer than the model context and is paired with a target range, and the tool extracts the 196,608 base-pair context window that keeps the requested range inside the model's output bins. Each result carries the predicted activity matrix of shape 896 bins by the number of selected tracks, together with the coordinates that map the output bins back onto the source sequence.

#### Applications

This tool is appropriate for analyses that relate noncoding DNA sequence to predicted regulatory activity. Representative applications include predicting the effect of a candidate variant by comparing the activity of the reference and alternate sequences, prioritizing noncoding variants for follow-up by the magnitude of their predicted regulatory change, screening designed or synthetic regulatory sequences for predicted promoter or enhancer activity, and surveying the predicted chromatin and expression landscape across a genomic locus of interest.

#### Usage Tips

- **Each input sequence must provide a full 196,608 base-pair model context.** A sequence supplied in exact-window form must be exactly this length. A longer sequence must be paired with a `target_range`, and the tool then extracts the model context window around that range. The tool does not pad missing context, so a sequence shorter than the model context without a target range is rejected.
- **`output_tracks` selects which of the model's tracks are returned and defaults to the single track at index 0.** The human head exposes 5,313 tracks and the mouse head exposes 1,643 tracks. Selecting only the tracks of interest, rather than all of them, keeps the returned activity matrix small and the analysis focused on the relevant assays.
- **`species` selects the human or mouse output head and defaults to human.** The track index meanings differ between the two heads, so the `output_tracks` indices should be chosen to match the selected species.
- **`batch_size` controls how many sequences are processed together on the GPU and defaults to 1.** Larger values increase throughput when many sequences are scored in one call, subject to the available GPU memory.
- **The output covers a 114,688 base-pair central window, narrower than the input context.** The 896 output bins span the central portion of the input, and the per-result `output_start` and `output_end` coordinates report exactly which part of the source sequence the bins cover. A feature of interest must fall within this central window to receive a prediction.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Enformer tool in this toolkit (`enformer-prediction`).

- **Enformer runs on a GPU and downloads its published weights on first use.** The model parameters are retrieved from the hosted `enformer-official-rough` checkpoint the first time the tool runs and are reused on subsequent runs. A CUDA-capable GPU is recommended because the 196,608 base-pair context makes CPU inference slow.
- **Output coordinates are 0-based with exclusive ends, following the genomics interval convention rather than the 1-based residue numbering used elsewhere in proto-tools.** The `context_start`, `context_end`, `output_start`, and `output_end` fields locate the model window and the output-bin span within the source sequence using this convention.
