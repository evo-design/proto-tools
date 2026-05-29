<a href="https://bio-pro.mintlify.app/tools/rna-splicing/splice-transformer"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# SpliceTransformer

> [!NOTE]
> **License:** SpliceTransformer is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/ShenLab-Genomics/SpliceTransformer/blob/main/LICENSE) for full terms.

## Overview

[SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer) (SpTransformer) is a deep learning model from the Shen Lab that predicts [pre-mRNA splice sites](https://en.wikipedia.org/wiki/RNA_splicing) directly from genomic sequence with tissue-specific resolution. For each position in a target sequence it reports the probability of an acceptor (3' splice site) or donor (5' splice site) along with the predicted usage of that site across fifteen human tissues. This toolkit runs SpliceTransformer through a single registered tool that returns per-position probabilities for the three splice-site classes and the fifteen tissues.

## Background

SpliceTransformer ([You et al., 2024](https://doi.org/10.1038/s41467-024-53088-6)) is a deep learning framework that predicts tissue-specific splicing from genomic sequence. The architecture combines convolutional encoders in the style of [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) with a [Sinkhorn transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) attention module, which lets the model capture the long-range sequence interactions that influence splice site selection. The published work reports that SpliceTransformer outperforms all previous methods on splicing prediction. Applied to roughly 1.3 million variants in the [ClinVar](https://en.wikipedia.org/wiki/ClinVar) database, it attributes 60 percent of intronic and synonymous pathogenic mutations to splicing alterations, and it connects tissue-specific splicing changes to human disease in validations spanning brain disease cohorts and a diabetic nephropathy dataset.

The model evaluates a 1,000 nucleotide target region flanked by 4,000 nucleotides of genomic context on each side, and it returns a prediction for every position in the target region. The first three output channels form a softmax over the splice-site class of the position, namely neither site, acceptor, or donor. The remaining fifteen channels report the predicted usage of the position as a splice site in each of fifteen human tissues, derived from [Genotype-Tissue Expression (GTEx)](https://www.gtexportal.org/) data. The tissue channels are produced by an independent sigmoid for each tissue, so a position can be a confident splice site overall while being used in only a subset of tissues.

### Learning Resources

- [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer) (Shen Lab). Official repository, containing the reference model definition and usage examples that this toolkit follows.

## Tools

### SpliceTransformer Splicing Prediction (`splice-transformer-prediction`)

Predicts splice sites at single-nucleotide resolution across a batch of target sequences and reports tissue-specific usage for each predicted site. The tool accepts one or more 1,000 nucleotide target sequences, each paired with a 4,000 nucleotide left context and a 4,000 nucleotide right context drawn from the same genomic locus, and returns a probability tensor of shape `[batch, 1000, 18]`. The first three channels give the probability that a position is neither a splice site, an acceptor, or a donor, and the remaining fifteen channels give the predicted usage of that position as a splice site in each tissue.

#### Applications

This tool is appropriate for any analysis that begins with a genomic locus and asks where the splice sites are and how their usage differs between tissues. Representative applications include annotating candidate splice sites in a newly characterised gene, comparing predicted acceptor and donor usage between tissues such as brain and liver to identify tissue-specific isoforms, and screening a region for positions whose splicing behaviour is restricted to a particular tissue. The fifteen tissue channels make the tool well suited to studies of [alternative splicing](https://en.wikipedia.org/wiki/Alternative_splicing) where the tissue of interest is known in advance.

#### Usage Tips

- **Sequence lengths are fixed by the published model and are enforced on input.** Every target sequence must be exactly 1,000 nucleotides and every left and right context must be exactly 4,000 nucleotides. Inputs of any other length are rejected before the model runs.
- **The target and its two contexts must come from the same genomic locus and be supplied in genomic order.** The model concatenates the left context, the target, and the right context into a single window, so a context drawn from a different region or assembled in the wrong order produces predictions that do not correspond to the intended locus.
- **The three input lists must contain the same number of sequences.** Each target is paired by position with one left context and one right context, and a mismatch in list length is rejected on input.
- **Acceptor and donor probabilities are most informative when read against the canonical [GT-AG rule](https://en.wikipedia.org/wiki/RNA_splicing).** Confident donor predictions are expected at GT dinucleotides and confident acceptor predictions at AG dinucleotides, so inspecting these positions helps confirm that a high score reflects a genuine splice site.
- **Tissue channels report usage rather than the presence of a splice site.** A position can carry a high acceptor or donor probability while showing strong usage in only a few tissues, so differential analysis across the tissue channels is the basis for identifying tissue-specific splicing.
- **The model was trained on human sequence and is intended for human loci.** Predictions on sequences from other species are not supported by the training data and should not be interpreted as reliable.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every SpliceTransformer tool in this toolkit (`splice-transformer-prediction`).

- **The eighteen output channels follow a fixed order.** Channel 0 is the probability of neither site, channel 1 the acceptor probability, and channel 2 the donor probability, and these three form a softmax that sums to one. Channels 3 through 17 carry the per-tissue usage in the order adipose tissue, blood, blood vessel, brain, colon, heart, kidney, liver, lung, muscle, nerve, small intestine, skin, spleen, and stomach. The `SPLICE_TISSUE_CHANNEL_INDEX` mapping exported by the toolkit resolves a tissue name to its channel.
- **The prediction is returned as a nested list and is most convenient to work with as an array.** The `prediction` field has shape `[batch, 1000, 18]` and can be converted with `numpy.array(...)` for slicing by position or channel. Results can be exported to NumPy `.npy` or JSON through the standard export method.
