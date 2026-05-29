<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/alphagenome"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# AlphaGenome

> [!NOTE]
> **License:** AlphaGenome uses Apache-2.0 for code and Custom (AlphaGenome Terms of Use) for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Model weights are gated and require accepting the provider's terms and authenticating with a HuggingFace token. Please refer to the [code license](https://github.com/google-deepmind/alphagenome_research/blob/main/LICENSE) and [model weights license](https://deepmind.google.com/science/alphagenome/model-terms) for full terms.

## Overview

AlphaGenome is a [deep learning](https://en.wikipedia.org/wiki/Deep_learning) model for regulatory genomics developed by Google DeepMind. It predicts a broad range of functional genomic measurements directly from DNA sequence across context windows of up to roughly one million base pairs, spanning [gene expression](https://en.wikipedia.org/wiki/Gene_expression), [chromatin accessibility](https://en.wikipedia.org/wiki/ATAC-seq), [transcription factor](https://en.wikipedia.org/wiki/Transcription_factor) binding, histone modifications, [RNA splicing](https://en.wikipedia.org/wiki/RNA_splicing), and three-dimensional [chromatin contacts](https://en.wikipedia.org/wiki/Chromosome_conformation_capture). This tool implementation provides six operations covering interval, sequence, and variant prediction alongside three scoring modes.

## Background

Gene regulation is encoded in non-coding DNA through [cis-regulatory elements](https://en.wikipedia.org/wiki/Cis-regulatory_element) such as promoters, [enhancers](https://en.wikipedia.org/wiki/Enhancer_(genetics)), and insulators, whose activity depends on sequence context that can extend across hundreds of kilobases. Relating a DNA sequence, or a non-coding [genetic variant](https://en.wikipedia.org/wiki/Mutation), to its functional consequences therefore requires models that read long stretches of sequence and predict many regulatory readouts together.

AlphaGenome ([Avsec et al., 2026](https://doi.org/10.1038/s41586-025-10014-0)) is a sequence-to-function model that accepts a genomic interval of up to roughly one megabase and predicts thousands of genome tracks at base or near-base resolution. The predicted assays span [RNA-seq](https://en.wikipedia.org/wiki/RNA-Seq) coverage, [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_of_gene_expression) and PRO-cap transcription initiation, [ATAC-seq](https://en.wikipedia.org/wiki/ATAC-seq) and [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq) chromatin accessibility, [ChIP-seq](https://en.wikipedia.org/wiki/ChIP_sequencing) profiles for histone modifications and transcription factors, splice site positions, splice site usage and junctions, and [chromatin contact maps](https://en.wikipedia.org/wiki/Chromosome_conformation_capture). Because the model scores arbitrary sequence, the effect of a variant can be estimated by comparing predictions for the reference and alternate alleles, which supports interpretation of non-coding variation and systematic [in silico mutagenesis](https://en.wikipedia.org/wiki/Saturation_mutagenesis) of regulatory regions. Models are available for both the human and mouse genomes.

### Learning Resources

- [AlphaGenome overview](https://deepmind.google.com/science/alphagenome/) (Google DeepMind) - the official project page summarizing what AlphaGenome does, how to access it, and its model terms.
- [AlphaGenome: AI for better understanding the genome](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/) (Google DeepMind) - the announcement blog post introducing the model, its capabilities, and intended research uses.
- [AlphaGenome research code (GitHub)](https://github.com/google-deepmind/alphagenome_research) - the reference client and model code, with example notebooks for prediction, variant scoring, and in silico mutagenesis.
- [AlphaGenome model weights (HuggingFace)](https://huggingface.co/google/alphagenome-all-folds) - the gated model card describing the released checkpoints and their terms of use.

## Tools

### Predict Intervals (`alphagenome-predict-intervals`)

Predicts base-resolution regulatory signal tracks for one or more genomic intervals specified by chromosome and coordinates.

#### Applications

This tool surveys predicted chromatin accessibility, expression, transcription factor binding, and histone marks across a locus of interest, and supports comparison of predicted regulatory activity between cell types or tissues by restricting the prediction to chosen ontology terms. The resulting profiles also serve as references that later variant or mutagenesis analyses can be compared against.

#### Usage Tips

- **The requested outputs are configurable.** Any combination of the available output types may be requested together in a single call.
- **Center the feature of interest.** The model has the most flanking context in both directions at the center of the requested interval, so predictions are best supported there.

### Predict Sequences (`alphagenome-predict-sequences`)

Predicts the same regulatory signal tracks directly from raw DNA sequences rather than from genome coordinates.

#### Applications

This tool scores synthetic or edited sequences, such as designed promoters and enhancers, that do not correspond to a reference genome position, and it is well suited to evaluating candidate sequences from a generative model before committing to laboratory synthesis.

#### Usage Tips

- **Raw sequences are not resized.** Each sequence must already be exactly one of the supported context lengths.
- **Only DNA bases are accepted.** Sequences may contain only the bases A, C, G, T, and N.

### Predict Variants (`alphagenome-predict-variants`)

Predicts regulatory signal tracks for both the reference and alternate alleles of a variant within its surrounding interval.

#### Applications

This tool shows how a non-coding variant reshapes predicted accessibility, expression, or splicing across a region, and it reveals the spatial extent of a variant's predicted effect rather than reducing it to a single number.

#### Usage Tips

- **The variant must lie within the interval.** A wider interval captures more of the distal regulatory consequences of the variant.
- **Use variant scoring for a ranked summary.** When only effect-size magnitudes are needed, the variant scoring operation is more direct than reading the raw tracks.

### Score Variants (`alphagenome-score-variants`)

Summarizes variant effects into per-track records using the model's recommended variant scorers, comparing reference and alternate predictions.

#### Applications

This tool prioritizes candidate causal variants from a fine-mapping or [genome-wide association study](https://en.wikipedia.org/wiki/Genome-wide_association_study) and annotates lists of non-coding variants with predicted effects across many assays at once.

#### Usage Tips

- **`variant_scorers` defaults to the full recommended set.** Leaving it unset gives broad coverage but takes longer, while naming individual scorers restricts the analysis to the assays that matter for the question.
- **The `_ACTIVE` suffix changes what is measured.** Standard scorers report the directional change between alleles (the log fold change of the alternate against the reference), whereas `_ACTIVE` scorers report the absolute activity level of the stronger allele and are non-directional.

### Score Intervals (`alphagenome-score-intervals`)

Produces gene-level RNA-seq expression scores summarizing predicted activity across one or more genomic intervals.

#### Applications

This tool estimates predicted expression for genes overlapping a region of interest and compares predicted activity across a panel of intervals on a common scale.

#### Usage Tips

- **Interval scoring is gene-centric.** It depends on a wide region of context, so intervals must be large enough to contain the surrounding gene. Very short intervals are not suitable.

### Score ISM Variants Batch (`alphagenome-score-ism-variants-batch`)

Performs in silico mutagenesis by scoring every single-base substitution across a chosen sub-window and returning the effects as per-track records.

#### Applications

This tool maps which exact positions within a promoter or enhancer drive a predicted regulatory signal and produces per-base importance profiles suitable for visualization as sequence logos.

#### Usage Tips

- **Keep the mutagenesis window narrow.** The sub-window must lie fully inside the surrounding interval, and because the number of scored substitutions grows with its width, windows of tens to low hundreds of bases keep each run tractable.
- **An existing variant can be applied first.** A known variant may be set before mutagenesis to explore how it changes the local sensitivity landscape.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **The model weights are gated.** Running any tool requires accepting the AlphaGenome Terms of Use and authenticating with a HuggingFace access token, and use is restricted to non-commercial scientific research.
- **Coordinates are 0-based and half-open.** Genomic intervals follow the [BED](https://en.wikipedia.org/wiki/BED_(file_format)) convention used throughout genome browsers, where the start is inclusive and the end is exclusive.
- **Context lengths are fixed.** AlphaGenome operates at 16,384, 131,072, 524,288, and 1,048,576 base pairs. Intervals that do not already match one of these lengths are centered and resized up to the smallest supported length that contains them, and longer windows capture more distal regulation at higher compute cost.
- **Output can be filtered by tissue and organism.** The prediction tools restrict their tracks to particular cell types or tissues through [UBERON](https://en.wikipedia.org/wiki/Uberon) ontology terms, and every tool runs against either the human or mouse genome through the organism setting.
