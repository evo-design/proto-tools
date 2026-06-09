<a href="https://bio-pro.mintlify.app/tools/gene-annotation/miranda"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# miRanda

![miRanda](https://proto-bio.github.io/proto-assets/images/tool/miranda/hero.png)

> [!NOTE]
> **License:** miRanda's own code is licensed under GPL-2.0, and it federates over bundled data sources and components, each under its own license terms.
>
> Bundled dependencies, each under its own license:
>
> - [ViennaRNA RNAlib (bundled, v1.5b)](https://bio-pro.mintlify.app/tools/structure-prediction/viennarna): ViennaRNA Package License
>
> Review each source's terms before commercial use or redistribution.

## Overview

[miRanda](https://github.com/mskcc/miRanda) is a microRNA target-site prediction program written by Enright et al. at Memorial Sloan Kettering Cancer Center. It predicts where a microRNA binds an RNA or DNA target by combining a complementarity-based local alignment with a thermodynamic estimate of the resulting duplex's stability, and returns each predicted site with its score, free energy, alignment, and coordinates. This makes it useful for nominating candidate microRNA targets in 3'UTRs and other transcript regions ahead of experimental validation.

## Background

[microRNAs](https://en.wikipedia.org/wiki/MicroRNA) are short (~22 nt) non-coding RNAs that regulate gene expression by base-pairing with complementary sites in target mRNAs — most often in the 3' untranslated region (3'UTR) — and directing those transcripts toward translational repression or degradation. Because pairing is only partial in animals and concentrated in a short 5' "seed" of the microRNA, identifying genuine target sites from sequence alone requires a method tuned to complementarity rather than ordinary sequence identity.

miRanda ([Enright et al., 2003](https://doi.org/10.1186/gb-2003-5-1-r1)) addresses this with a two-phase algorithm. First, a [Smith-Waterman](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) local alignment scores microRNA-to-target *complementarity* (rewarding A:U and G:C pairs, and tolerating G:U wobble) instead of identity, with extra weighting on the microRNA's 5' seed region to reflect its outsized role in binding. Second, for alignments that clear a score threshold, the bundled [ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) RNAlib ([Hofacker et al., 1994](https://doi.org/10.1007/BF00818163)) estimates the duplex minimum free energy (ΔG, kcal/mol). Sites that pass both the score and the energy threshold are reported as candidate target sites.

### Learning Resources

- [miRBase](https://www.mirbase.org/) (Griffiths-Jones et al.) - the reference registry of published microRNA sequences and names, and the standard place to obtain the microRNA queries you scan with.

## Tools

### miRanda Target Scan (`miranda-scan`)

Scans one or more microRNA queries against one or more RNA/DNA target sequences and returns, per target, the predicted target sites — each with its complementarity score, ViennaRNA duplex free energy, 1-indexed inclusive coordinates on both strands, percent identity and similarity, and the aligned strings.

#### Applications

Use this to nominate candidate microRNA target sites in a transcript before committing to experimental validation. Typical workflows scan a small panel of microRNAs (from [miRBase](https://www.mirbase.org/)) against a gene's 3'UTR to rank putative binding sites, or screen one microRNA across a set of candidate target transcripts to shortlist which are most likely regulated.

#### Usage Tips

- **`score_threshold` and `energy_threshold` set the sensitivity-versus-specificity trade-off.** Lowering **score_threshold** below the default of 50 and raising **energy_threshold** toward 0 from the default of -20 kcal/mol both admit weaker sites and recover more candidates at the cost of more false positives; tighten them in the other direction for high-confidence calls.
- **Enable `strict` for higher confidence.** **strict** defaults to off; turning it on passes `-strict` for stringent miRNA:target 5'-seed duplex heuristics that drop noisier, lower-confidence hits.
- **Disable `compute_energy` for a much faster score-only scan.** **compute_energy** runs the ViennaRNA free-energy phase by default; turning it off skips that phase entirely (the reported energy is then 0.0), which is useful for quick large-scale complementarity sweeps where ΔG ranking is not yet needed.
- **All reported positions are 1-indexed and inclusive** on both the target and the microRNA, matching biological coordinate conventions.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every miRanda tool in this toolkit (`miranda-scan`).

- **microRNA queries are user-supplied.** miRanda does not ship a microRNA database; obtain query sequences from [miRBase](https://www.mirbase.org/) (or your own designs) and pass them as `mirna_queries`.
- **Target sequences should be mRNAs or 3'UTRs, not whole genomes.** The aligner builds a query-by-target dynamic-programming matrix, so memory scales with target length. Scan transcript-scale sequences, and use `trim` to cap very long targets rather than feeding whole chromosomes.
- **Statistical shuffling and Z-scores are not supported.** The randomization-based Z-score significance machinery is non-functional upstream and is not exposed by the wrapper; rank candidates by score and free energy instead.
