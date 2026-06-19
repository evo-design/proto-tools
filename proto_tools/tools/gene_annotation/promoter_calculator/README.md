<a href="https://bio-pro.mintlify.app/tools/gene-annotation/promoter-calculator"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Salis Lab Promoter Calculator

![Salis Lab Promoter Calculator](https://proto-bio.github.io/proto-assets/images/tool/promoter_calculator/hero.png)

> [!NOTE]
> **License:** Salis Lab Promoter Calculator has a GPL-3.0 license. Please refer to [the license](https://github.com/barricklab/promoter-calculator/blob/master/LICENSE) for full terms.

## Overview

The [Salis Lab Promoter Calculator](https://github.com/barricklab/promoter-calculator) is a 346-parameter biophysical and machine-learning model from the [Salis Lab](https://salislab.net/) that predicts the strength of [σ70](https://en.wikipedia.org/wiki/Sigma_factor) housekeeping promoters in *Escherichia coli*. It scans both strands of an input DNA sequence for every candidate transcription start site, decomposes the surrounding region into the canonical promoter elements ([UP element](https://en.wikipedia.org/wiki/Promoter_(genetics)), −35 hexamer, spacer, [−10 hexamer (Pribnow box)](https://en.wikipedia.org/wiki/Pribnow_box), and discriminator), and returns the predicted [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy) of RNA polymerase holoenzyme binding (`dG_total`) together with a calibrated transcription initiation rate (`Tx_rate`).

## Background

The Promoter Calculator ([LaFleur et al., 2022](https://doi.org/10.1038/s41467-022-32829-5)) predicts site-specific σ70 transcription initiation rates from DNA sequence alone. σ70 is the housekeeping [sigma factor](https://en.wikipedia.org/wiki/Sigma_factor) that recruits *E. coli* RNA polymerase to the majority of constitutive promoters, and its strength is set by the sequence and geometry of the surrounding promoter elements. The model decomposes a candidate promoter into the UP element, the −35 hexamer, the spacer, the extended and core −10 hexamers, the discriminator, and the initial transcribed region, then fits 346 Ridge-regression coefficients that link these elements to an overall binding free energy ΔG_total. The fitted ΔG_total is mapped to an absolute transcription initiation rate via `log(TX / TX_ref) = -β (ΔG_total - ΔG_total_ref)`, with the fit grounded in [massively parallel reporter assay](https://en.wikipedia.org/wiki/Reporter_gene) measurements. The released coefficients were trained on 5,193 designed promoters with a single dominant transcription start site and validated against 22,132 diverse bacterial σ70 promoters drawn from multiple datasets.

The reference Python implementation used here is the [barricklab/promoter-calculator](https://github.com/barricklab/promoter-calculator) fork from the [Barrick Lab](https://barricklab.org/), which packages the original [Salis Lab algorithm](https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator) for streamlined installation and adds optional multi-threading for the internal transcription-start-site scan.

### Learning Resources

- [barricklab/promoter-calculator](https://github.com/barricklab/promoter-calculator) (Barrick Lab) - the Python fork used here, with installation instructions and the command-line surface that the wrapper drives.
- [hsalis/SalisLabCode (Promoter_Calculator)](https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator) (Salis Lab) - the original reference implementation distributed alongside the publication and the source of the trained coefficients.
- [salislab.net](https://salislab.net/) (Salis Lab) - the lab home page and entry point to the hosted Promoter Calculator web service.

## Tools

### Salis Lab Promoter Calculator (`promoter-calculator`)

Scans one or more DNA sequences on both strands for every candidate σ70 transcription start site and returns, per sequence, a list of `PromoterPrediction` rows. Each row carries the predicted TSS position and strand, the binding free energy `dG_total` (kcal/mol), the transcription initiation rate `Tx_rate` (arbitrary units), the DNA spanning the predicted promoter, and the start and end positions of the UP, −35, spacer, −10, and discriminator elements.

#### Applications

Use this to quantify σ70 promoter strength when designing or analysing *E. coli* expression cassettes. Common workflows include ranking a synthetic library such as the [Anderson collection](https://parts.igem.org/Promoters/Catalog/Anderson) of J231xx variants by predicted `Tx_rate` to pick a target strength, sweeping a candidate construct to flag unintended cryptic promoters that could drive off-target transcription before ordering DNA, and annotating native intergenic regions across an *E. coli* genome to estimate baseline σ70 activity. Pair the predicted transcription rate with downstream [ribosome binding site](https://en.wikipedia.org/wiki/Shine-Dalgarno_sequence) strength and plasmid copy number when projecting end-to-end protein expression.

#### Usage Tips

- **Set `circular=True` for plasmids and bacterial chromosomes.** Linear scanning of a circular sequence cannot see candidates that span the wraparound origin. When the input is circular, the calculator examines the junction explicitly and recovers any promoter that straddles it.
- **Short inputs need flanking context to score.** The element scan needs roughly 20 nucleotides on either side of the promoter region; sequences shorter than the scan window return no predictions. Pad with neutral flanking sequence when scoring a single promoter element in isolation.
- **Predictions are calibrated for *E. coli* σ70 only.** Applying the model to other organisms or to alternative sigma factors (σS, σ32, σ54, σ28) is not validated. Treat any cross-organism output as a relative ranking, not as a calibrated rate.
- **`Tx_rate` is transcription initiation, not protein expression.** The model captures RNA polymerase binding and open-complex formation. End-to-end expression also depends on the ribosome binding site, mRNA stability, copy number, and growth state, none of which the calculator sees.
- **The model does not see transcription factors, attenuation, or anti-σ factors.** Repression by a TF bound near the promoter, riboswitch attenuation, and anti-σ sequestration all reduce in vivo output without changing the predicted `Tx_rate`. Use the prediction as the unregulated upper bound.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Promoter Calculator tool in this toolkit (`promoter-calculator`).

- **Runs on CPU only.** The model is a Ridge-regression sum over 346 coefficients that ship with the Python source. No neural network is loaded at inference time, and there is no GPU acceleration to enable.
- **Self-contained after install.** The standalone setup builds a Python virtual environment and pulls dependencies once; subsequent runs need no further network access and no model-weight downloads.
- **`threads` parallelises the internal TSS scan within a single sequence.** Raising it shortens wall-clock time on long inputs but does not change the predictions. Across input sequences the wrapper itself runs sequentially.
