<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/crispritz"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# CRISPRitz

![CRISPRitz](https://proto-bio.github.io/proto-assets/images/tool/crispritz/hero.png)

> [!NOTE]
> **License:** CRISPRitz is licensed under Custom (CRISPRitz Dual License) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://github.com/pinellolab/CRISPRitz/blob/v2.8.1/LICENSE) for full terms.

## Overview

`crispritz-search` wraps CRISPRitz 2.8.1 for exhaustive, PAM-aware CRISPR
target enumeration with substitutions, DNA bulges, and RNA bulges.

## Background

CRISPRitz indexes every PAM-compatible site in a caller-supplied genome and
then aligns batches of guides under explicit mismatch and bulge limits. Unlike
a general local-alignment search, its index and traversal retain nuclease PAM
orientation and distinguish DNA bulges, RNA bulges, and joint bulge alignments.

## Tools

### CRISPRitz Off-Target Search (`crispritz-search`)

Builds or reuses a content-addressed TST index and returns every native target
row as structured data.

#### Applications

Use this tool to enumerate candidate genomic off-targets for CRISPR guide
ranking, to audit exact mismatch and bulge thresholds, and to search custom or
newly fetched genome assemblies without maintaining a bundled database.

#### Usage Tips

- `guides`: one or more **PAM-free protospacers** of equal length. Supply the
  20-nt guide for SpCas9, not a 23-nt guide-plus-PAM and not CRISPRitz's `NNN`
  placeholder form. RNA `U` is normalized to DNA `T`.
- `reference_fasta`: either one uncompressed `.fa`, `.fasta`, or `.fna` file
  (multi-record files are supported), or a directory containing those files.
  An `.fna` artifact returned by `ncbi-reference-bundle` can be passed directly.
- `reference_id`: optional versioned assembly/accession provenance, such as
  `GCF_000001405.40`.
- `pam` and `pam_side`: the IUPAC PAM without the guide. SpCas9 is `NGG` on
  `three_prime`; Cas12a is commonly `TTTV` on `five_prime`.
- mismatch, DNA-bulge, RNA-bulge, and optional combined edit limits are
  inclusive maxima.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

The reference is never modified. The wrapper splits multi-record FASTA only in
its private staging area and builds a CRISPRitz TST index under a cache key made
from the reference checksum, guide length, PAM, PAM side, bulge capacity, and
CRISPRitz version. No genome or curated sequence database ships with Proto.

Setup compiles CRISPRitz from its pinned source with `g++ -fopenmp`, which
Apple's `clang++` rejects, so the environment builds on Linux only and setup
exits early on macOS.

## Outputs

Every upstream target row is returned, including aligned guide and genomic
strings, explicit DNA/RNA bulge class and size, mismatch count, total edits,
strand, chromosome, and the aligned target's `start` and `end`, converted from
CRISPRitz's zero-based coordinates to the 1-indexed inclusive convention used
throughout Proto. CRISPRitz 2.8 can also emit joint `RNA,DNA` alignments; these
are represented as `dna_rna` while the unmodified upstream label remains
available. The output also records the combined reference checksum, index reuse,
all search parameters, and the pinned CRISPRitz version.
