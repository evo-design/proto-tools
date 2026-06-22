<a href="https://bio-pro.mintlify.app/tools/structure-prediction/x3dna"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# X3DNA Fiber

![X3DNA Fiber](https://proto-bio.github.io/proto-assets/images/tool/x3dna/hero.png)

> [!NOTE]
> **License:** X3DNA Fiber has a CC-BY-NC-4.0 license and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://x3dna.org/) for full terms. X3DNA is user-provisioned: it is not bundled or auto-downloaded, and must be obtained yourself after free registration at [x3dna.org](https://x3dna.org/). Please cite Lu & Olson (2003, 2008).

## Overview

X3DNA Fiber wraps the `fiber` program from the [3DNA/X3DNA](https://x3dna.org/) suite, developed by Xiang-Jun Lu and Wilma K. Olson. From a base sequence it builds an idealized canonical (Arnott) fiber model of a nucleic-acid duplex: A-DNA, B-DNA, Z-DNA, or an A-form RNA duplex, with the complementary strand generated automatically. It is deterministic, CPU-only, and produces three-dimensional atomic coordinates of a regular helix rather than a predicted or energy-minimized structure.

## Background

The `fiber` program in 3DNA ([Lu & Olson, 2003](https://doi.org/10.1093/nar/gkg680); [Lu & Olson, 2008](https://doi.org/10.1038/nprot.2008.104)) constructs regular helices by repeating the canonical geometry of a chosen fiber model along a user-supplied sequence. These geometries are the idealized repeating units derived from fiber-diffraction studies, most of them based on the work of Chandrasekaran & Arnott, that define the canonical [A-DNA](https://en.wikipedia.org/wiki/A-DNA), [B-DNA](https://en.wikipedia.org/wiki/B-DNA), and [Z-DNA](https://en.wikipedia.org/wiki/Z-DNA) conformations, together with an A-form RNA duplex. Each base or base pair is placed according to the fixed helical parameters (rise, twist, and the associated base-pair geometry) of the selected form, so the result is a uniform, sequence-threaded helix.

Because every base step uses the same canonical geometry, the output is an idealized model, not a structure prediction: it does not account for sequence-dependent deformation, flexibility, non-canonical pairing, or solvent and counterion effects, and no energy is computed or minimized. The models are most useful as clean, reproducible starting structures for visualization, as templates for docking or molecular-dynamics setup, or as canonical references against which experimental or predicted structures can be compared.

The 3DNA/X3DNA software was originally developed by Xiang-Jun Lu while in Wilma K. Olson's laboratory at Rutgers University and is maintained by Lu (now at Columbia University) in collaboration with Olson. The distribution is gated behind free registration on the 3DNA Forum at [x3dna.org](https://x3dna.org/).

### Learning Resources

- [X3DNA-DSSR homepage](https://x3dna.org/) (Lu & Olson) - the canonical homepage, forum, downloads, and documentation, including the registration required to obtain the software.
- [3DNA fiber models article](https://x3dna.org/articles/3dna-fiber-models) (Xiang-Jun Lu) - a walkthrough of the `fiber` program, the catalog of canonical fiber models, and the command-line interface.

## Tools

### X3DNA Fiber (`x3dna-fiber`)

Builds an idealized canonical fiber structure from each input base sequence and returns one `Structure` per sequence (the duplex by default, or the sense strand alone), exportable to PDB or mmCIF.

#### Applications

Use this to generate clean canonical helices from a sequence, for example to obtain a B-DNA duplex as a starting model for docking or a molecular-dynamics build, to produce an A-form RNA duplex template, or to create a canonical reference structure to compare against an experimental or predicted model. It is a generator of idealized models rather than a method for predicting the real structure of a given sequence.

#### Usage Tips

- **`form` (default `B-DNA`) selects the canonical model to build.** Choose `A-DNA`, `B-DNA`, `Z-DNA`, or `RNA` (an A-form RNA duplex). `T` and `U` are interconverted automatically to match the form (`T` for the DNA forms, `U` for RNA), so the same sequence can be built in any form without manual editing.
- **`single_stranded` (default `False`) returns only the sense strand.** Leave it `False` to build the full duplex with the auto-generated complement; set it `True` (fiber `-single`) when you need a single strand, such as a single-stranded RNA helix.
- **X3DNA must be installed by you (it is user-provisioned).** X3DNA is not auto-downloaded (it is gated behind free registration at [x3dna.org](https://x3dna.org/) and distributed under CC-BY-NC-4.0). After downloading X3DNA v2.4, place it in the managed cache so `bin/fiber` is found automatically with no environment variable, or point the tool at the install root via the `X3DNA` environment variable (or `PROTO_X3DNA_WEIGHTS_DIR`, or the `x3dna_dir` config field). See [SETUP.md](SETUP.md) for copy-paste steps. Without a resolvable install, the tool cannot run.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to the X3DNA Fiber tool in this toolkit (`x3dna-fiber`).

- **Runs on CPU and is deterministic.** The `fiber` program is a fast command-line builder with no GPU and no random sampling; the same sequence and `form` always yield the same coordinates. Generation is near-instant for typical sequences.
- **User-provisioned local install only.** X3DNA is gated behind free registration at [x3dna.org](https://x3dna.org/) under CC-BY-NC-4.0 and is not bundled or downloaded automatically. After downloading X3DNA v2.4, place it in the managed cache so `bin/fiber` resolves automatically, or expose it through the `X3DNA` environment variable, `PROTO_X3DNA_WEIGHTS_DIR`, or the `x3dna_dir` config field (see [SETUP.md](SETUP.md)). The tool therefore runs locally with `device='cpu'` and is not available on hosted (`device='cloud'`) workers.
- **Produces idealized canonical models, not predictions.** Output is a regular fiber helix with fixed canonical geometry for the chosen form; it does not model sequence-dependent deformation, non-canonical pairing, or energetics. Use a structure-prediction or refinement method when the real conformation of a specific sequence is required.
