<a href="https://bio-pro.mintlify.app/tools/structure-alignment/pymol-rmsd"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# PyMOL RMSD

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

PyMOL RMSD performs pairwise structure alignment using Open-Source PyMOL and returns the post-alignment RMSD (`pymol-rmsd-alignment`). It supports both PyMOL `cealign` and regular `align`, which lets callers choose between CE-style structural alignment and PyMOL's sequence-aware alignment workflow.

## Background

RMSD measures the average distance between corresponding atoms after superposition. It is useful when the target fold is known and the design goal is a close geometric match to a reference structure. PyMOL's `cealign` is often useful for global structure alignment, while `align` performs sequence alignment followed by structural refinement and can be useful when residue numbering or sequence length differs.

## Tools

### PyMOL RMSD Alignment (`pymol-rmsd-alignment`)

Align two `Structure` inputs with Open-Source PyMOL and return RMSD plus method-specific alignment metrics. The `method` configuration field accepts `cealign` or `align`; `target_selection` and `mobile_selection` can be used for advanced PyMOL selection strings.
