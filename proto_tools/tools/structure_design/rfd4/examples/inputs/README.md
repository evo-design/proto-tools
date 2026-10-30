# Example inputs

Inputs for `examples/example.ipynb`, taken verbatim from the upstream
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) repository (BSD-3-Clause)
so the notebook reproduces upstream's own worked examples.

| File | Upstream path | What it is |
|---|---|---|
| `pdl1_target_5o45_cropped.cif` | `docs/examples/inputs/5o45_cropped.cif.zst` | PD-L1 (PDB 5O45), chain A, residues 17-131. The binder-design target used by upstream's `condition_spec_ppi.yaml` and `binder-in-10-lines.md`. Decompressed from Zstandard. |
| `unconditional_monomer_100.cif` | `data/benchmarks/monomer/monomer-short.cif.zst` | The unconditional-generation benchmark input: a single collapsed expandable-segment sentinel (`res_name` `100-100`) carrying condition annotations, so it must be loaded with `input_as_annotated_cif=True`. Decompressed from Zstandard. |

<!-- docs:ignore start -->
`pdl1_target_5o45_cropped.cif` contains 15 atoms with NaN coordinates across 6 residues,
all of them unresolved crystallographic *sidechain* atoms (`ASN17` CG/OD1/ND2, `GLU45` OE2,
`LYS46` CD/NZ, `ARG82` CZ/NH1/NH2, `ARG125` CZ/NH1/NH2, `LYS129` CD/CE/NZ) — the backbone is
fully resolved. That is why upstream's own rigid-group selection filters with
`and not has_nan_coord()`; its three hotspot selections carry no such filter.
It is a binder *target*, not a suitable
inverse-folding input: upstream's own CLI returns a CA-only poly-alanine trace for it in
`generation=inverse_folding` mode, and our implementation reproduces that exactly.
<!-- docs:ignore end -->
