# Structures and Complexes

A `Structure` holds coordinates; a `Complex` holds the entities a tool reads or writes: `Chain`s for polymers and `Fragment`s for ligands. This note covers how one becomes the other, so every tool that turns a structure into chains produces the same chains.

## Reading chains from a structure

`Structure.get_chain(chain_id)` is the one residue-level reader. Everything else that needs a chain's residues goes through it:
- `get_chain_sequence(s)`
- `Complex.from_structure`
- `DesignedComplex.from_design`

- **Sequence:** one letter per observed polymer residue. It contains no gap markers and no lowercase letters, so `len(chain.sequence) == len(structure.get_chain_positions(chain_id))`.
- **Modified residues:** written as the parent residue's letter plus a `ChainModification` carrying the CCD code, with the parent taken from `get_canonical_component`. For example, `MSE` becomes `M` + `(i, "MSE")`.
- **Residues with no known parent:** a D-amino acid such as `DAL`, for example, becomes `X` + `(i, "DAL")`, so the identity survives. Only `UNK` and names outside the CCD become a bare `X`.
- **Entity type:** the polymer type gemmi assigns the chain (`protein`, `dna`, `rna`), never inferred from the letters. A glycine-only backbone is therefore a protein.
- **Breaks:** `get_chain_breaks(chain_id)` gives the 1-based sequence positions after which consecutive residues are not bonded, for example around an unresolved loop. Breaks are not stored on `Chain`.
- **Split chains:** a chain whose ligands are listed after the other chains under the same chain ID is merged back into one chain when the structure is parsed.

Predictors that set `ALLOWS_CHAIN_MODIFICATIONS = False` reject a complex carrying modifications. To predict the unmodified sequence instead, call `complex.clear_all_modifications()`.

## Inverse-folding designs

Every inverse-folding sample tool builds its outputs with `DesignedComplex.from_design(input_structure, sequences, designed, tool=..., include_ligands=...)`:

- **Every chain starts from the input structure,** read by `Complex.from_structure`. Context chains are therefore identical whichever model produced the design.
- **Each chain in `sequences` takes the model's sequence** as an unmodified protein chain.
- **A length mismatch raises.** The model's sequence must match the chain's observed residue count. Models skip residues they cannot read, such as residues missing backbone atoms or HETATM modified residues, and such a sequence cannot be laid back onto the structure.
- **`include_ligands` is true only for models that condition on ligands** (LigandMPNN). Ligands are always undesigned context.

By default a tool redesigns the structure's protein chains. DNA and RNA chains join every design as context.
