"""Shared molecular-chain primitives reusable across tool categories."""

import string
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.ligands import Fragment, Ligands, is_valid_ccd_code
from proto_tools.entities.ligands.ccd_utils import (
    get_canonical_component,
    get_modifications_for_component,
)
from proto_tools.entities.ligands.ligands import parse_smiles_string
from proto_tools.utils import detect_sequence_type

if TYPE_CHECKING:
    from proto_tools.entities.structures import Structure

CHAIN_IDS: Final[list[str]] = list(string.ascii_uppercase)


def chain_label(i: int) -> str:
    """Deterministic positional chain ID for the i-th chain: A..Z, AA..ZZ, AAA..., spreadsheet-style.

    Used by every consumer that needs a fallback chain ID when the entity's own ``id`` is None.
    Unbounded so complexes with more than 26 chains roll over to two-letter labels rather than
    crashing with ``IndexError``.

    Args:
        i (int): Non-negative chain index.

    Returns:
        str: Chain label (e.g. ``0 → "A"``, ``26 → "AA"``, ``701 → "ZZ"``, ``702 → "AAA"``).

    Raises:
        ValueError: If ``i`` is negative.
    """
    if i < 0:
        raise ValueError(f"chain index must be non-negative, got {i}")
    label = ""
    n = i
    while True:
        label = string.ascii_uppercase[n % 26] + label
        n = n // 26 - 1
        if n < 0:
            return label


class ChainModification(BaseModel):
    """Represents a modification to a specific position in a molecular chain.

    Modifications are specified using Chemical Component Dictionary (CCD) codes
    from the wwPDB.

    IMPORTANT NOTE: THE MODIFICATION POSITIONS USE 1-BASED INDEXING, WHICH
    FOLLOWS THE STANDARD BIOLOGICAL CONVENTIONS.

    Attributes:
        position (int): 1-based position in the sequence where the modification occurs.
            Must be greater than or equal to 1 and within the sequence length.

        modification_code (str): Chemical Component Dictionary (CCD) code identifying
            the modification. Commonly used examples for different entity types include:

            - Protein PTMs: "SEP" (phosphoserine), "TPO" (phosphothreonine),
              "HY3" (hydroxyproline), "P1L" (pyroglutamic acid)
            - RNA modifications: "2MG" (2'-O-methylguanosine), "5MC" (5-methylcytidine),
              "PSU" (pseudouridine)
            - DNA modifications: "6OG" (8-oxoguanine), "6MA" (N6-methyladenine)

    Examples:
        >>> # Phosphoserine at position 5
        >>> mod = ChainModification(position=5, modification_code="SEP")
        >>>
        >>> # 2'-O-methylguanosine at position 1 in RNA
        >>> mod = ChainModification(position=1, modification_code="2MG")

    Note:
        Position indexing is 1-based (first residue/base is position 1), following
        standard biological conventions and matching the format expected by most
        structure prediction tools.
    """

    position: int = Field(description="1-based position in the sequence where modification occurs")
    modification_code: str = Field(description="Chemical Component Dictionary (CCD) code for the modification")

    @field_validator("position")
    @classmethod
    def validate_position(cls, pos: int) -> int:
        """Validate that position is 1-based (>= 1)."""
        if pos < 1:
            raise ValueError(f"Position must be 1-based (>= 1). Got {pos}. Note: positions count from 1, not 0.")
        return pos

    @field_validator("modification_code")
    @classmethod
    def validate_modification_code(cls, code: str) -> str:
        """Validate that modification code is a valid CCD code."""
        code = code.strip()

        if not is_valid_ccd_code(code):
            raise ValueError(f"Invalid CCD code: {code}. Must be a valid CCD code.")

        return code


class Chain(BaseModel):
    """Represents a single molecular chain with optional modifications.

    A chain consists of a sequence (protein, DNA, RNA, or ligand) along with
    its entity type and any chemical modifications. This class supports both
    simple use cases (just a sequence) and complex cases with post-translational
    modifications or nucleotide modifications.

    Attributes:
        id (str | None): Optional chain identifier. ``None`` when the
            consumer assigns identifiers positionally; set explicitly when the
            chain's identity must be preserved (e.g. inverse-folding outputs
            keyed to an input structure's chains).

        sequence (str): The sequence of the chain. Format depends on entity_type:

            - Protein: Amino acid sequence in single-letter code (e.g., "MVLSPADKTN")
            - DNA: Nucleotide sequence (e.g., "ATCGATCG")
            - RNA: Nucleotide sequence (e.g., "AUGCAUGC")
            - Ligand: SMILES string or other model-specific format

        entity_type (str | None): Type of molecular entity. Valid options:
            ``"protein"``, ``"dna"``, ``"rna"``, ``"ligand"``. If ``None``,
            automatically inferred from sequence composition using
            ``detect_sequence_type()``.

        modifications (list[ChainModification]): List of
            modifications to apply to this chain. Each modification can be either:

            - A ChainModification object
            - A dict with ``position`` and ``modification_code`` keys (e.g. from JSON deserialization)
            - A tuple of (position, modification_code) for convenience

            All dicts and tuples are automatically converted to ChainModification objects.
            Default: empty list (no modifications).

    Examples:
        >>> # Simple protein chain (no modifications)
        >>> chain = Chain(sequence="MVLSPADKTN")
        >>>
        >>> # Protein with phosphorylation using ChainModification objects
        >>> chain = Chain(
        ...     sequence="MVLSPADKTN",
        ...     entity_type="protein",
        ...     modifications=[ChainModification(position=5, modification_code="SEP")],
        ... )
        >>>
        >>> # Protein with phosphorylation using tuples (more convenient)
        >>> chain = Chain(sequence="MVLSPADKTN", entity_type="protein", modifications=[(5, "SEP")])
        >>>
        >>> # RNA with multiple modifications using tuples
        >>> chain = Chain(sequence="AUGCAUGC", entity_type="rna", modifications=[(1, "2MG"), (4, "5MC")])

    Note:
        The entity_type is automatically inferred if not provided, but can be
        explicitly set for clarity or to override auto-detection. Modifications
        are validated to ensure they don't exceed the sequence length.
    """

    id: str | None = Field(
        default=None,
        description="Optional chain identifier; None when identifiers are assigned positionally.",
    )
    sequence: str = Field(description="Sequence of the chain (protein, DNA, RNA, or ligand SMILES)")
    entity_type: str | None = Field(
        default=None,
        description="Entity type: 'protein', 'dna', 'rna', or 'ligand'. Auto-inferred if None.",
    )
    modifications: list[ChainModification] = Field(
        default_factory=list, description="List of modifications to apply to this chain"
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, seq: str) -> str:
        """Validate that sequence is non-empty."""
        if not seq or not seq.strip():
            raise ValueError("Sequence cannot be empty")
        return seq

    @field_validator("modifications", mode="before")
    @classmethod
    def convert_modifications(cls, mods: Any) -> list[ChainModification]:
        """Convert tuples to ChainModification objects."""
        if not isinstance(mods, list):
            raise ValueError(f"modifications must be a list, got {type(mods)}")

        normalized_mods = []
        for idx, mod in enumerate(mods):
            if isinstance(mod, tuple):
                # Convert tuple (position, code) to ChainModification
                if len(mod) != 2:
                    raise ValueError(
                        f"Modification tuple at index {idx} must have exactly 2 elements "
                        f"(position, modification_code), got {len(mod)}"
                    )
                position, code = mod
                normalized_mods.append(ChainModification(position=position, modification_code=code))
            elif isinstance(mod, ChainModification):
                # Already a ChainModification object
                normalized_mods.append(mod)
            elif isinstance(mod, dict):
                try:
                    normalized_mods.append(ChainModification(**mod))
                except Exception as e:
                    raise ValueError(f"Modification dict at index {idx} is invalid: {e}") from e
            else:
                raise ValueError(
                    f"Modification at index {idx} must be a ChainModification object, "
                    f"a dict, or a tuple (position, modification_code). Got {type(mod)}"
                )

        return normalized_mods

    @model_validator(mode="after")
    def infer_entity_type(self) -> Any:
        """Auto-infer entity type if not provided."""
        if self.entity_type is None:
            self.entity_type = detect_sequence_type(self.sequence)
        return self

    @model_validator(mode="after")
    def validate_modifications(self) -> Any:
        """Ensure modifications are within sequence bounds and compatible with residues or bases."""
        seq_length = len(self.sequence)

        # If the sequence is a ligand, we can't have modifications
        if self.entity_type == "ligand":
            if self.modifications:
                raise ValueError(f"Ligands cannot have modifications. Found: {self.modifications}")
            return self

        for mod in self.modifications:
            # Check position is within bounds
            if mod.position > seq_length:
                raise ValueError(
                    f"Modification at position {mod.position} exceeds "
                    f"sequence length {seq_length} for chain with sequence: {self.sequence}"
                )
            if mod.position < 1:
                raise ValueError(
                    f"Modification at position {mod.position} is 0-based, "
                    f"but must be 1-based. Positions count from 1, not 0."
                )

            # Check that the modification is compatible with the residue or base at that position
            residue_or_base_char = self.sequence[mod.position - 1]  # Convert to 0-based indexing
            canonical_for_mod = get_canonical_component(mod.modification_code)

            # If the modification has a canonical parent, validate compatibility
            if canonical_for_mod is not None and residue_or_base_char.upper() != canonical_for_mod.upper():
                # Get allowed modifications for this residue or base to show in error
                allowed_mods = get_modifications_for_component(self.entity_type, residue_or_base_char.upper())  # type: ignore[arg-type]

                mods_str = ", ".join(allowed_mods) if allowed_mods else "none"

                raise ValueError(
                    f"Invalid modification '{mod.modification_code}' at position {mod.position}. "
                    f"This modification is for residue or base '{canonical_for_mod}', but position "
                    f"{mod.position} contains '{residue_or_base_char}'. "
                    f"Allowed modifications for '{residue_or_base_char}' in {self.entity_type}: {mods_str}"
                )

        return self

    def add_modification(self, position: int, modification_code: str) -> "Chain":
        """Add a modification to this chain.

        Args:
            position (int): 1-based position in the sequence
            modification_code (str): CCD code for the modification

        Returns:
            Chain: Self for method chaining

        Raises:
            ValueError: If position exceeds sequence length or modification is incompatible

        Examples:
            >>> chain = Chain(sequence="MVLSPADKTN")
            >>> chain.add_modification(4, "SEP")  # Position 4 is 'S' (serine)
            >>> chain.add_modification(9, "TPO")  # Position 9 is 'T' (threonine)
        """
        mod = ChainModification(position=position, modification_code=modification_code)

        # Validate position is within bounds
        if mod.position > len(self.sequence):
            raise ValueError(f"Modification at position {mod.position} exceeds sequence length {len(self.sequence)}")

        # Validate modification is compatible with residue or base at this position
        if self.entity_type != "ligand":
            residue_or_base_char = self.sequence[mod.position - 1]  # Convert to 0-based
            canonical_for_mod = get_canonical_component(mod.modification_code)

            if canonical_for_mod is not None and residue_or_base_char.upper() != canonical_for_mod.upper():
                # Get allowed modifications for this residue or base
                allowed_mods = get_modifications_for_component(self.entity_type, residue_or_base_char.upper())  # type: ignore[arg-type]

                mods_str = ", ".join(allowed_mods) if allowed_mods else "none"

                raise ValueError(
                    f"Invalid modification '{mod.modification_code}' at position {mod.position}. "
                    f"This modification is for residue or base '{canonical_for_mod}', but position "
                    f"{mod.position} contains '{residue_or_base_char}'. "
                    f"Allowed modifications for '{residue_or_base_char}' in {self.entity_type}: {mods_str}"
                )

        self.modifications.append(mod)
        return self

    def clear_modifications(self) -> "Chain":
        """Remove all modifications from this chain.

        Returns:
            Chain: Self for method chaining

        Examples:
            >>> chain = Chain(sequence="MVLSPADKTN")
            >>> chain.add_modification(5, "SEP")
            >>> chain.clear_modifications()
        """
        self.modifications.clear()
        return self

    def has_modifications(self) -> bool:
        """Check if this chain has any modifications."""
        return len(self.modifications) > 0

    def __len__(self) -> int:
        """Returns the length of the sequence."""
        return len(self.sequence)


def _normalize_chain_entry(chain: Any) -> list[Chain | Fragment]:
    """Normalize a single user-provided chain entry to a list of Chain/Fragment objects.

    Multi-fragment SMILES strings and ``Ligands`` collections expand into multiple
    ``Fragment`` objects; everything else produces a single-entry list.

    Args:
        chain (Any): String, dict, Chain, Fragment, or Ligands.

    Returns:
        list[Chain | Fragment]: Normalized chain entries (length >= 1).

    Raises:
        ValueError: If the input type is not supported.
    """
    if isinstance(chain, Fragment):
        return [chain]
    if isinstance(chain, Ligands):
        return list(chain.fragments)
    if isinstance(chain, Chain):
        if chain.entity_type == "ligand":
            return [Fragment(smiles=chain.sequence)]
        return [chain]
    if isinstance(chain, dict):
        if chain.get("entity_type") == "ligand" or "ccd_code" in chain or "smiles" in chain:
            data = {k: v for k, v in chain.items() if k != "entity_type"}
            # Translate biopolymer-style 'sequence' key to Fragment's 'smiles'
            if "sequence" in data and "smiles" not in data:
                data["smiles"] = data.pop("sequence")
            return [Fragment(**data)]
        # Build the Chain first so infer_entity_type runs; convert to Fragment if auto-classified as ligand.
        chain_obj = Chain(**chain)
        if chain_obj.entity_type == "ligand":
            return [Fragment(smiles=chain_obj.sequence)]
        return [chain_obj]
    if isinstance(chain, str):
        if detect_sequence_type(chain) == "ligand":
            return list(parse_smiles_string(chain))
        return [Chain(sequence=chain)]
    raise ValueError(f"Chain entry must be a string, dict, Chain, Fragment, or Ligands. Got {type(chain).__name__}")


class Complex(BaseModel):
    """An ordered multi-entity molecular complex.

    Represents one or more chains (biopolymer ``Chain`` and/or ligand ``Fragment``)
    in a deterministic input order, each carrying an optional ``id``. Used as the
    canonical multi-entity input to structure-prediction tools (whose inputs declare
    ``complexes: list[Complex]``) and as the base of the inverse-folding output
    type (``DesignedComplex``), so a design feeds a structure predictor directly
    via LSP.

    Construction is lenient. ``chains`` accepts:

    - A bare string (shorthand for a single-chain complex).
    - A list mixing strings (biopolymer sequences / SMILES — multi-fragment SMILES
      auto-splits into N ``Fragment`` entries), dicts (``Chain`` or ``Fragment``
      kwargs), ``Chain``/``Fragment`` instances, or ``Ligands`` collections
      (expanded into one ``Fragment`` per fragment).

    There is no chain-count cap; ``chain_label(i)`` generates positional fallback
    IDs A..Z, AA..ZZ, AAA... for chains without an explicit ``id``.

    Use ``Complex.from_structure(structure)`` to build a faithful ``Complex`` from
    an existing ``Structure`` (polymer chains + ligand HETATMs in input order).

    Attributes:
        chains (list[Chain | Fragment]): Chains in the complex, in input order.
    """

    model_config = ConfigDict(extra="forbid")

    chains: list[Chain | Fragment] = Field(
        description="Chains in the complex. Strings, dicts, Chain, Fragment, or Ligands."
    )

    @field_validator("chains", mode="before")
    @classmethod
    def normalize_chains(cls, chains: Any) -> list[Chain | Fragment]:
        """Normalize chains to a flat list of ``Chain`` (biopolymer) or ``Fragment`` (ligand) objects."""
        if isinstance(chains, str):
            chains = [chains]
        if not isinstance(chains, list):
            raise ValueError(f"chains must be a list, got {type(chains)}")

        normalized: list[Chain | Fragment] = []
        for chain_idx, chain in enumerate(chains):
            try:
                normalized.extend(_normalize_chain_entry(chain))
            except Exception as e:  # noqa: PERF203 -- preserve per-entry index in error
                raise ValueError(f"Chain {chain_idx} is invalid: {e}") from e

        return normalized

    @property
    def chain_sequences(self) -> list[str]:
        """Per-chain sequences in chain order; SMILES (or CCD code) for Fragments."""
        out: list[str] = []
        for chain in self.chains:
            if isinstance(chain, Fragment):
                out.append(chain.smiles or chain.ccd_code or "")
            else:
                out.append(chain.sequence)
        return out

    @property
    def entity_types(self) -> list[str]:
        """Entity types for all chains in the complex, in chain order."""
        return [chain.entity_type for chain in self.chains]  # type: ignore[misc]

    def num_chains(self) -> int:
        """Number of chain entries in the complex."""
        return len(self.chains)

    def sum_of_chain_lengths(self) -> int:
        """Total residue count across chains; each Fragment ligand contributes 1."""
        return sum(1 if isinstance(chain, Fragment) else len(chain.sequence) for chain in self.chains)

    def get_entity_type_set(self) -> set[str]:
        """Set of unique entity types in the complex."""
        return set(self.entity_types)

    def as_chain_map(self) -> dict[str, str]:
        """Mapping of ``id`` to sequence (or SMILES/CCD for ligands); entries without ``id`` are skipped."""
        out: dict[str, str] = {}
        for c in self.chains:
            if c.id is None:
                continue
            if isinstance(c, Fragment):
                out[c.id] = c.smiles or c.ccd_code or ""
            else:
                out[c.id] = c.sequence
        return out

    @classmethod
    def from_structure(cls, structure: "Structure") -> "Complex":
        """Build a faithful Complex from a Structure: all polymer chains and ligand residues in input order.

        Polymer chains become ``Chain`` (entity_type auto-inferred from sequence:
        protein, DNA, or RNA). Ligand HETATM residues become ``Fragment`` entries
        keyed by their CCD-code residue names. Order follows the input file's
        chain order; ligand residues within a polymer chain are emitted after
        that chain in their seqid order. Each entity's ``id`` is set from the
        structure's chain identifier.

        Args:
            structure (Structure): Input structure to enumerate.

        Returns:
            Complex: A complex whose chains reproduce the input structure's
                entities. Use ``DesignedComplex`` for inverse-folding outputs.
        """
        entries: list[Chain | Fragment] = []
        chain_types = structure.get_chain_types()
        for chain_id, chain_type in chain_types.items():
            if chain_type == "polymer":
                sequence = structure.get_chain_sequence(chain_id)
                if sequence:
                    entries.append(Chain(id=chain_id, sequence=sequence))
            for resname, _seqid in structure.get_chain_ligands(chain_id):
                entries.append(Fragment(id=chain_id, ccd_code=resname))
        return cls(chains=entries)

    def extract_protein_chains(self) -> tuple[list[str], list[str]]:
        """Extract protein sequences and their chain identifiers.

        Returns one entry per protein ``Chain`` in input order. Each chain's
        identifier is its ``id`` if set, otherwise a spreadsheet-style label
        (``chain_label(i)``: A..Z, AA..ZZ, AAA...) keyed to the chain's index
        in ``self.chains`` so complexes with more than 26 chains roll over to
        multi-letter IDs instead of crashing.

        Returns:
            tuple[list[str], list[str]]: Tuple of (protein_sequences, protein_chain_ids).
        """
        protein_seqs: list[str] = []
        protein_chain_ids: list[str] = []
        for i, chain in enumerate(self.chains):
            if isinstance(chain, Chain) and chain.entity_type == "protein":
                protein_seqs.append(chain.sequence)
                protein_chain_ids.append(chain.id if chain.id is not None else chain_label(i))
        return protein_seqs, protein_chain_ids

    def add_modification_to_chain(self, chain_index: int, position: int, modification_code: str) -> "Complex":
        """Add a modification to a specific biopolymer chain in the complex.

        Args:
            chain_index (int): 0-based index of the chain to modify.
            position (int): 1-based position in the chain sequence.
            modification_code (str): CCD code for the modification.

        Returns:
            Complex: Self for method chaining.

        Raises:
            IndexError: If chain_index is out of bounds.
            ValueError: If the target chain is a ligand Fragment (which cannot have modifications).
        """
        if chain_index < 0 or chain_index >= len(self.chains):
            raise IndexError(
                f"Chain index {chain_index} out of bounds. "
                f"Complex has {len(self.chains)} chains (indices 0-{len(self.chains) - 1})"
            )
        target = self.chains[chain_index]
        if isinstance(target, Fragment):
            raise ValueError(f"Chain {chain_index} is a ligand Fragment and cannot have modifications")
        target.add_modification(position, modification_code)
        return self

    def clear_all_modifications(self) -> "Complex":
        """Remove all modifications from all biopolymer chains.

        Returns:
            Complex: Self for method chaining.
        """
        for chain in self.chains:
            if isinstance(chain, Chain):
                chain.clear_modifications()
        return self

    def has_modifications(self) -> bool:
        """Whether any biopolymer chain in this complex has any modifications."""
        return any(isinstance(chain, Chain) and chain.has_modifications() for chain in self.chains)

    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()

    def __str__(self) -> str:
        """String representation: class name and chain count."""
        return f"{type(self).__name__}(with {self.num_chains()} chains)"

    def __iter__(self) -> Iterator[Chain | Fragment]:  # type: ignore[override]
        """Iterate over the chains."""
        return iter(self.chains)

    def __getitem__(self, index: int) -> Chain | Fragment:
        """Get a chain by index."""
        return self.chains[index]
