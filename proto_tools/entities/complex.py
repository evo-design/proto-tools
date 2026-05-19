"""Shared molecular-chain primitives reusable across tool categories."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.entities.ligands import is_valid_ccd_code
from proto_tools.entities.ligands.ccd_utils import (
    get_canonical_component,
    get_modifications_for_component,
)
from proto_tools.utils import detect_sequence_type


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
        chain_id (str | None): Optional chain identifier. ``None`` when the
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

    chain_id: str | None = Field(
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
