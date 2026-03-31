"""proto_tools/tools/structure_prediction/shared_data_models.py

Shared data models (configs and outputs) for structure prediction tools."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Iterator, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.entities.ligands import is_valid_ccd_code
from proto_tools.entities.ligands.ccd_utils import (
    get_canonical_component,
    get_modifications_for_component,
)
from proto_tools.entities.structures import Structure
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
)
from proto_tools.tools.sequence_alignment.msas import MSA
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    detect_sequence_type,
)

logger = logging.getLogger(__name__)


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

    position: int = Field(
        description="1-based position in the sequence where modification occurs"
    )
    modification_code: str = Field(
        description="Chemical Component Dictionary (CCD) code for the modification"
    )

    @field_validator("position")
    @classmethod
    def validate_position(cls, pos: int) -> int:
        """Validate that position is 1-based (>= 1)."""
        if pos < 1:
            raise ValueError(
                f"Position must be 1-based (>= 1). Got {pos}. "
                "Note: positions count from 1, not 0."
            )
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
            - A tuple of (position, modification_code) for convenience

            All tuples are automatically converted to ChainModification objects.
            Default: empty list (no modifications).

    Examples:
        >>> # Simple protein chain (no modifications)
        >>> chain = Chain(sequence="MVLSPADKTN")
        >>>
        >>> # Protein with phosphorylation using ChainModification objects
        >>> chain = Chain(
        ...     sequence="MVLSPADKTN",
        ...     entity_type="protein",
        ...     modifications=[
        ...         ChainModification(position=5, modification_code="SEP")
        ...     ]
        ... )
        >>>
        >>> # Protein with phosphorylation using tuples (more convenient)
        >>> chain = Chain(
        ...     sequence="MVLSPADKTN",
        ...     entity_type="protein",
        ...     modifications=[(5, "SEP")]
        ... )
        >>>
        >>> # RNA with multiple modifications using tuples
        >>> chain = Chain(
        ...     sequence="AUGCAUGC",
        ...     entity_type="rna",
        ...     modifications=[(1, "2MG"), (4, "5MC")]
        ... )

    Note:
        The entity_type is automatically inferred if not provided, but can be
        explicitly set for clarity or to override auto-detection. Modifications
        are validated to ensure they don't exceed the sequence length.
    """

    sequence: str = Field(
        description="Sequence of the chain (protein, DNA, RNA, or ligand SMILES)"
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Entity type: 'protein', 'dna', 'rna', or 'ligand'. Auto-inferred if None.",
    )
    modifications: List[ChainModification] = Field(
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
    def convert_modifications(cls, mods) -> List[ChainModification]:
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
            else:
                raise ValueError(
                    f"Modification at index {idx} must be a ChainModification object or "
                    f"a tuple (position, modification_code). Got {type(mod)}"
                )

        return normalized_mods

    @model_validator(mode="after")
    def infer_entity_type(self):
        """Auto-infer entity type if not provided."""
        if self.entity_type is None:
            self.entity_type = detect_sequence_type(self.sequence)
        return self

    @model_validator(mode="after")
    def validate_modifications(self):
        """Ensure modifications are within sequence bounds and compatible with residues or bases."""
        seq_length = len(self.sequence)

        # If the sequence is a ligand, we can't have modifications
        if self.entity_type == "ligand":
            if self.modifications:
                raise ValueError(
                    f"Ligands cannot have modifications. Found: {self.modifications}"
                )
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
            if canonical_for_mod is not None:
                if residue_or_base_char.upper() != canonical_for_mod.upper():
                    # Get allowed modifications for this residue or base to show in error
                    allowed_mods = get_modifications_for_component(
                        self.entity_type, residue_or_base_char.upper()
                    )

                    if allowed_mods:
                        mods_str = ", ".join(allowed_mods)
                    else:
                        mods_str = "none"

                    raise ValueError(
                        f"Invalid modification '{mod.modification_code}' at position {mod.position}. "
                        f"This modification is for residue or base '{canonical_for_mod}', but position "
                        f"{mod.position} contains '{residue_or_base_char}'. "
                        f"Allowed modifications for '{residue_or_base_char}' in {self.entity_type}: {mods_str}"
                    )

        return self

    def add_modification(
        self, position: int, modification_code: str
    ) -> Chain:
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
            raise ValueError(
                f"Modification at position {mod.position} exceeds "
                f"sequence length {len(self.sequence)}"
            )

        # Validate modification is compatible with residue or base at this position
        if self.entity_type != "ligand":
            residue_or_base_char = self.sequence[mod.position - 1]  # Convert to 0-based
            canonical_for_mod = get_canonical_component(mod.modification_code)

            if canonical_for_mod is not None:
                if residue_or_base_char.upper() != canonical_for_mod.upper():
                    # Get allowed modifications for this residue or base
                    allowed_mods = get_modifications_for_component(
                        self.entity_type, residue_or_base_char.upper()
                    )

                    if allowed_mods:
                        mods_str = ", ".join(allowed_mods)
                    else:
                        mods_str = "none"

                    raise ValueError(
                        f"Invalid modification '{mod.modification_code}' at position {mod.position}. "
                        f"This modification is for residue or base '{canonical_for_mod}', but position "
                        f"{mod.position} contains '{residue_or_base_char}'. "
                        f"Allowed modifications for '{residue_or_base_char}' in {self.entity_type}: {mods_str}"
                    )

        self.modifications.append(mod)
        return self

    def clear_modifications(self) -> Chain:
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


class StructurePredictionComplex(BaseModel):
    """Represents a single biological complex for structure prediction.

    A complex is defined as one or more molecular chains (proteins, nucleic acids,
    or ligands) that are predicted together in a single 3D structure. Chains within
    a complex are expected to interact, and their relative orientations and interfaces
    are modeled together.

    Attributes:
        chains (list[Chain]): Chains in the complex. Each chain can be:

            - A string sequence (automatically converted to Chain object)
            - A Chain object (with optional modifications)
            - A dictionary with Chain fields (``sequence``, ``entity_type``, ``modifications``)

            After validation, all chains are stored as Chain objects internally.

            Supported sequence types:
            - Protein sequences (amino acids in single-letter code)
            - DNA sequences (nucleotide bases: A, T, C, G)
            - RNA sequences (nucleotide bases: A, U, C, G)
            - Ligand representations (SMILES strings)

            All chains in the same complex are predicted together to model their
            interactions and relative 3D arrangement.

    Examples:
        >>> # Simple usage with strings (backward compatible)
        >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "ACDEFGHIKL"])
        >>>
        >>> # Explicit entity types via Chain objects
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         Chain(sequence="MVLSPADKTN", entity_type="protein"),
        ...         Chain(sequence="ATCG", entity_type="dna")
        ...     ]
        ... )
        >>>
        >>> # Mix strings and Chain objects
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         "MVLSPADKTN",
        ...         Chain(
        ...             sequence="ACDEFGHIKL",
        ...             modifications=[(3, "SEP")]
        ...         )
        ...     ]
        ... )
        >>>
        >>> # Chain objects with modifications using tuples (convenient)
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         Chain(
        ...             sequence="MVLSPADKTN",
        ...             entity_type="protein",
        ...             modifications=[(5, "HY3"), (10, "TPO")]
        ...         ),
        ...         Chain(
        ...             sequence="AUGCAUGC",
        ...             entity_type="rna",
        ...             modifications=[(1, "2MG")]
        ...         )
        ...     ]
        ... )
        >>>
        >>> # Using dictionaries (most flexible)
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         {
        ...             "sequence": "MVLSPADKTN",
        ...             "entity_type": "protein",
        ...             "modifications": [(5, "HY3"), (10, "TPO")]
        ...         },
        ...         {
        ...             "sequence": "AUGCAUGC",
        ...             "entity_type": "rna",
        ...             "modifications": [(1, "2MG")]
        ...         }
        ...     ]
        ... )
        >>>
        >>> # Mix all formats (strings, dicts, Chain objects)
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         "MVLSPADKTN",  # String
        ...         {"sequence": "ATCG", "entity_type": "dna"},  # Dictionary
        ...         Chain(sequence="AUGC", entity_type="rna")  # Chain object
        ...     ]
        ... )
        >>>
        >>> # Access entity types
        >>> complex.entity_types  # Returns ["protein", "dna", "rna"]
        >>> complex.get_entity_type_set()  # Returns {"protein", "dna", "rna"}

    Note:
        Each ``StructurePredictionComplex`` instance represents exactly one structure
        prediction input. Chains are always normalized to Chain objects during
        validation for consistent downstream processing. Entity types are stored
        at the Chain level and accessed via the entity_types property.
    """

    chains: List[Chain] = Field(
        description="Chains in the complex. Can be strings, Chain objects, or dictionaries with chain fields."
    )

    @model_validator(mode="before")
    @classmethod
    def reject_entity_types_parameter(cls, data):
        """Reject the entity_types parameter with a clear error message."""
        if isinstance(data, dict) and "entity_types" in data:
            raise ValueError(
                "'entity_types' parameter is not accepted. "
                "Specify entity types at the Chain level instead:\n"
                "  StructurePredictionComplex(chains=[Chain(sequence='MVLS...', entity_type='protein')]) or\n"
                "  StructurePredictionComplex(chains=[{'sequence': 'MVLS...', 'entity_type': 'protein'}])"
            )
        return data

    @field_validator("chains", mode="before")
    @classmethod
    def normalize_chains(cls, chains) -> List[Chain]:
        """Normalize chains to Chain objects, converting strings and dicts as needed."""
        if not isinstance(chains, list):
            raise ValueError(f"chains must be a list, got {type(chains)}")

        normalized_chains = []
        for chain_idx, chain in enumerate(chains):
            if isinstance(chain, str):
                # Convert string to Chain object
                normalized_chains.append(Chain(sequence=chain))
            elif isinstance(chain, dict):
                # Convert dictionary to Chain object
                try:
                    normalized_chains.append(Chain(**chain))
                except Exception as e:
                    raise ValueError(
                        f"Chain {chain_idx} dictionary is invalid: {e}"
                    )
            elif isinstance(chain, Chain):
                # Already a Chain object
                normalized_chains.append(chain)
            else:
                raise ValueError(
                    f"Chain {chain_idx} must be a string, dictionary, or Chain object. "
                    f"Got {type(chain)}"
                )

        return normalized_chains

    @property
    def chain_sequences(self) -> List[str]:
        """Get the sequences of all chains in the complex."""
        return [chain.sequence for chain in self.chains]

    @property
    def entity_types(self) -> List[str]:
        """Get the entity types for all chains in the complex.

        Returns:
            list[str]: List of entity types, one for each chain.
        """
        return [chain.entity_type for chain in self.chains]

    def sum_of_chain_lengths(self) -> int:
        """Get the sum of the lengths of all chains in the complex."""
        return sum(len(chain.sequence) for chain in self.chains)

    def num_chains(self) -> int:
        """Get the number of chains in the complex."""
        return len(self.chains)

    def get_entity_type_set(self) -> Set[str]:
        """Get the set of unique entity types in the complex."""
        return set(self.entity_types)

    def add_modification_to_chain(
        self, chain_index: int, position: int, modification_code: str
    ) -> StructurePredictionComplex:
        """Add a modification to a specific chain in the complex.

        Args:
            chain_index (int): 0-based index of the chain to modify
            position (int): 1-based position in the chain sequence
            modification_code (str): CCD code for the modification

        Returns:
            StructurePredictionComplex: Self for method chaining

        Raises:
            IndexError: If chain_index is out of bounds
            ValueError: If position exceeds chain sequence length

        Examples:
            >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "ATCGATCG"])
            >>> complex.add_modification_to_chain(0, 5, "SEP")
            >>> complex.add_modification_to_chain(0, 10, "TPO")
        """
        if chain_index < 0 or chain_index >= len(self.chains):
            raise IndexError(
                f"Chain index {chain_index} out of bounds. "
                f"Complex has {len(self.chains)} chains (indices 0-{len(self.chains)-1})"
            )

        self.chains[chain_index].add_modification(position, modification_code)
        return self

    def clear_all_modifications(self) -> StructurePredictionComplex:
        """Remove all modifications from all chains in the complex.

        Returns:
            StructurePredictionComplex: Self for method chaining

        Examples:
            >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "ATCGATCG"])
            >>> complex.add_modification_to_chain(0, 5, "SEP")
            >>> complex.clear_all_modifications()
        """
        for chain in self.chains:
            chain.clear_modifications()
        return self

    def has_modifications(self) -> bool:
        """Check if any chain in this complex has any modifications."""
        return any(chain.has_modifications() for chain in self.chains)

    def __repr__(self) -> str:
        """Get a string representation of the complex."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the complex."""
        return f"StructurePredictionComplex(with {self.num_chains()} chains)"

    def __iter__(self) -> Iterator[Chain]:
        """Iterate over the chains in the complex."""
        return iter(self.chains)

    def __getitem__(self, index: int) -> Chain:
        """Get a chain by index."""
        return self.chains[index]


class StructurePredictionInput(BaseToolInput):
    """Input object for structure prediction models.

    This class provides a flexible interface for specifying one or more biological
    complexes for structure prediction. After validation, always contains a list of
    ``StructurePredictionComplex`` instances.

    Supports multiple input formats:

    - List of ``StructurePredictionComplex`` instances (explicit format)
    - List of sequence strings (each treated as a single-chain complex)
    - List of lists of sequences (each sublist treated as a multi-chain complex)

    Attributes:
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Each complex contains one or more chains with their
            corresponding entity types. After validation, always a list of
            ``StructurePredictionComplex`` instances regardless of input format.

        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence
            string. Populated by ``Config.preprocess()`` via ColabFold search, or
            supplied directly to skip MSA generation. Default: ``None``.

        SUPPORTED_ENTITY_TYPES: Set of entity types supported by this tool.
            Must be defined by subclasses. Valid options: "protein", "dna", "rna",
            "ligand", "glycan".

    Examples:
        >>> # Single-chain complexes (entity types auto-inferred)
        >>> inputs = StructurePredictionInput(
        ...     complexes=["MVLSPADKTN", "ACDEFGHIKL"]
        ... )
        >>>
        >>> # Multi-chain complex (entity types auto-inferred)
        >>> inputs = StructurePredictionInput(
        ...     complexes=[["MVLSPADKTN", "ACDEFGHIKL"]]
        ... )
        >>>
        >>> # Explicit format with entity types using Chain objects
        >>> complex1 = StructurePredictionComplex(
        ...     chains=[
        ...         Chain(sequence="MVLSPADKTN", entity_type="protein"),
        ...         Chain(sequence="ATCGATCG", entity_type="dna")
        ...     ]
        ... )
        >>> inputs = StructurePredictionInput(complexes=[complex1])
        >>>
        >>> # Explicit format with entity types using dictionaries
        >>> complex2 = StructurePredictionComplex(
        ...     chains=[
        ...         {"sequence": "MVLSPADKTN", "entity_type": "protein"},
        ...         {"sequence": "ATCGATCG", "entity_type": "dna"}
        ...     ]
        ... )
        >>> inputs = StructurePredictionInput(complexes=[complex2])

    Note:
        All input formats are automatically normalized to ``List[StructurePredictionComplex]``
        during validation for consistent downstream processing.

        Subclasses must define SUPPORTED_ENTITY_TYPES to specify which entity types
        are supported by the structure prediction tool.
    """

    # Subclasses must define this
    SUPPORTED_ENTITY_TYPES: ClassVar[Set[str]]
    ALLOWS_CHAIN_MODIFICATIONS: ClassVar[bool]

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "SUPPORTED_ENTITY_TYPES"):
            raise TypeError(
                f"{cls.__name__} must define SUPPORTED_ENTITY_TYPES class attribute. "
                f"Example: SUPPORTED_ENTITY_TYPES = {{'protein', 'dna', 'rna'}}"
            )

        if not hasattr(cls, "ALLOWS_CHAIN_MODIFICATIONS"):
            raise TypeError(
                f"{cls.__name__} must define ALLOWS_CHAIN_MODIFICATIONS class attribute. "
                f"Example: ALLOWS_CHAIN_MODIFICATIONS = True"
            )

    complexes: List[StructurePredictionComplex] = InputField(
        description="List of complexes to predict structure for containing chains and entity types."
    )
    msas: dict[str, MSA] | None = InputField(
        default=None,
        description="Pre-computed MSAs keyed by sequence. Populated by preprocess() or supplied directly.",
        hidden=True,
    )

    @classmethod
    def item_cost(cls, item: StructurePredictionComplex) -> float:
        """Cost ~ total residues across all chains in the complex."""
        return float(item.sum_of_chain_lengths())

    @field_validator("complexes", mode="before", check_fields=False)
    @classmethod
    def normalize_complexes(cls, value):
        if value is None:
            raise ValueError("Input cannot be None")

        # If the value is not a list, convert it to a list
        if not isinstance(value, list):
            value = [value]

        final_complexes = []
        for sp_complex in value:

            # If item is a StructurePredictionComplex, add it to the final list
            if isinstance(sp_complex, StructurePredictionComplex):
                final_complexes.append(sp_complex)
            # If item is a string, treat it as a single-chain complex
            elif isinstance(sp_complex, str):
                final_complexes.append(StructurePredictionComplex(chains=[sp_complex]))
            # If item is a list that contains strings, treat it as a multi-chain complex
            elif isinstance(sp_complex, list) and all(isinstance(x, str) for x in sp_complex):
                final_complexes.append(StructurePredictionComplex(chains=sp_complex))
            else:
                raise ValueError(
                    f"Unsupported input format for auto-normalizing complexes: {sp_complex}"
                    "Expected type: StructurePredictionComplex, str, or list of strings"
                )

        # VALIDATION ==========================================================
        # Validate entity types are supported by this tool
        for comp_idx, comp in enumerate(final_complexes):
            entity_types = comp.get_entity_type_set()
            unsupported = entity_types - cls.SUPPORTED_ENTITY_TYPES

            if unsupported:
                raise ValueError(
                    f"Complex {comp_idx} contains unsupported entity types: "
                    f"{', '.join(sorted(unsupported))}. "
                    f"{cls.__name__} only supports: "
                    f"{', '.join(sorted(cls.SUPPORTED_ENTITY_TYPES))}"
                )

            if not cls.ALLOWS_CHAIN_MODIFICATIONS:
                if comp.has_modifications():
                    raise ValueError(
                        f"Complex {comp_idx} contains modifications. "
                        f"{cls.__name__} does not allow chain modifications."
                    )

        return final_complexes

    def __len__(self) -> int:
        """Get the number of complexes."""
        return len(self.complexes)

    def __getitem__(self, index: int) -> StructurePredictionComplex:
        """Get a complex by index."""
        return self.complexes[index]

    def __iter__(self) -> Iterator[StructurePredictionComplex]:
        """Iterate over the complexes."""
        return iter(self.complexes)

    def __repr__(self) -> str:
        """Get a string representation of the complexes."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the complexes."""
        return f"StructurePredictionInput(complexes={self.complexes})"

    def get_entity_type_set(self) -> Set[str]:
        """Get the set of unique entity types in the input across all complexes."""
        entity_type_set = set()
        for complex in self.complexes:
            entity_type_set.update(complex.get_entity_type_set())
        return entity_type_set


class StructurePredictionConfig(BaseConfig):
    """Base configuration for structure prediction models.

    This class provides common configuration parameters for all structure prediction
    tools. Individual prediction models should inherit from this class and add
    model-specific parameters.

    Attributes:
        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), or specific GPU devices like ``"cuda:0"``.
            Structure prediction is computationally intensive and strongly benefits
            from GPU acceleration. Default: ``"cuda"``.

    """
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )


class MSAStructurePredictionConfig(StructurePredictionConfig):
    """Configuration for structure prediction models that support MSA-based inference.

    Extends ``StructurePredictionConfig`` with optional MSA generation via ColabFold
    search. Tools that support MSA preprocessing should inherit from this class.

    Attributes:
        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. If ``False``, runs in single-sequence
            mode without MSAs. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``.
            Default: Uses ColabfoldSearchConfig defaults.
    """

    use_msa: bool = ConfigField(
        title="Use MSA",
        default=True,
        description="Whether to generate and use MSAs for protein chains using ColabFold search",
    )
    colabfold_search_config: Optional[ColabfoldSearchConfig] = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Nested configuration for ColabFold MSA search. If None, uses default settings.",
        hidden=True,
    )

    def preprocess(self, inputs: StructurePredictionInput) -> StructurePredictionInput:
        if not self.use_msa:
            return inputs
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return _preprocess_structure_prediction_msas(
            inputs, self.colabfold_search_config, self.verbose
        )


class StructurePredictionOutput(BaseToolOutput):
    """Output from structure prediction models.

    This class encapsulates the results of structure prediction, containing predicted
    3D structures for one or more input complexes.

    Attributes:
        structures (list[Structure]): List of predicted structures, one per
            input complex. Each structure contains the 3D coordinates in CIF format
            along with model-specific confidence metrics. The order matches the input
            complexes order.

    Note:
        This class supports list-like operations for convenient access to individual
        structures: indexing (``output[0]``), iteration (``for struct in output``),
        and length (``len(output)``).
    """

    structures: List[Structure] = Field(description="List of predicted structures")

    def __len__(self) -> int:
        """Get the number of predicted structures."""
        return len(self.structures)

    def __getitem__(self, index: int) -> Structure:
        """Get a predicted structure by index."""
        return self.structures[index]

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over the predicted structures."""
        return iter(self.structures)

    def __repr__(self) -> str:
        """Get a string representation of the predicted structures."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the predicted structures."""
        return f"StructurePredictionOutput(structures={self.structures})"

    # ===============================
    # Export Options
    # ===============================
    @property
    def output_format_options(self) -> List[str]:
        """List of valid file formats for exporting the tool output."""
        return ["cif", "pdb"]

    @property
    def output_format_default(self) -> str:
        """Default file format for exporting the tool output."""
        return "pdb"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        """Export the tool output to a file or directory of files."""
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for structure_idx, structure in enumerate(self.structures):
            if file_format == "pdb":
                structure.write_pdb(path / f"structure_{structure_idx}.pdb")
            elif file_format == "cif":
                structure.write_cif(path / f"structure_{structure_idx}.cif")
            else:
                raise ValueError(f"Invalid file format: {file_format}")


# ============================================================================
# Shared preprocessing
# ============================================================================
def _preprocess_structure_prediction_msas(
    inputs: StructurePredictionInput,
    colabfold_search_config: Any,
    verbose: bool = False,
) -> StructurePredictionInput:
    """Generate MSAs for all unique protein sequences and attach to inputs.

    Collects unique protein sequences across all complexes, runs ColabFold
    search once, and stores the raw MSA objects on ``inputs.msas`` keyed by
    sequence string. Skips ColabFold if MSAs are already supplied.

    Args:
        inputs (StructurePredictionInput): Structure prediction input containing complexes.
        colabfold_search_config (Any): ColabFold search configuration.
        verbose (bool): Whether to log progress.

    Returns:
        StructurePredictionInput: Updated inputs with ``msas`` field populated.
    """
    if inputs.msas is not None:
        if verbose:
            logger.info(
                f"Using {len(inputs.msas)} pre-supplied MSA(s), "
                f"skipping ColabFold search"
            )
        return inputs

    from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
        ColabfoldSearchInput,
        run_colabfold_search,
    )

    # Collect unique protein sequences across all complexes
    unique_seqs: dict[str, str] = {}  # sequence -> query name
    for comp in inputs.complexes:
        for chain in comp.chains:
            if chain.entity_type == "protein" and chain.sequence not in unique_seqs:
                unique_seqs[chain.sequence] = f"seq_{len(unique_seqs)}"

    if not unique_seqs:
        return inputs

    if verbose:
        logger.info(
            f"Generating MSAs for {len(unique_seqs)} unique protein sequence(s) "
            f"using ColabFold search..."
        )

    queries = [(seq, name) for seq, name in unique_seqs.items()]
    colabfold_input = ColabfoldSearchInput(queries=queries)
    colabfold_output = run_colabfold_search(colabfold_input, colabfold_search_config)

    # Build MSA dict keyed by sequence
    name_to_seq = {v: k for k, v in unique_seqs.items()}
    msas: dict[str, MSA] = {}
    for result in colabfold_output.results:
        if result.msa is not None:
            seq = name_to_seq[result.sequence_id]
            msas[seq] = result.msa
            if verbose:
                logger.info(
                    f"Generated MSA for {result.sequence_id}: "
                    f"{result.num_homologs_found} homologs found"
                )
        else:
            if verbose:
                logger.warning(f"No homologs found for {result.sequence_id}")

    return inputs.model_copy(update={"msas": msas}) if msas else inputs
