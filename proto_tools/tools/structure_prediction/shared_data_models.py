"""proto_tools/tools/structure_prediction/shared_data_models.py.

Shared data models (configs and outputs) for structure prediction tools.
"""

import logging
import string
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.entities.complex import Chain as Chain
from proto_tools.entities.complex import ChainModification as ChainModification
from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.entities.ligands.ligands import parse_smiles_string
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

CHAIN_IDS: Final[list[str]] = list(string.ascii_uppercase)


class StructurePredictionComplex(BaseModel):
    """Represents a single biological complex for structure prediction.

    A complex is defined as one or more molecular chains (proteins, nucleic acids,
    or ligands) that are predicted together in a single 3D structure. Chains within
    a complex are expected to interact, and their relative orientations and interfaces
    are modeled together.

    Biopolymer chains (protein, DNA, RNA) are represented as ``Chain`` objects.
    Ligand chains are represented as ``Fragment`` objects. Both expose
    ``entity_type`` so iteration code can branch uniformly:

    >>> for chain in complex.chains:
    ...     if chain.entity_type == "ligand":
    ...         smi, ccd = chain.smiles, chain.ccd_code
    ...     else:
    ...         seq, mods = chain.sequence, chain.modifications

    Attributes:
        chains (list[Chain | Fragment]): Chains in the complex. Each input element
            is normalized to a single ``Chain`` (biopolymer) or ``Fragment``
            (ligand). Accepted input forms:

            - String sequence: protein/DNA/RNA → ``Chain``; SMILES → ``Fragment``
              (multi-fragment SMILES like ``"ATP.MG"`` auto-split into N Fragments).
            - Dict with ``sequence``/``entity_type``/... → ``Chain``;
              dict with ``smiles``/``ccd_code`` → ``Fragment``.
            - ``Chain`` object → used as-is (a ligand-typed Chain is converted to
              a Fragment).
            - ``Fragment`` object → used as-is.
            - ``Ligands`` collection → expanded into one Fragment per fragment.

    Examples:
        >>> # Strings — biopolymers and ligands all auto-detected
        >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "CC(C)C"])
        >>>
        >>> # Mix Chain, Fragment, and Ligands
        >>> complex = StructurePredictionComplex(
        ...     chains=[
        ...         Chain(sequence="MVLSPADKTN", entity_type="protein"),
        ...         Fragment(ccd_code="ATP"),
        ...         Ligands(ccd_codes=["MG", "MG"]),  # expands to 2 chains
        ...     ]
        ... )
        >>>
        >>> # Multi-fragment SMILES auto-splits at the chains level
        >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "ATP.MG.MG"])
        >>>
        >>> # Access entity types
        >>> complex.entity_types  # Returns ["protein", "ligand", "ligand", "ligand"]

    Note:
        Each ``StructurePredictionComplex`` instance represents exactly one
        structure-prediction input. The 26-chain cap (``len(CHAIN_IDS)``) is
        checked after expansion of ``Ligands`` and multi-fragment SMILES.
    """

    chains: list[Chain | Fragment] = Field(
        title="Chains",
        description="Chains in the complex. Strings, dicts, Chain, Fragment, or Ligands.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_entity_types_parameter(cls, data: Any) -> Any:
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
    def normalize_chains(cls, chains: Any) -> list[Chain | Fragment]:
        """Normalize chains to a flat list of ``Chain`` (biopolymer) or ``Fragment`` (ligand) objects."""
        if not isinstance(chains, list):
            raise ValueError(f"chains must be a list, got {type(chains)}")

        normalized: list[Chain | Fragment] = []
        for chain_idx, chain in enumerate(chains):
            try:
                normalized.extend(_normalize_chain_entry(chain))
            except Exception as e:  # noqa: PERF203 -- preserve per-entry index in error
                raise ValueError(f"Chain {chain_idx} is invalid: {e}") from e

        if len(normalized) > len(CHAIN_IDS):
            raise ValueError(
                f"Cannot provide more than {len(CHAIN_IDS)} chains (got {len(normalized)} after expansion)"
            )
        return normalized

    @property
    def chain_sequences(self) -> list[str]:
        """Sequences for all chains. SMILES for Fragments, sequence for Chains."""
        return [chain.smiles if isinstance(chain, Fragment) else chain.sequence for chain in self.chains]  # type: ignore[misc]

    @property
    def entity_types(self) -> list[str]:
        """Entity types for all chains in the complex.

        Returns:
            list[str]: List of entity types, one for each chain.
        """
        return [chain.entity_type for chain in self.chains]  # type: ignore[misc]

    def extract_protein_chains(self) -> tuple[list[str], list[str]]:
        """Extract protein sequences and their chain IDs from this complex.

        Iterates over all chains and returns only those with entity_type
        "protein", along with their corresponding uppercase-letter chain IDs.

        Returns:
            tuple[list[str], list[str]]: Tuple of (protein_sequences, protein_chain_ids).
        """
        protein_seqs: list[str] = []
        protein_chain_ids: list[str] = []
        for i, chain in enumerate(self.chains):
            if isinstance(chain, Chain) and chain.entity_type == "protein":
                protein_seqs.append(chain.sequence)
                protein_chain_ids.append(CHAIN_IDS[i])
        return protein_seqs, protein_chain_ids

    def sum_of_chain_lengths(self) -> int:
        """Total residue/atom count across chains. Ligand Fragments contribute 1 each."""
        return sum(1 if isinstance(chain, Fragment) else len(chain.sequence) for chain in self.chains)

    def num_chains(self) -> int:
        """Get the number of chains in the complex."""
        return len(self.chains)

    def get_entity_type_set(self) -> set[str]:
        """Get the set of unique entity types in the complex."""
        return set(self.entity_types)

    def add_modification_to_chain(
        self, chain_index: int, position: int, modification_code: str
    ) -> "StructurePredictionComplex":
        """Add a modification to a specific (biopolymer) chain in the complex.

        Args:
            chain_index (int): 0-based index of the chain to modify.
            position (int): 1-based position in the chain sequence.
            modification_code (str): CCD code for the modification.

        Returns:
            StructurePredictionComplex: Self for method chaining.

        Raises:
            IndexError: If chain_index is out of bounds.
            ValueError: If position exceeds chain sequence length, or if the
                target chain is a ligand Fragment (which cannot have modifications).

        Examples:
            >>> complex = StructurePredictionComplex(chains=["MVLSPADKTN", "ATCGATCG"])
            >>> complex.add_modification_to_chain(0, 5, "SEP")
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

    def clear_all_modifications(self) -> "StructurePredictionComplex":
        """Remove all modifications from all biopolymer chains in the complex.

        Returns:
            StructurePredictionComplex: Self for method chaining.
        """
        for chain in self.chains:
            if isinstance(chain, Chain):
                chain.clear_modifications()
        return self

    def has_modifications(self) -> bool:
        """Check if any (biopolymer) chain in this complex has any modifications."""
        return any(isinstance(chain, Chain) and chain.has_modifications() for chain in self.chains)

    def __repr__(self) -> str:
        """Get a string representation of the complex."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the complex."""
        return f"StructurePredictionComplex(with {self.num_chains()} chains)"

    def __iter__(self) -> Iterator[Chain | Fragment]:  # type: ignore[override]
        """Iterate over the chains in the complex."""
        return iter(self.chains)

    def __getitem__(self, index: int) -> Chain | Fragment:
        """Get a chain by index."""
        return self.chains[index]


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
        # Build the Chain first so infer_entity_type runs; if it auto-classifies as
        # ligand from a SMILES string, convert to Fragment (mirrors the Chain-object branch).
        chain_obj = Chain(**chain)
        if chain_obj.entity_type == "ligand":
            return [Fragment(smiles=chain_obj.sequence)]
        return [chain_obj]
    if isinstance(chain, str):
        if detect_sequence_type(chain) == "ligand":
            return list(parse_smiles_string(chain))
        return [Chain(sequence=chain)]
    raise ValueError(f"Chain entry must be a string, dict, Chain, Fragment, or Ligands. Got {type(chain).__name__}")


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
        >>> inputs = StructurePredictionInput(complexes=["MVLSPADKTN", "ACDEFGHIKL"])
        >>>
        >>> # Multi-chain complex (entity types auto-inferred)
        >>> inputs = StructurePredictionInput(complexes=[["MVLSPADKTN", "ACDEFGHIKL"]])
        >>>
        >>> # Explicit format with entity types using Chain objects
        >>> complex1 = StructurePredictionComplex(
        ...     chains=[
        ...         Chain(sequence="MVLSPADKTN", entity_type="protein"),
        ...         Chain(sequence="ATCGATCG", entity_type="dna"),
        ...     ]
        ... )
        >>> inputs = StructurePredictionInput(complexes=[complex1])
        >>>
        >>> # Explicit format with entity types using dictionaries
        >>> complex2 = StructurePredictionComplex(
        ...     chains=[
        ...         {"sequence": "MVLSPADKTN", "entity_type": "protein"},
        ...         {"sequence": "ATCGATCG", "entity_type": "dna"},
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
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]]
    ALLOWS_CHAIN_MODIFICATIONS: ClassVar[bool]

    def __init_subclass__(cls, **kwargs: Any) -> None:
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

    complexes: list[StructurePredictionComplex] = InputField(
        title="Complexes",
        description="List of complexes to predict structure for containing chains and entity types.",
    )
    msas: dict[str, MSA] | None = InputField(
        default=None,
        title="MSAs",
        description="Pre-computed MSAs keyed by sequence. Populated by preprocess() or supplied directly.",
    )

    @classmethod
    def item_cost(cls, item: StructurePredictionComplex) -> float:
        """Cost ~ total residues across all chains in the complex."""
        return float(item.sum_of_chain_lengths())

    @field_validator("complexes", mode="before", check_fields=False)
    @classmethod
    def normalize_complexes(cls, value: Any) -> Any:
        """Validate and normalize complex specifications from raw input."""
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
            # If item is a dict (e.g. from JSON deserialization), construct the model
            elif isinstance(sp_complex, dict):
                try:
                    final_complexes.append(StructurePredictionComplex(**sp_complex))
                except Exception as e:
                    raise ValueError(f"Invalid complex dictionary: {e}") from e
            # If item is a string, treat it as a single-chain complex
            elif isinstance(sp_complex, str):
                final_complexes.append(StructurePredictionComplex(chains=[sp_complex]))  # type: ignore[list-item]
            # If item is a list that contains strings, treat it as a multi-chain complex
            elif isinstance(sp_complex, list) and all(isinstance(x, str) for x in sp_complex):
                final_complexes.append(StructurePredictionComplex(chains=sp_complex))
            else:
                raise ValueError(
                    f"Unsupported input format for auto-normalizing complexes: {sp_complex}. "
                    "Expected type: StructurePredictionComplex, dict, str, or list of strings"
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

            if not cls.ALLOWS_CHAIN_MODIFICATIONS and comp.has_modifications():
                raise ValueError(
                    f"Complex {comp_idx} contains modifications. {cls.__name__} does not allow chain modifications."
                )

        return final_complexes

    def __len__(self) -> int:
        """Get the number of complexes."""
        return len(self.complexes)

    def __getitem__(self, index: int) -> StructurePredictionComplex:
        """Get a complex by index."""
        return self.complexes[index]

    def __iter__(self) -> Iterator[StructurePredictionComplex]:  # type: ignore[override]
        """Iterate over the complexes."""
        return iter(self.complexes)

    def __repr__(self) -> str:
        """Get a string representation of the complexes."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the complexes."""
        return f"StructurePredictionInput(complexes={self.complexes})"

    def get_entity_type_set(self) -> set[str]:
        """Get the set of unique entity types in the input across all complexes."""
        entity_type_set = set()
        for cmplx in self.complexes:
            entity_type_set.update(cmplx.get_entity_type_set())
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
        include_pae_matrix (bool): If True, attach the full per-residue PAE
            matrix. Off by default — the matrix is ``O(n_residues^2)`` floats
            and adds noticeable serialization cost. Default: ``False``.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        include_in_key=False,
    )
    include_pae_matrix: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Attach the full per-residue PAE matrix.",
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
    colabfold_search_config: ColabfoldSearchConfig | None = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Nested configuration for ColabFold MSA search. If None, uses default settings.",
    )

    @classmethod
    def minimal(cls, **kwargs: Any) -> "MSAStructurePredictionConfig":
        """Create a minimal-cost config with MSA generation disabled.

        Args:
            **kwargs (Any): Field values passed to the config constructor.

        Returns:
            MSAStructurePredictionConfig: Config with ``use_msa=False``.
        """
        kwargs.setdefault("use_msa", False)
        return super().minimal(**kwargs)  # type: ignore[return-value]

    def preprocess(self, inputs: StructurePredictionInput) -> StructurePredictionInput:  # type: ignore[override]
        """Preprocess structure prediction inputs before execution."""
        if not self.use_msa:
            return inputs
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return _preprocess_structure_prediction_msas(inputs, self.colabfold_search_config, self.verbose)


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

    structures: list[Structure] = Field(
        title="Structures",
        description="List of predicted structures, one per input complex.",
    )

    def __len__(self) -> int:
        """Get the number of predicted structures."""
        return len(self.structures)

    def __getitem__(self, index: int) -> Structure:
        """Get a predicted structure by index."""
        return self.structures[index]

    def __iter__(self) -> Iterator[Structure]:  # type: ignore[override]
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
    def output_format_options(self) -> list[str]:
        """List of valid file formats for exporting the tool output."""
        return ["cif", "pdb"]

    @property
    def output_format_default(self) -> str:
        """Default file format for exporting the tool output."""
        return "pdb"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
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
    verbose: int = 0,
) -> StructurePredictionInput:
    """Generate MSAs for all unique protein sequences and attach to inputs.

    Collects unique protein sequences across all complexes, runs ColabFold
    search once, and stores the raw MSA objects on ``inputs.msas`` keyed by
    sequence string. Skips ColabFold if MSAs are already supplied.

    Args:
        inputs (StructurePredictionInput): Structure prediction input containing complexes.
        colabfold_search_config (Any): ColabFold search configuration.
        verbose (int): Verbosity level (truthy = log progress); see :class:`BaseConfig`.

    Returns:
        StructurePredictionInput: Updated inputs with ``msas`` field populated.
    """
    if inputs.msas is not None:
        if verbose:
            logger.info(f"Using {len(inputs.msas)} pre-supplied MSA(s), skipping ColabFold search")
        return inputs

    from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
        ColabfoldSearchInput,
        run_colabfold_search,
    )

    # Collect unique protein sequences across all complexes
    unique_seqs: dict[str, str] = {}  # sequence -> query name
    for comp in inputs.complexes:
        for chain in comp.chains:
            if isinstance(chain, Chain) and chain.entity_type == "protein" and chain.sequence not in unique_seqs:
                unique_seqs[chain.sequence] = f"seq_{len(unique_seqs)}"

    if not unique_seqs:
        return inputs

    if verbose:
        logger.info(f"Generating MSAs for {len(unique_seqs)} unique protein sequence(s) using ColabFold search...")

    queries = list(unique_seqs.items())
    colabfold_input = ColabfoldSearchInput(queries=queries)  # type: ignore[arg-type]
    colabfold_output = run_colabfold_search(colabfold_input, colabfold_search_config)

    # Build MSA dict keyed by sequence
    name_to_seq = {v: k for k, v in unique_seqs.items()}
    msas: dict[str, MSA] = {}
    for result in colabfold_output.results:
        if result.msa is not None:
            seq = name_to_seq[result.sequence_id]
            msas[seq] = result.msa
            if verbose:
                logger.info(f"Generated MSA for {result.sequence_id}: {result.num_homologs_found} homologs found")
        else:
            if verbose:
                logger.warning(f"No homologs found for {result.sequence_id}")

    return inputs.model_copy(update={"msas": msas}) if msas else inputs
