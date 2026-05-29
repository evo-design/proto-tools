"""proto_tools/tools/structure_prediction/shared_data_models.py.

Shared data models (configs, outputs) and helpers for structure prediction tools.

Chain-ID convention (see ``resolve_chain_ids`` / ``normalize_output_chain_ids``):
a predicted structure's chain IDs match the input ``Complex`` — explicit ``id``
when set, else positional ``chain_label(i)``; PDB tools are single-character only.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.complex import CHAIN_IDS as CHAIN_IDS
from proto_tools.entities.complex import Chain as Chain
from proto_tools.entities.complex import ChainModification as ChainModification
from proto_tools.entities.complex import Complex as Complex
from proto_tools.entities.complex import chain_label as chain_label
from proto_tools.entities.ligands import Fragment, count_heavy_atoms_for_ccd
from proto_tools.entities.msa import MSA, PairedMSA
from proto_tools.entities.structures import Structure
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
)
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

logger = logging.getLogger(__name__)


class ComplexMSAs(BaseModel):
    """MSAs for one complex's protein chains, keyed by chain index.

    ``paired`` marks the per-chain MSAs as row-aligned across chains by taxonomy
    (row N of every chain is the same organism), which lets downstream predictors
    engage cross-chain pairing. Unpaired MSAs are independent per chain. A bare
    ``dict[int, MSA]`` supplied to ``StructurePredictionInput.msas`` is coerced to
    an unpaired ``ComplexMSAs``.

    Attributes:
        per_chain (dict[int, MSA]): Chain index (position in ``Complex.chains``)
            → its MSA.
        paired (bool): Whether the per-chain MSAs are row-aligned across chains.
            When ``True``, all chains must share the same row count.
    """

    model_config = ConfigDict(extra="forbid")

    per_chain: dict[int, MSA] = Field(description="Chain index → MSA for each protein chain.")
    paired: bool = Field(default=False, description="Whether rows are taxonomy-aligned across chains.")

    @model_validator(mode="after")
    def _validate_paired_row_counts(self) -> "ComplexMSAs":
        """A paired complex's chain MSAs must form a valid (row-aligned) PairedMSA."""
        if self.paired:
            PairedMSA(msas=list(self.per_chain.values()))
        return self

    def as_paired_msa(self) -> PairedMSA:
        """Return the chain MSAs in chain-index order as a general :class:`PairedMSA`."""
        return PairedMSA(msas=[self.per_chain[idx] for idx in sorted(self.per_chain)])


class StructurePredictionInput(BaseToolInput):
    """Input object for structure prediction models.

    This class provides a flexible interface for specifying one or more biological
    complexes for structure prediction. After validation, always contains a list of
    ``Complex`` instances.

    Supports multiple input formats:

    - List of ``Complex`` instances (explicit format)
    - List of sequence strings (each treated as a single-chain complex)
    - List of lists of sequences (each sublist treated as a multi-chain complex)

    Attributes:
        complexes (list[Complex]): List of complexes to predict
            structures for. Each complex contains one or more chains with their
            corresponding entity types. After validation, always a list of
            ``Complex`` instances regardless of input format.

        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one
            :class:`ComplexMSAs` per complex (parallel to ``complexes``). A bare
            ``dict[int, MSA]`` per entry is accepted and coerced to an unpaired
            ``ComplexMSAs``. Populated by ``Config.preprocess()`` via ColabFold
            search, or supplied directly to skip MSA generation. Default: ``None``.

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
        >>> complex1 = Complex(
        ...     chains=[
        ...         Chain(sequence="MVLSPADKTN", entity_type="protein"),
        ...         Chain(sequence="ATCGATCG", entity_type="dna"),
        ...     ]
        ... )
        >>> inputs = StructurePredictionInput(complexes=[complex1])
        >>>
        >>> # Explicit format with entity types using dictionaries
        >>> complex2 = Complex(
        ...     chains=[
        ...         {"sequence": "MVLSPADKTN", "entity_type": "protein"},
        ...         {"sequence": "ATCGATCG", "entity_type": "dna"},
        ...     ]
        ... )
        >>> inputs = StructurePredictionInput(complexes=[complex2])

    Note:
        All input formats are automatically normalized to ``List[Complex]``
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

    complexes: list[Complex] = InputField(
        title="Complexes",
        description="List of complexes to predict structure for containing chains and entity types.",
    )
    msas: list[ComplexMSAs] | None = InputField(
        default=None,
        title="MSAs",
        description="Per-complex MSAs; a bare dict[int, MSA] is coerced to an unpaired ComplexMSAs.",
    )

    @field_validator("msas", mode="before")
    @classmethod
    def _coerce_msas(cls, value: Any) -> Any:
        """Accept a bare ``dict[int, MSA]`` per complex; wrap it as an unpaired ``ComplexMSAs``.

        A serialized ``ComplexMSAs`` (``{"per_chain": ..., "paired": ...}``, e.g. from a
        ``model_dump`` round-trip) is left for Pydantic to validate; only a bare chain dict
        (no ``per_chain`` key) is wrapped.
        """
        if not isinstance(value, list):
            return value  # let Pydantic raise the canonical "should be a list" error
        return [
            ComplexMSAs(per_chain=entry) if isinstance(entry, dict) and "per_chain" not in entry else entry
            for entry in value
        ]

    @classmethod
    def item_cost(cls, item: Complex) -> float:
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
            # If item is a Complex, add it to the final list
            if isinstance(sp_complex, Complex):
                final_complexes.append(sp_complex)
            # If item is a dict (e.g. from JSON deserialization), construct the model
            elif isinstance(sp_complex, dict):
                try:
                    final_complexes.append(Complex(**sp_complex))
                except Exception as e:
                    raise ValueError(f"Invalid complex dictionary: {e}") from e
            # If item is a string, treat it as a single-chain complex
            elif isinstance(sp_complex, str):
                final_complexes.append(Complex(chains=[sp_complex]))  # type: ignore[list-item]
            # If item is a list that contains strings, treat it as a multi-chain complex
            elif isinstance(sp_complex, list) and all(isinstance(x, str) for x in sp_complex):
                final_complexes.append(Complex(chains=sp_complex))
            else:
                raise ValueError(
                    f"Unsupported input format for auto-normalizing complexes: {sp_complex}. "
                    "Expected type: Complex, dict, str, or list of strings"
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

    def __getitem__(self, index: int) -> Complex:
        """Get a complex by index."""
        return self.complexes[index]

    def __iter__(self) -> Iterator[Complex]:  # type: ignore[override]
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
    """Generate per-complex MSAs and attach them as ``list[dict[int, MSA]]``.

    Heterocomplex protein chains within the same complex are submitted as one
    taxonomy-paired query so the model can exploit cross-chain co-evolutionary
    signal. Single-chain and homo-multimer complexes use the unpaired path with
    cross-complex sequence deduplication.

    Args:
        inputs (StructurePredictionInput): Structure prediction input containing complexes.
        colabfold_search_config (Any): ColabFold search configuration.
        verbose (int): Verbosity level (truthy = log progress); see :class:`BaseConfig`.

    Returns:
        StructurePredictionInput: Updated inputs with ``msas`` set to a list of
            ``ComplexMSAs`` parallel to ``complexes``; ``paired=True`` for
            heterocomplexes searched as one taxonomy-paired query.
    """
    if inputs.msas is not None:
        if len(inputs.msas) != len(inputs.complexes):
            raise ValueError(
                f"Pre-supplied msas length ({len(inputs.msas)}) does not match "
                f"complexes length ({len(inputs.complexes)})."
            )
        if verbose:
            total = sum(len(cm.per_chain) for cm in inputs.msas)
            logger.info(f"Using {total} pre-supplied chain MSA(s), skipping ColabFold search")
        return inputs

    from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
        ColabfoldSearchInput,
        run_colabfold_search,
    )

    # Build the query list and a per-complex plan for reassembling results by position.
    # Heterocomplex protein chains → one paired query (a list); single-chain / homo-multimer
    # → one unpaired query (a str), deduplicated across complexes by sequence.
    queries_input: list[str | list[str]] = []
    unpaired_seq_to_query_idx: dict[str, int] = {}
    # Per complex: (protein chains as (chain_idx, sequence), paired query index or None).
    complex_plans: list[tuple[list[tuple[int, str]], int | None]] = []

    for comp in inputs.complexes:
        protein_chains: list[tuple[int, str]] = [
            (ci, ch.sequence)
            for ci, ch in enumerate(comp.chains)
            if isinstance(ch, Chain) and ch.entity_type == "protein"
        ]
        if not protein_chains:
            complex_plans.append(([], None))
            continue

        unique_seqs_in_complex = {seq for _, seq in protein_chains}
        if len(unique_seqs_in_complex) == 1:
            seq = protein_chains[0][1]
            if seq not in unpaired_seq_to_query_idx:
                unpaired_seq_to_query_idx[seq] = len(queries_input)
                queries_input.append(seq)
            complex_plans.append((protein_chains, None))
        else:
            queries_input.append([seq for _, seq in protein_chains])
            complex_plans.append((protein_chains, len(queries_input) - 1))

    if not queries_input:
        return inputs

    if verbose:
        n_paired = sum(1 for _, qi in complex_plans if qi is not None)
        logger.info(
            f"Generating MSAs: {len(unpaired_seq_to_query_idx)} unpaired sequence(s), "
            f"{n_paired} paired group(s) for heterocomplexes..."
        )

    # queries accepts raw str / list[str] entries; the field validator coerces to ColabfoldSearchQuery.
    colabfold_output = run_colabfold_search(
        ColabfoldSearchInput(queries=queries_input),  # type: ignore[arg-type]
        colabfold_search_config,
    )

    if colabfold_output.success is False:
        errors = colabfold_output.errors or ["unknown error"]
        raise RuntimeError(f"colabfold-search MSA generation failed: {' | '.join(errors)}")

    # Results are parallel to queries_input; each result's msas list is parallel to its chains.
    results = colabfold_output.results
    seq_to_msa: dict[str, MSA] = {}
    for seq, query_idx in unpaired_seq_to_query_idx.items():
        msa = results[query_idx].msas[0]  # unpaired query → single-chain result
        if msa is not None:
            seq_to_msa[seq] = msa

    # Reassemble per-complex MSAs parallel to inputs.complexes, one ComplexMSAs each.
    msas_list: list[ComplexMSAs] = []
    for protein_chains, paired_query_idx in complex_plans:
        per_chain: dict[int, MSA] = {}
        if paired_query_idx is not None:
            for (ch_idx, _seq), msa in zip(protein_chains, results[paired_query_idx].msas, strict=True):
                if msa is not None:
                    per_chain[ch_idx] = msa
            # A heterocomplex query is paired regardless of how many chains found homologs.
            msas_list.append(ComplexMSAs(per_chain=per_chain, paired=True))
        else:
            for ch_idx, seq in protein_chains:
                msa = seq_to_msa.get(seq)
                if msa is not None:
                    per_chain[ch_idx] = msa
            msas_list.append(ComplexMSAs(per_chain=per_chain, paired=False))

    has_any = any(m.per_chain for m in msas_list)
    return inputs.model_copy(update={"msas": msas_list}) if has_any else inputs


def unwrap_complex_msas(complex_msas: "ComplexMSAs | None") -> tuple[dict[int, MSA], bool]:
    """Return ``(per_chain_dict, is_paired)`` from a per-complex MSA bundle.

    Helper for downstream structure-prediction tools that need uniform access to a
    complex's MSAs and a flag indicating whether the MSAs are taxonomy-paired (so
    they should populate pairing slots on the per-tool input format).
    """
    if complex_msas is None:
        return {}, False
    return complex_msas.per_chain, complex_msas.paired


_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _row_to_base36_5(n: int) -> str:
    """5-char uppercase base36 of n, zero-padded. Covers 36**5 ≈ 60M rows."""
    out = []
    while n > 0:
        n, r = divmod(n, 36)
        out.append(_BASE36[r])
    return "".join(reversed(out)).rjust(5, "0")


def write_paired_a3m_with_uniprot_headers(msa: MSA, paired_a3m_path: str) -> None:
    r"""Write A3M with UniProt-format headers so species-id-based predictors engage cross-chain pairing.

    AlphaFold3 and Protenix pair MSA rows by extracting a species token from each
    sequence header via regex ``(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|[A-Z0-9]{1,10}_[A-Z0-9]{1,5}``;
    rows whose species appears in ≥2 chains become paired rows. Raw ColabFold
    ``pair.a3m`` headers (``UniRef100_*``) do not match the regex, so we
    synthesize ``>tr|PAIR{B36}|P_{B36}`` for each non-query row where
    ``{B36}`` is the 5-char base36 encoding of the row index. Identical B36
    across chains for matching rows engages pairing by row position (which is
    how ColabFold's paired output is already aligned by taxonomy upstream).
    The query row keeps an inert ``>query`` header; both predictors special-case
    row 0 as the query regardless.
    """
    sequences = msa.aligned_sequences
    lines = [">query", sequences[0]]
    for row_idx, seq in enumerate(sequences[1:], start=1):
        token = _row_to_base36_5(row_idx)
        lines.append(f">tr|PAIR{token}|P_{token}")
        lines.append(seq)
    with open(paired_a3m_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def resolve_chain_ids(chains: Sequence[Chain | Fragment]) -> list[str]:
    """Per-chain IDs in order: explicit ``chain.id`` when set, else ``chain_label(i)``.

    The single source of truth for the chain IDs predictors send to their worker
    and select on, guaranteeing each is unique. A ligand whose id collides with
    another chain — e.g. ``Complex.from_structure`` keys a bound ligand to its
    polymer's chain id — is reassigned the next free positional label that no
    other chain claims. Duplicate *polymer* ids are a caller error and raise.
    """
    preferred = [chain.id or chain_label(idx) for idx, chain in enumerate(chains)]
    reserved = set(preferred)  # every id some chain wants; never hand these out as a fallback
    used: set[str] = set()
    ids: list[str] = []
    for chain, preferred_id in zip(chains, preferred, strict=True):
        cid = preferred_id
        if cid in used:
            if chain.entity_type != "ligand":
                raise ValueError(
                    f"complex has duplicate chain ID {cid!r} (an explicit 'id' collides with another "
                    "chain's id or positional A/B/... label); give each chain a distinct 'id'."
                )
            free = 0
            while chain_label(free) in reserved or chain_label(free) in used:
                free += 1
            cid = chain_label(free)
        used.add(cid)
        ids.append(cid)
    return ids


def normalize_output_chain_ids(structure: Structure, chains: Sequence[Chain | Fragment]) -> Structure:
    """Remap a predicted structure's polymer chain IDs to ``resolve_chain_ids(chains)``.

    Predictors may emit positional or entity-derived names that differ from the
    IDs we sent. Remap output polymer chains by order when the counts match;
    ligand-only chains are excluded on both sides (mirrors
    :meth:`Structure.get_chain_ids`). No-op when already aligned or on count mismatch.
    """
    expected_ids = [
        cid for chain, cid in zip(chains, resolve_chain_ids(chains), strict=True) if chain.entity_type != "ligand"
    ]
    observed_ids = structure.get_chain_ids()
    if observed_ids == expected_ids:
        return structure
    if len(observed_ids) != len(expected_ids):
        logger.warning(
            "structure prediction returned %d polymer chain(s) but input had %d; leaving chain IDs unchanged.",
            len(observed_ids),
            len(expected_ids),
        )
        return structure
    return structure.with_renamed_chains(dict(zip(observed_ids, expected_ids, strict=True)))


# ============================================================================
# AF3-style token counting
# ============================================================================
def count_structure_tokens(chains: list[Chain | Fragment]) -> int:
    """Count tokens for a complex under AlphaFold3-style tokenization.

    1 token per amino acid / nucleotide; heavy-atom count per ligand Fragment and
    per modified residue (from the free CCD component — exact for Protenix, a safe
    +1 over-estimate for Chai-1). Shared by the AF3-family predictors that enforce a
    token budget; callers tokenize any tool-specific entity types (e.g. Chai-1
    glycans) themselves.

    Args:
        chains (list[Chain | Fragment]): Chains in the complex.

    Returns:
        int: Total token count.

    Raises:
        ValueError: If a modification CCD code cannot be resolved.
    """
    total = 0
    for chain in chains:
        if isinstance(chain, Fragment):
            total += chain.heavy_atom_count
            continue
        # Dedupe by position so the residue-token removal and heavy-atom addition stay consistent.
        mods_by_position = {mod.position: mod.modification_code for mod in chain.modifications}
        standard_count = len(chain.sequence) - len(mods_by_position)
        modified_token_count = sum(count_heavy_atoms_for_ccd(code) for code in mods_by_position.values())
        total += standard_count + modified_token_count
    return total
