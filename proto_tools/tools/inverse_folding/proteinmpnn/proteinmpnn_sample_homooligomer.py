"""proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample_homooligomer.py.

ProteinMPNN symmetric homooligomer sampling tool.
"""

import contextlib
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from proto_tools.entities.structures.selection import ChainSelection, ResidueSelection, StructureInputBase
from proto_tools.tools.inverse_folding.proteinmpnn._sampling import collect_samples, split_chain_segments
from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
    ProteinMPNNDesign,
    ProteinMPNNDesignMetrics,
    ProteinMPNNDesignSet,
    ProteinMPNNSampleConfig,
    ProteinMPNNSampleOutput,
)
from proto_tools.tools.inverse_folding.structure_staging import StagedStructure, staged_structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, InputField
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ProteinMPNNHomooligomerStructureInput(StructureInputBase):
    """One symmetric homooligomer backbone and its per-protomer design constraints.

    Selections are expressed **per protomer**, not per chain. One protomer sequence
    is sampled for the whole assembly, so a constraint that applied to only some
    protomers would have no coherent meaning; this schema makes that unrepresentable
    rather than rejecting it after the fact.

    Attributes:
        structure (Structure): Backbone of the oligomer.
        protomer_chains (ChainSelection): The protomer chains to design, at least two
            and all of equal length. Accepts shorthand ``"A,B"`` or ``["A", "B"]``.
        protomer_fixed_positions (list[int] | None): Positions held at their input
            identity, 1-indexed within a single protomer and applied to every
            protomer chain.

    Examples:
        >>> inp = ProteinMPNNHomooligomerStructureInput(
        ...     structure="/path/to/homodimer.pdb",
        ...     protomer_chains=["A", "B"],
        ...     protomer_fixed_positions=[10, 11],
        ... )
    """

    protomer_chains: ChainSelection = Field(
        title="Protomer Chains",
        description="Protomer chains to design; at least two, all of equal length.",
        examples=[["A", "B"]],
    )
    protomer_fixed_positions: list[int] | None = Field(
        default=None,
        title="Protomer Fixed Positions",
        description="Positions held fixed, 1-indexed within one protomer, applied to every protomer.",
        examples=[[10, 11]],
    )

    @model_validator(mode="after")
    def _validate_protomers(self) -> Self:
        """Reject selections that cannot be tied, and warn on non-identical protomers."""
        chain_ids = self.protomer_chain_ids
        sequences = {cid: self.structure.get_chain_sequence(cid) for cid in chain_ids}
        lengths = {cid: len(seq) for cid, seq in sequences.items()}

        if len(chain_ids) < 2:
            raise ValueError(f"protomer_chains needs at least two chains to tie; got {chain_ids}")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"protomer_chains must all be the same length; got {lengths}")

        protomer_length = next(iter(lengths.values()))
        if self.protomer_fixed_positions is not None:
            out_of_range = sorted(p for p in self.protomer_fixed_positions if not 1 <= p <= protomer_length)
            if out_of_range:
                raise ValueError(
                    f"protomer_fixed_positions {out_of_range} fall outside the "
                    f"1..{protomer_length} protomer (positions are 1-indexed within one protomer)"
                )

        # Tying is driven by the backbone, so designing a pseudo-symmetric assembly whose
        # protomers differ in sequence is legitimate; it is just rarely what was intended.
        if len(set(sequences.values())) != 1:
            logger.warning(
                "Protomer chains %s are equal length but not sequence-identical, so this is a "
                "pseudo-symmetric assembly. One sequence will be sampled and applied to all of "
                "them. Use proteinmpnn-sample to design these chains independently.",
                chain_ids,
            )
        return self

    @property
    def protomer_chain_ids(self) -> list[str]:
        """Protomer chain IDs, in the order the designed segments are returned.

        Returns:
            list[str]: Chain IDs selected by ``protomer_chains``.
        """
        return list(self.protomer_chains.chains)

    @property
    def protomer_length(self) -> int:
        """Residue count of a single protomer.

        Returns:
            int: Length of the first protomer chain; all protomers are equal length.
        """
        return len(self.structure.get_chain_sequence(self.protomer_chain_ids[0]))

    @property
    def fixed_positions(self) -> ResidueSelection | None:
        """Per-protomer fixed positions broadcast to every protomer chain.

        Returns:
            ResidueSelection | None: The same positions on each protomer chain, or
                ``None`` when no positions are fixed.
        """
        if not self.protomer_fixed_positions:
            return None
        positions = sorted(set(self.protomer_fixed_positions))
        return ResidueSelection(chains={cid: list(positions) for cid in self.protomer_chain_ids})

    @contextlib.contextmanager
    def staged(self) -> Iterator[StagedStructure]:
        """Stage this input's structure for the model, numbered 1..N per chain.

        Yields:
            StagedStructure: The staged file and the maps needed to read model output back.
        """
        with staged_structure(self.structure) as staged:
            yield staged


class ProteinMPNNHomooligomerSampleInput(BaseToolInput):
    """Tool input for ProteinMPNN symmetric homooligomer sampling.

    Attributes:
        inputs (list[ProteinMPNNHomooligomerStructureInput]): One entry per oligomer
            backbone to design.
    """

    inputs: list[ProteinMPNNHomooligomerStructureInput] = InputField(
        title="Inputs",
        description="One entry per homooligomer backbone to design.",
    )


class ProteinMPNNHomooligomerSampleConfig(ProteinMPNNSampleConfig):
    """Configuration for ProteinMPNN symmetric homooligomer sampling.

    Shares every field with :class:`ProteinMPNNSampleConfig`. Note that ``temperature``
    is not directly comparable to the general sampling tool: tied decoding averages the
    per-protomer logits, including their Gumbel noise, so the effective sampling noise
    falls by roughly ``1/sqrt(n)`` for an ``n``-mer at the same nominal temperature.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits. See the note
            above: an ``n``-mer is less diverse than a monomer at the same value.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes of amino acids to exclude.
        seed (int | None): Random seed to use for sampling; None draws a fresh seed.
        model_choice (Literal['proteinmpnn', 'v_48_002', 'v_48_010', 'v_48_030', 'abmpnn', 'soluble']): Model
            weights. ``"proteinmpnn"`` is ColabDesign's default ``v_48_020`` (0.20 Å backbone noise). The
            ``v_48_002`` / ``v_48_010`` / ``v_48_030`` variants are the same architecture trained at
            0.02 / 0.10 / 0.30 Å backbone noise (the suffix is the noise level in Å x 100); more noise
            yields more diverse, robust designs at some cost in recovery. ``"abmpnn"``
            is antibody-optimized; ``"soluble"`` is soluble-protein-trained.
        backbone_noise (float): Gaussian noise (Å) added to backbone coordinates before each forward pass.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNHomooligomerSampleInput(
        inputs=[
            ProteinMPNNHomooligomerStructureInput(
                structure=str(Path(__file__).parent / "example_homooligomer_fixture.pdb"),  # type: ignore[arg-type]
                protomer_chains=["A", "B"],  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="proteinmpnn-sample-homooligomer",
    label="ProteinMPNN Homooligomer Sampling",
    category="inverse_folding",
    input_class=ProteinMPNNHomooligomerSampleInput,
    config_class=ProteinMPNNHomooligomerSampleConfig,
    output_class=ProteinMPNNSampleOutput,
    description="Design a symmetric homooligomer with ProteinMPNN by tying protomer positions",
    uses_gpu=True,
    pin_visible_devices=True,
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
    max_chunk_size=32,
)
def run_proteinmpnn_sample_homooligomer(
    inputs: ProteinMPNNHomooligomerSampleInput,
    config: ProteinMPNNHomooligomerSampleConfig,
    instance: Any = None,
) -> ProteinMPNNSampleOutput:
    """Design a symmetric homooligomer by tying corresponding protomer positions.

    Ties corresponding positions across the protomer chains during decoding, so a single
    protomer sequence is sampled while the whole oligomer is in context and then returned
    on every protomer. The alternative — designing one protomer against placeholder
    partner sequences and copying the result — chooses a sequence the model never saw in
    its own interface.

    Args:
        inputs (ProteinMPNNHomooligomerSampleInput): Oligomer backbones, each with its
            ``protomer_chains`` and optional ``protomer_fixed_positions``.
        config (ProteinMPNNHomooligomerSampleConfig): Configuration for sampling.
        instance (Any): A ToolInstance, or a string referencing one pre-registered via
            ToolInstance.get/persist_tool (unknown names raise); None runs one-shot.

    Returns:
        ProteinMPNNSampleOutput: One design set per input structure, each design a
            complete complex carrying the same sampled sequence on every protomer.

    Note:
        Chains outside ``protomer_chains`` are carried through unchanged with
        ``redesigned=False``, exactly as in ``proteinmpnn-sample``.
    """
    design_sets = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Draw a fresh seed per chunk so identical structures across inputs do not collide.
    seed_rng = random.Random(base_seed)  # noqa: S311 -- non-cryptographic

    for inp in progress_bar(
        inputs.inputs,
        desc="ProteinMPNN homooligomer sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        protomer_ids = inp.protomer_chain_ids

        all_seqs, all_perp, all_seq_recovery = collect_samples(
            inp,
            config,
            chain_ids=protomer_ids,
            fixed_positions=inp.fixed_positions,
            homooligomer=True,
            seed_rng=seed_rng,
            instance=instance,
        )

        complexes = []
        for raw_seq, perp, recovery in zip(all_seqs, all_perp, all_seq_recovery, strict=True):
            segments = split_chain_segments(raw_seq, protomer_ids, tied=True)
            complexes.append(
                ProteinMPNNDesign.from_design(
                    inp.structure,
                    dict(zip(protomer_ids, segments, strict=True)),
                    set(protomer_ids),
                    tool="proteinmpnn-sample-homooligomer",
                    include_ligands=False,
                    metrics=ProteinMPNNDesignMetrics(perplexity=perp, sequence_recovery=recovery),
                )
            )
        design_sets.append(ProteinMPNNDesignSet(complexes=complexes))
    return ProteinMPNNSampleOutput(design_sets=design_sets)
