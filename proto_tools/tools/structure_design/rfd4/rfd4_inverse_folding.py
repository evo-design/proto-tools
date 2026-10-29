"""proto_tools/tools/structure_design/rfd4/rfd4_inverse_folding.py.

Example:
    >>> from proto_tools.entities.structures import Structure
    >>> from proto_tools.tools.structure_design.rfd4 import (
    ...     run_rfd4_inverse_folding,
    ...     RFD4Input,
    ...     RFD4DesignSpec,
    ...     RFD4InverseFoldingConfig,
    ... )
    >>> inputs = RFD4Input(design_specs=[RFD4DesignSpec(input_structure=Structure.from_file("backbone.pdb"))])
    >>> result = run_rfd4_inverse_folding(inputs, RFD4InverseFoldingConfig(num_replicates=8))
    >>> sequences, chain_ids = result[0][0].complex.extract_protein_chains()
    >>> print(sequences[0])
"""

import logging
from typing import Any

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_design.rfd4.rfd4_design import (
    EXAMPLE_INPUT_FIXTURE,
    RFD4CommonConfig,
    RFD4DesignSpec,
    RFD4Input,
    RFD4Output,
    _dispatch_rfd4,
)
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)


class RFD4InverseFoldingConfig(RFD4CommonConfig):
    """Configuration for RFD4-Proteina inverse folding.

    Inverse folding holds the input backbone on independently resampled forward
    marginals while generating the sequence and sidechain latents, so the backbone is
    preserved and only the sequence is designed. Sequence diversity is controlled
    through ``sequence_sampling``: lowering ``tsr_k`` (upstream default ``4.0``) raises
    the sampling temperature.

    Note:
        Classifier-free guidance and partial diffusion are design-time controls and are
        not exposed here; see ``rfd4-design`` for those.

    Attributes:
        num_replicates (int): Designs generated per input spec.
        batch_size (int): Designs denoised in parallel on the GPU. Confirm the longest
            input fits at ``1`` before raising it.
        nsteps (int): Integration steps per design (upstream default 60). A
            ``recycle_strategy`` preset expands to exactly 60 steps and upstream requires
            the two to agree, so changing this requires ``recycle_strategy=None``.
        recycle_strategy (RecycleStrategy | None): Trunk recycling schedule as
            ``<low|medium|high>/<flat|midpeak|front_loaded>``, trading compute for
            quality. ``midpeak`` totals 120/150/180 trunk passes; ``flat`` and
            ``front_loaded`` total 120/180/300. The ``thinking_*`` aliases are the
            front-loaded schedules. Setting any preset pins ``nsteps`` to 60. ``None``
            disables the schedule and runs a fixed recycle depth of 0, which is the only
            way to use a custom ``nsteps``.
        backbone_sampling (RFD4SamplerTuning): Temperature for the backbone modality.
        sequence_sampling (RFD4SamplerTuning): Temperature for the sidechain/sequence
            latent modality. Lowering ``tsr_k`` raises sequence diversity.
        deterministic_sampling (bool): Switch both modalities to ODE integration, with
            no mid-trajectory noise injection.
        model_checkpoint (str): HuggingFace repo id or a local model-bundle directory.
            The denoiser and its matching autoencoder always load together.
        model_revision (str | None): HuggingFace commit to pin, for reproducible loads.
            Ignored when ``model_checkpoint`` is a local directory.
        adapter_path (str | None): Local PEFT/LoRA adapter directory, loaded onto the
            base and autoencoder pair it was fine-tuned against.
        use_torch_compile (bool): Compile the trunk across the upstream length buckets.
            Off by default: the first call otherwise pays a long compile across 15
            buckets, which only pays off for large runs.
        save_trajectory_every (int): Snapshot the trajectory every N steps. ``0``
            disables. Rejected unless ``output_dir`` is set.
        output_dir (str | None): Directory for run outputs; ``None`` uses a tempdir. Each
            call writes into a fresh ``run-<id>`` subdirectory of it, so a reused
            ``output_dir`` cannot return a previous run's designs, and the trajectory
            frames for a run land in that run's own subdirectory.
        device (str): Execution device. ``rfd4-generate`` builds its Lightning trainer
            with ``accelerator="gpu"`` unconditionally, so a GPU is required.
    """

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Select the upstream ``inverse_folding`` generation preset."""
        cli_kwargs = super().get_cli_kwargs()
        # The preset sets generation.args.inverse_folding=true and the only supported
        # inverse_folding_method, so neither is forwarded separately.
        cli_kwargs["generation"] = "inverse_folding"
        return cli_kwargs


def example_input() -> RFD4Input:
    """Minimal valid input for testing and examples."""
    return RFD4Input(design_specs=[RFD4DesignSpec(input_structure=Structure.from_file(EXAMPLE_INPUT_FIXTURE))])


@tool(
    key="rfd4-inverse-folding",
    label="RFD4-Proteina Inverse Folding",
    category="structure_design",
    input_class=RFD4Input,
    config_class=RFD4InverseFoldingConfig,
    output_class=RFD4Output,
    description="All-atom sequence and sidechain design on a fixed backbone with RFD4-Proteina",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["design_specs"],
    iterable_output_field="designed_structures",
    max_chunk_size=1,
    cacheable=True,
    stochastic=True,
)
def run_rfd4_inverse_folding(inputs: RFD4Input, config: RFD4InverseFoldingConfig, instance: Any = None) -> RFD4Output:
    """Design a sequence and sidechains for a fixed backbone with RFD4-Proteina.

    The input backbone is held fixed while the sequence and sidechain latents are
    resampled, so each replicate is an independent sequence for the same fold, returned
    with its packed all-atom sidechains.

    Conditions on the input spec still apply, so a run can fix part of the sequence
    (``known_sequence``), steer burial, or exclude an identity outright (for example a
    ``sequence_probabilities`` entry setting ``CYS`` to ``0.0``).

    Args:
        inputs (RFD4Input): One or more backbones to redesign, with optional conditions.
        config (RFD4InverseFoldingConfig): Sampling, recycling, and checkpoint settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RFD4Output: ``designed_structures`` holds one ``RFD4Designs`` bundle per input
            spec, in input order. Each ``RFD4Design`` carries the repacked structure and
            a ``Complex`` whose per-chain sequences are the designed sequences.

    See Also:
        - RFD4-Proteina code: https://github.com/RosettaCommons/RFD4-Proteina
        - Inverse folding guide: https://rfd4-proteina.org/latest/
        - ``proteinmpnn-sample``, ``ligandmpnn-sample``, ``fampnn-sample`` for
          message-passing alternatives in the ``inverse_folding`` category.

    Example:
        >>> # Eight diverse sequences for one backbone, at a raised temperature
        >>> inputs = RFD4Input(design_specs=[RFD4DesignSpec(input_structure=Structure.from_file("backbone.pdb"))])
        >>> config = RFD4InverseFoldingConfig(num_replicates=8, sequence_sampling={"tsr_k": 2.0})
        >>> result = run_rfd4_inverse_folding(inputs, config)
        >>> for design in result[0]:
        ...     print(design.complex.extract_protein_chains()[0][0])

    Note:
        Unlike the message-passing inverse-folding tools, RFD4 returns repacked all-atom
        coordinates alongside the sequence, so no separate sidechain-packing step is needed.
    """
    logger.debug("Using local GPU for RFD4-Proteina inverse folding")
    return _dispatch_rfd4(inputs, config, instance=instance, operation="inverse_folding", source="rfd4-inverse-folding")
