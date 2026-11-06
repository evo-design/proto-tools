"""Terms, configuration and result assembly shared by the ESMFold2 inversion gradient tools."""

import logging
from typing import Any, ClassVar, Literal, get_args

from pydantic import Field

from proto_tools.utils import (
    ConfigField,
    DifferentiableTerm,
    GradientConfig,
    GradientMetadata,
)
from proto_tools.utils.tool_io import Directionality

logger = logging.getLogger(__name__)

#: Inversion checkpoints these tools differentiate. The maintained ESMFold2 prediction
#: checkpoints are a separate family and are not interchangeable with these.
ESMFold2InversionCheckpoint = Literal["ESMFold2-Experimental-Fast", "ESMFold2-Experimental-Fast-Cutoff2025"]
INVERSION_CHECKPOINTS: tuple[str, ...] = get_args(ESMFold2InversionCheckpoint)


class ESMFold2InversionTerm(DifferentiableTerm):
    """One differentiable ESMFold2 inversion quantity, read off the predicted distogram.

    Attributes:
        id (str): Fixed identifier, pinned by each concrete quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"


class ESMFold2ChainTerm(ESMFold2InversionTerm):
    """A quantity evaluated over one chain of the complex.

    Attributes:
        chain_id (str | None): Chain to evaluate. ``None`` means the only designable chain,
            which is unambiguous whenever exactly one chain is being designed.
    """

    chain_id: str | None = Field(
        default=None,
        title="Chain ID",
        description="Chain to evaluate; None means the only designable chain.",
    )


class ESMFold2IntraContactTerm(ESMFold2ChainTerm):
    """Self-contact within one chain: its residues' two best contacts inside that same chain.

    Partners are restricted to the chain itself, so a residue cannot satisfy this by touching a
    neighbour across a chain boundary.

    Attributes:
        id (Literal["intra_contact"]): Fixed identifier for this kind of quantity.
        chain_id (str | None): Inherited; chain to evaluate.
    """

    DESCRIPTION: ClassVar[str] = "Self-contact entropy over a chain's two best intra-chain contacts"

    id: Literal["intra_contact"] = Field(
        default="intra_contact",
        title="Term ID",
        description="Identifies this quantity as the intra-chain contact objective.",
    )


class ESMFold2GlobularityTerm(ESMFold2ChainTerm):
    """Excess radius of gyration of one chain over a packed globular protein of its length.

    Attributes:
        id (Literal["globularity"]): Fixed identifier for this kind of quantity.
        chain_id (str | None): Inherited; chain to evaluate.
    """

    DESCRIPTION: ClassVar[str] = "Chain radius of gyration in excess of a packed protein"
    UNITS: ClassVar[str | None] = "angstroms"

    id: Literal["globularity"] = Field(
        default="globularity",
        title="Term ID",
        description="Identifies this quantity as the per-chain globularity objective.",
    )


class ESMFold2PseudoPerplexityTerm(ESMFold2ChainTerm):
    """ESM C masked pseudo-perplexity of one chain, scored as its own sequence.

    Chains are scored independently, so a chain boundary is never read as a peptide bond.
    Selecting this term loads a second model alongside the folding trunk.

    Attributes:
        id (Literal["plm_nll"]): Fixed identifier for this kind of quantity.
        chain_id (str | None): Inherited; chain to evaluate.
    """

    DESCRIPTION: ClassVar[str] = "ESM C masked pseudo-perplexity of a designed chain"
    UNITS: ClassVar[str | None] = "nats per masked position"

    id: Literal["plm_nll"] = Field(
        default="plm_nll",
        title="Term ID",
        description="Identifies this quantity as the ESM C pseudo-perplexity sequence prior.",
    )


class ESMFold2InterContactTerm(ESMFold2InversionTerm):
    """Interface contact between a fixed target and the designed binder.

    Deliberately asymmetric: scored over the target and partnered against the binder, so it
    rewards the binder reaching the target. It is not a generic inter-chain objective, which is
    why it is offered only by the binder tool.

    Attributes:
        id (Literal["inter_contact"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Interface contact entropy between the target and the binder"

    id: Literal["inter_contact"] = Field(
        default="inter_contact",
        title="Term ID",
        description="Identifies this quantity as the target-binder interface contact objective.",
    )


class ESMFold2EpitopeTerm(ESMFold2InversionTerm):
    """Contact surprisal between the binder and the requested target hotspots.

    Attributes:
        id (Literal["epitope"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Contact surprisal between the binder and the target hotspots"

    id: Literal["epitope"] = Field(
        default="epitope",
        title="Term ID",
        description="Identifies this quantity as the target-hotspot contact objective.",
    )


class ESMFold2InversionConfigBase(GradientConfig):
    """Settings shared by both ESMFold2 inversion gradient tools.

    Attributes:
        model_name (ESMFold2InversionCheckpoint): Inversion checkpoint to fold with.
        plm_passes (int): Independent mask draws averaged by the pseudo-perplexity term.
        plm_batch_size (int): Masked sequences per language-model forward pass.
        compute_gradient (bool): Run the backward passes; ``False`` returns values only.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        seed (int): Inherited seed for checkpoint loading, dropout and mask draws.
        device (str): Inherited device to run the model on.
    """

    model_name: ESMFold2InversionCheckpoint = ConfigField(
        title="Model Name",
        default=INVERSION_CHECKPOINTS[0],
        description="Inversion checkpoint to fold with",
        reload_on_change=True,
    )
    plm_passes: int = ConfigField(
        title="Pseudo-perplexity Passes",
        default=4,
        ge=1,
        description="Independent mask draws averaged by the pseudo-perplexity term",
    )
    plm_batch_size: int = ConfigField(
        title="Pseudo-perplexity Batch Size",
        default=128,
        ge=1,
        description="Masked sequences per language-model forward pass",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward passes and return gradients; False scores the terms only",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


def basis_metadata(
    matrices: list[list[list[float]]],
    alphabet: str,
    temperature: float | None,
    soft: float,
    hard: float,
) -> GradientMetadata:
    """Describe the point every term of one candidate was differentiated at.

    All terms come from one fold, so they share one basis. With several designable chains the
    gradient rows are their matrices concatenated in chain order, and the basis describes that
    same concatenation, which is what keeps ``compose`` able to combine them.

    Args:
        matrices (list[list[list[float]]]): Designable matrices, in chain order.
        alphabet (str): Column order shared by those matrices.
        temperature (float | None): The input's temperature.
        soft (float): Relaxation blend in force.
        hard (float): Straight-through coefficient in force.

    Returns:
        GradientMetadata: The shared basis for this candidate's terms.
    """
    concatenated = [row for matrix in matrices for row in matrix]
    return GradientMetadata.from_input(
        concatenated,
        vocab=list(alphabet),
        temperature=temperature,
        soft=soft,
        hard=hard,
    )


def resolve_catalog_names(value: Any, catalog: Any) -> Any:
    """Accept catalog names and term classes alongside instances in a ``gradient_terms`` field."""
    from proto_tools.utils import resolve_terms

    if not isinstance(value, list):
        return value
    return [resolve_terms([entry], catalog)[0] if isinstance(entry, str | type) else entry for entry in value]
