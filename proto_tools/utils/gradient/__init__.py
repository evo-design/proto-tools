"""proto_tools/utils/gradient/__init__.py.

Shared contracts for relaxed-sequence gradient tools: a tool declares the quantities it can
differentiate, and returns each one's raw value and raw derivative.

``terms`` holds what can be asked for and what comes back for one request — the term classes, the
catalogs naming them, and :class:`TermGradient`. ``metadata`` holds what a gradient was taken with
respect to, and whether two of them can be combined. ``models`` holds the input, config and output
base classes every gradient tool builds on.
"""

from proto_tools.utils.gradient.metadata import GradientMetadata, gradient_l2_norm, input_fingerprint
from proto_tools.utils.gradient.models import (
    BatchedNucleotideGradientInput,
    GradientConfig,
    GradientInput,
    GradientOutput,
    GradientResult,
    ProteinGradientInput,
    RelaxationConfig,
    coerce_to_item_list,
)
from proto_tools.utils.gradient.terms import (
    DifferentiableTerm,
    GroupedTermCatalog,
    TermCatalog,
    TermGradient,
    resolve_terms,
)

__all__ = [
    "BatchedNucleotideGradientInput",
    "DifferentiableTerm",
    "GradientConfig",
    "GradientInput",
    "GradientMetadata",
    "GradientOutput",
    "GradientResult",
    "GroupedTermCatalog",
    "ProteinGradientInput",
    "RelaxationConfig",
    "TermCatalog",
    "TermGradient",
    "coerce_to_item_list",
    "gradient_l2_norm",
    "input_fingerprint",
    "resolve_terms",
]
