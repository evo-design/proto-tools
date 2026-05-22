"""proto_tools/tools/inverse_folding/esm_if1/esm_if1_score.py.

ESM-IF1/ProteinDPO scoring tool.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringMetrics,
    InverseFoldingScoringOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)

# Three-letter residue codes biotite's ProteinSequence.convert_letter_3to1 can
# convert. Mirrors the standalone's filter so wrapper-side length validation
# matches the count the model will actually consume.
_KNOWN_AA_3LETTER: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ASX",
        "GLX",
        "UNK",
        "SEC",
        "MSE",
    }
)


# ============================================================================
# Data Models
# ============================================================================
class ESMIF1ScoringPair(BaseModel):
    """Sequence-structure pair for ESM-IF1 scoring.

    Unlike ProteinMPNN/LigandMPNN scoring (which scores the full multi-chain
    sequence against the full structure), ESM-IF1 scores a single chain at a
    time within its multi-chain structural context. ``sequence`` is therefore
    just the target chain's sequence, never the concatenation of every chain.

    Attributes:
        sequence (str): Target chain sequence to score. Length must equal the
            number of residues in the chain identified by ``target_chain``.
        structure (Structure): Protein structure providing the (optionally
            multi-chain) coordinate context.
        target_chain (str | None): Chain ID within ``structure`` whose sequence
            is being scored. ``None`` is permitted only for single-chain
            structures, in which case the sole chain is used. For multi-chain
            structures this field is required.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(title="Sequence", description="Target chain sequence to score")
    structure: Structure = Field(
        title="Input Structure", description="Structure providing the (multi-chain) coordinate context"
    )
    target_chain: str | None = Field(
        default=None,
        title="Target Chain",
        description="Chain ID whose sequence is scored. Required for multi-chain structures.",
    )

    @model_validator(mode="after")
    def validate_sequence_length_matches_target_chain(self) -> "ESMIF1ScoringPair":
        """Reject if ``sequence`` length differs from the model-visible length of ``target_chain``."""
        chain_ids = self.structure.get_chain_ids()
        target = _resolve_target_chain(self.target_chain, chain_ids)
        expected = _model_visible_chain_length(self.structure, target)
        if len(self.sequence) != expected:
            raise ValueError(
                f"esm-if1-score: sequence length {len(self.sequence)} does not match target chain "
                f"{target!r} length {expected}. ESM-IF1 scores one chain at a time; `sequence` must "
                f"cover only the residues the model sees (standard amino acids plus MSE/SEC/UNK)."
            )
        return self


class ESMIF1ScoringInput(BaseToolInput):
    """Input for ESM-IF1/ProteinDPO scoring.

    Attributes:
        sequence_structure_pairs (list[ESMIF1ScoringPair]): List of pairs to score.
            Each pair contains a target chain sequence, a structure, and the chain ID
            within that structure whose sequence is being scored.
    """

    sequence_structure_pairs: list[ESMIF1ScoringPair] = InputField(
        title="Sequence-Structure Pairs",
        description="List of sequence-structure pairs to score",
    )


ESMIF1ScoringOutput = InverseFoldingScoringOutput


class ESMIF1ScoringConfig(BaseConfig):
    """Configuration for ESM-IF1/ProteinDPO scoring.

    Attributes:
        weights_variant (Literal['esmif', 'protein_dpo']): Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
        device (str): Device to run the model on.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description="'esmif' for vanilla ESM-IF1, 'protein_dpo' for DPO-aligned weights",
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def _model_visible_chain_length(structure: Structure, target_chain: str) -> int:
    """Count residues in ``target_chain`` that the standalone's filter will keep."""
    structure.gemmi_struct.setup_entities()
    chain = structure.gemmi_struct[0][target_chain]
    return sum(1 for res in chain.get_polymer() if res.name in _KNOWN_AA_3LETTER)


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    structure = Structure.from_file(str(Path(__file__).parents[1] / "example_input_fixture.pdb"))
    target_chain = structure.get_chain_ids()[0]
    sequence = structure.get_chain_sequence(target_chain)
    return ESMIF1ScoringInput(
        sequence_structure_pairs=[
            ESMIF1ScoringPair(
                sequence=sequence,
                structure=structure,
                target_chain=target_chain,
            )
        ]
    )


@tool(
    key="esm-if1-score",
    label="ESM-IF1 Scoring",
    category="inverse_folding",
    input_class=ESMIF1ScoringInput,
    config_class=ESMIF1ScoringConfig,
    output_class=ESMIF1ScoringOutput,
    metrics_class=InverseFoldingScoringMetrics,
    description=(
        "Score protein sequences against backbone structures using "
        "ESM-IF1 or ProteinDPO. Computes average log-likelihood and "
        "perplexity with multi-chain complex support."
    ),
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequence_structure_pairs",
    iterable_output_field="scores",
    cacheable=True,
)
def run_esm_if1_score(
    inputs: ESMIF1ScoringInput,
    config: ESMIF1ScoringConfig,
    instance: Any = None,
) -> ESMIF1ScoringOutput:
    """Score protein sequences using ESM-IF1/ProteinDPO.

    Scores each sequence against its paired structure using the full complex
    structural context (score_sequence_in_complex). Returns average
    log-likelihood and perplexity.

    Args:
        inputs (ESMIF1ScoringInput): Sequence-structure pairs to score.
        config (ESMIF1ScoringConfig): Configuration including weights variant.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMIF1ScoringOutput: ESMIF1ScoringOutput with scores for each input pair.
    """
    scores = []

    for pair in progress_bar(
        inputs.sequence_structure_pairs,
        desc="ESM-IF1 scoring",
        unit="pair",
        total=len(inputs.sequence_structure_pairs),
    ):
        chain_ids = pair.structure.get_chain_ids()
        target_chain = _resolve_target_chain(pair.target_chain, chain_ids)

        with pair.structure.temp_file() as pdb_path:
            input_dict = {
                "operation": "score",
                "pdb_path": str(pdb_path),
                "chain_ids": chain_ids,
                "target_chain": target_chain,
                "sequence": pair.sequence,
                "seed": config.seed,
                "device": config.device,
                "weights_variant": config.weights_variant,
                "verbose": config.verbose,
            }
            result = ToolInstance.dispatch(
                "esm_if1",
                input_dict,
                instance=instance,
                config=config,
            )
        scores.append(InverseFoldingScoringMetrics(**result["metrics"]))

    return ESMIF1ScoringOutput(scores=scores)


def _resolve_target_chain(target_chain: str | None, chain_ids: list[str]) -> str:
    """Pick the target chain for scoring, validating against the structure's chains.

    Args:
        target_chain (str | None): User-supplied chain ID, or ``None``.
        chain_ids (list[str]): Chain IDs present in the structure.

    Returns:
        str: The resolved chain ID to score.

    Raises:
        ValueError: If ``target_chain`` is ``None`` for a multi-chain structure,
            or if it names a chain not present in the structure.
    """
    if target_chain is None:
        if len(chain_ids) != 1:
            raise ValueError(
                f"esm-if1-score: `target_chain` is required for multi-chain structures "
                f"(structure has chains {chain_ids}). Set `target_chain` on ESMIF1ScoringPair "
                f"to the chain whose sequence you want to score."
            )
        return chain_ids[0]
    if target_chain not in chain_ids:
        raise ValueError(
            f"esm-if1-score: `target_chain={target_chain!r}` is not in the structure's chains {chain_ids}."
        )
    return target_chain
