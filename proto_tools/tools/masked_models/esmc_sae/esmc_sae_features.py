"""proto_tools/tools/masked_models/esmc_sae/esmc_sae_features.py.

ESM C sparse autoencoder (SAE) feature extraction.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from proto_tools.tools.masked_models.shared_data_models import MaskedModelInput
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolOutput, ConfigField, ToolInstance

logger = logging.getLogger(__name__)

ESMC_SAE_CHECKPOINTS = Literal["esmc_300m", "esmc_600m", "esmc_6b"]
ESMC_SAE_TARGETS = Literal["hidden_states", "mlp_outputs"]
ESMC_SAE_K = Literal[16, 32, 64, 128, 256, 512]
ESMC_SAE_CODEBOOK_SIZES = Literal[8192, 16384, 32768, 65536, 131072]
ESMC_SAE_BACKBONES = Literal["transformers", "esm"]

# HuggingFace repo stem per backbone, its transformer depth, and its hidden width.
_BACKBONES: dict[str, tuple[str, int, int]] = {
    "esmc_300m": ("biohub/ESMC-300M", 30, 960),
    "esmc_600m": ("biohub/ESMC-600M", 36, 1152),
    "esmc_6b": ("biohub/ESMC-6B", 80, 2560),
}

# Layer targeted by the single-layer sweep: ~75% depth, where Biohub found
# representations generalize best.
_SWEEP_LAYER: dict[str, int] = {"esmc_300m": 23, "esmc_600m": 27, "esmc_6b": 60}

# Codebook sizes published as all-layer SAEs, keyed by (target, checkpoint).
# These are the only combinations trained against every backbone layer.
_ALL_LAYER_CODEBOOKS: dict[tuple[str, str], frozenset[int]] = {
    ("hidden_states", "esmc_300m"): frozenset({16384}),
    ("hidden_states", "esmc_600m"): frozenset({16384}),
    ("hidden_states", "esmc_6b"): frozenset({16384, 131072}),
    ("mlp_outputs", "esmc_300m"): frozenset({131072}),
    ("mlp_outputs", "esmc_600m"): frozenset({131072}),
    ("mlp_outputs", "esmc_6b"): frozenset({131072}),
}

# All-layer SAEs are only published at k=64.
_ALL_LAYER_K = 64

# Warn past this much estimated SAE download. MLP-output SAEs use a 131072 codebook,
# so one 6B layer is ~2.7 GB and a full sweep runs to hundreds of gigabytes.
_DOWNLOAD_WARN_GB = 10.0


def resolve_sae_repo(model_checkpoint: str, sae_target: str, layers: list[int], k: int, codebook_size: int) -> str:
    """Resolve an SAE configuration to the HuggingFace repo that publishes it.

    Two families are available. Single-layer SAEs sweep ``k`` and ``codebook_size`` but
    exist only at one layer per backbone, so a request for exactly that layer is served
    from there — those are the SAEs Biohub studied and published feature descriptions
    for. Anything else falls to the all-layer SAEs, which cover every backbone layer but
    only at ``k=64`` and a fixed codebook size. The two families hold independently
    trained weights, so their feature indices are not interchangeable.

    Args:
        model_checkpoint (str): Backbone variant key, e.g. ``"esmc_300m"``.
        sae_target (str): Activations the SAE was trained on.
        layers (list[int]): Backbone layer indices to attach SAEs to.
        k (int): Active features per position.
        codebook_size (int): Total features the SAE can represent.

    Returns:
        str: The HuggingFace repo id holding the requested SAE.

    Raises:
        ValueError: If no published SAE matches the combination.

    Examples:
        >>> resolve_sae_repo("esmc_300m", "hidden_states", [11, 23], 64, 16384)
        'biohub/ESMC-300M-sae-k64-codebook16384'
        >>> resolve_sae_repo("esmc_300m", "hidden_states", [23], 64, 16384)
        'biohub/ESMC-300M-sae-layer23-k64-codebook16384'
    """
    stem = _BACKBONES[model_checkpoint][0]
    mlp = "mlp-" if sae_target == "mlp_outputs" else ""
    sweep_layer = _SWEEP_LAYER[model_checkpoint]

    # A request for exactly the sweep layer is served by the SAE trained for that layer.
    # The single-layer and all-layer repos hold independently trained weights, and the
    # single-layer one is what Biohub studied and generated feature descriptions for, so
    # the more specific match wins.
    if sae_target == "hidden_states" and layers == [sweep_layer]:
        return f"{stem}-sae-layer{sweep_layer}-k{k}-codebook{codebook_size}"

    if k == _ALL_LAYER_K and codebook_size in _ALL_LAYER_CODEBOOKS[(sae_target, model_checkpoint)]:
        return f"{stem}-sae-{mlp}k{k}-codebook{codebook_size}"

    all_layer_sizes = sorted(_ALL_LAYER_CODEBOOKS[(sae_target, model_checkpoint)])
    raise ValueError(
        f"No published SAE for {model_checkpoint} / {sae_target} with k={k}, "
        f"codebook_size={codebook_size}, layers={layers}. Use k=64 with "
        f"codebook_size in {all_layer_sizes} for any layers, or (hidden_states only) "
        f"layers=[{sweep_layer}] to sweep k and codebook_size."
    )


# ============================================================================
# Data Models
# ============================================================================
# Input:
ESMCSAEFeaturesInput = MaskedModelInput


# Output:
class SAELayerFeatures(BaseModel):
    """Sparse SAE features for one sequence at one backbone layer.

    The two arrays are index-parallel and cover the input residues only; the
    tokenizer's start and end tokens are stripped. Each inner list holds the
    ``k`` active features at that residue, ordered by descending magnitude.

    Both arrays are in sequence order, so list index ``i`` describes residue
    ``i + 1`` of the input under proto-tools' 1-indexed coordinate convention.

    Attributes:
        layer (int): Backbone transformer layer the SAE reads from.
        feature_indices (list[list[int]]): Active codebook indices per residue, in
            sequence order; entry ``i`` is residue ``i + 1`` (1-indexed).
        feature_magnitudes (list[list[float]]): Activation values per residue.
    """

    layer: int = Field(title="Layer", description="Backbone transformer layer the SAE reads from")
    feature_indices: list[list[int]] = Field(
        title="Feature Indices",
        description="Active codebook indices per residue, descending by magnitude",
    )
    feature_magnitudes: list[list[float]] = Field(
        title="Feature Magnitudes",
        description="Activation value of each active feature, descending",
    )


class SequenceSAEFeatures(BaseModel):
    """SAE features for one input sequence across every requested layer.

    Bundling per-layer results in one object per sequence keeps the tool's
    iterable output 1:1 with ``inputs.sequences`` so per-item caching and dedup
    expand correctly.

    Attributes:
        sequence (str): The input sequence these features were computed from, echoed so
            an exported row can name the residue a feature fired on.
        layers (list[SAELayerFeatures]): One entry per requested layer, ascending.
    """

    sequence: str = Field(
        title="Sequence",
        description="Input sequence these features were computed from",
    )
    layers: list[SAELayerFeatures] = Field(
        title="Layers",
        description="Per-layer sparse features for this sequence, ascending by layer",
    )


class ESMCSAEFeaturesOutput(BaseToolOutput):
    """Output from ESM C sparse autoencoder feature extraction.

    Attributes:
        results (list[SequenceSAEFeatures]): Per-sequence SAE features, index-parallel
            with the input sequences.
    """

    results: list[SequenceSAEFeatures] = Field(
        title="Results",
        description="Per-sequence sparse SAE features",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            payload = [r.model_dump() for r in self.results]
            with open(path, "w") as f:
                json.dump(payload, f)
        elif file_format == "csv":
            import csv

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_index", "layer", "position", "residue", "feature_index", "magnitude"])
                for seq_idx, result in enumerate(self.results):
                    for layer in result.layers:
                        for pos, (indices, magnitudes) in enumerate(
                            zip(layer.feature_indices, layer.feature_magnitudes, strict=True)
                        ):
                            residue = result.sequence[pos]
                            for feature_index, magnitude in zip(indices, magnitudes, strict=True):
                                # position is 1-indexed, per the repo's coordinate convention.
                                writer.writerow([seq_idx, layer.layer, pos + 1, residue, feature_index, magnitude])
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class ESMCSAEFeaturesConfig(BaseConfig):
    """Configuration for ESM C sparse autoencoder feature extraction.

    Sparse autoencoders decompose ESM C's internal activations into a large,
    sparsely-active feature space that is more interpretable than raw embeddings.
    An SAE is trained against one backbone layer, so ``layers`` selects which
    activations to decompose, and ``sae_target``, ``k``, and ``codebook_size``
    select which published SAE to load.

    Attributes:
        model_checkpoint (ESMC_SAE_CHECKPOINTS): ESM C backbone whose activations are
            decomposed. The SAE must match the backbone it was trained on.
        layers (list[int] | None): Backbone layers to attach SAEs to. ``None`` uses the
            ~75%-depth layer Biohub sweeps (300M: 23, 600M: 27, 6B: 60). Each layer
            adds a download and GPU memory.
        sae_target (ESMC_SAE_TARGETS): Which activations the SAE was trained on. Hidden
            states give a global view; MLP outputs isolate one layer's computation.
        k (ESMC_SAE_K): Active features per residue. Fixed in the SAE's weights, so this
            selects a model rather than a threshold; only ``64`` was trained against
            every layer, and other values exist solely at the sweep layer.
        codebook_size (ESMC_SAE_CODEBOOK_SIZES): Total features the SAE can represent,
            also fixed in its weights. Larger codebooks split concepts more finely; which
            sizes exist depends on ``model_checkpoint`` and ``sae_target``.
        backbone (ESMC_SAE_BACKBONES): Which ESM C implementation supplies the activations
            the SAE reads. ``"transformers"`` matches the published SAE documentation.
            ``"esm"`` reads the ``esmc`` toolkit's weights instead, avoiding a second
            backbone download at the cost of ~1% disagreement in active features.
        batch_size (int): Sequences per forward pass.
        device (str): Device to run the model on.

    Note:
        Only published combinations are accepted; the validator names the valid
        alternatives when a combination does not exist.
    """

    model_checkpoint: ESMC_SAE_CHECKPOINTS = ConfigField(
        title="ESM C Backbone",
        default="esmc_300m",
        description="ESM C backbone whose activations the SAE decomposes",
        reload_on_change=True,
    )
    layers: list[int] | None = ConfigField(
        title="Layers",
        default=None,
        description="Backbone layers to attach SAEs to; None uses the ~75%-depth sweep layer",
        reload_on_change=True,
    )
    sae_target: ESMC_SAE_TARGETS = ConfigField(
        title="SAE Target",
        default="hidden_states",
        description="Selects the SAE trained on this activation source: residual stream or per-layer MLP",
        reload_on_change=True,
    )
    k: ESMC_SAE_K = ConfigField(
        title="Active Features",
        default=64,
        description="Selects the SAE trained with this many active features per residue; 64 serves any layer",
        reload_on_change=True,
    )
    codebook_size: ESMC_SAE_CODEBOOK_SIZES = ConfigField(
        title="Codebook Size",
        default=16384,
        description="Selects the SAE trained with this many features in total; larger splits concepts finer",
        reload_on_change=True,
    )
    backbone: ESMC_SAE_BACKBONES = ConfigField(
        title="Backbone Source",
        default="transformers",
        description="Which ESM C implementation supplies activations; 'esm' reuses the esmc toolkit weights",
        reload_on_change=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Sequences per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_sae_available(self) -> "ESMCSAEFeaturesConfig":
        """Reject SAE combinations that Biohub has not published."""
        depth = _BACKBONES[self.model_checkpoint][1]
        for layer in self.resolved_layers:
            if not 0 <= layer <= depth:
                raise ValueError(
                    f"layer {layer} is out of range for {self.model_checkpoint}: valid layers are 0-{depth}"
                )
        if len(set(self.resolved_layers)) != len(self.resolved_layers):
            raise ValueError(f"layers must be unique, got {self.resolved_layers}")

        resolve_sae_repo(self.model_checkpoint, self.sae_target, self.resolved_layers, self.k, self.codebook_size)
        return self

    @property
    def resolved_layers(self) -> list[int]:
        """Requested layers, defaulting to the backbone's sweep layer."""
        if self.layers is None:
            return [_SWEEP_LAYER[self.model_checkpoint]]
        return sorted(self.layers)

    @property
    def estimated_download_gb(self) -> float:
        """Approximate SAE download for the requested layers, in gigabytes.

        Counts the two dominant tensors in a layer file, ``W_enc`` and ``W_dec``, each
        ``d_model x codebook_size`` fp32. Excludes the decoder bias and the ``idf`` /
        ``max`` normalization buffers, which together add well under a percent.
        Computed rather than queried so config validation stays offline; agrees with
        the published file sizes to two decimal places.
        """
        d_model = _BACKBONES[self.model_checkpoint][2]
        per_layer = d_model * self.codebook_size * 2 * 4
        return len(self.resolved_layers) * per_layer / 1e9

    @property
    def sae_repo(self) -> str:
        """HuggingFace repo id for the configured SAE."""
        return resolve_sae_repo(
            self.model_checkpoint, self.sae_target, self.resolved_layers, self.k, self.codebook_size
        )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESMCSAEFeaturesInput(sequences=["MKTAYIAKQR"])


@tool(
    key="esmc-sae-features",
    label="ESM C SAE Features",
    category="masked_models",
    input_class=ESMCSAEFeaturesInput,
    config_class=ESMCSAEFeaturesConfig,
    output_class=ESMCSAEFeaturesOutput,
    description="Decompose ESM C activations into interpretable sparse autoencoder features",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    cacheable=True,
)
def run_esmc_sae_features(
    inputs: ESMCSAEFeaturesInput, config: ESMCSAEFeaturesConfig, instance: Any = None
) -> ESMCSAEFeaturesOutput:
    """Extract sparse autoencoder features from ESM C activations.

    Loads an ESM C backbone, attaches the sparse autoencoders trained against the
    requested layers, and returns the active features at every residue. Only the
    requested layers are downloaded from the SAE repo, so selecting one layer of
    the 6B model fetches a fraction of the full collection.

    Args:
        inputs (ESMCSAEFeaturesInput): Validated input containing one or more protein
            sequences.
        config (ESMCSAEFeaturesConfig): Validated configuration selecting the backbone,
            layers, and SAE variant.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMCSAEFeaturesOutput: One ``SequenceSAEFeatures`` per input sequence, each
            holding per-layer active feature indices and magnitudes for every residue,
            in sequence order (list index ``i`` is residue ``i + 1``).

    See Also:
        - ESM C SAE collection: https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers
        - ESM GitHub: https://github.com/Biohub/esm

    Examples:
        >>> from proto_tools.tools.masked_models.esmc_sae import (
        ...     ESMCSAEFeaturesConfig,
        ...     ESMCSAEFeaturesInput,
        ...     run_esmc_sae_features,
        ... )
        >>> inputs = ESMCSAEFeaturesInput(sequences=["MVLSPADKTNVKAAW"])
        >>> result = run_esmc_sae_features(inputs, ESMCSAEFeaturesConfig(layers=[11, 23]))
        >>> print([layer.layer for layer in result.results[0].layers])
        [11, 23]
    """
    estimated_gb = config.estimated_download_gb
    if estimated_gb > _DOWNLOAD_WARN_GB:
        logger.warning(
            f"{len(config.resolved_layers)} layers of {config.sae_repo} is about "
            f"{estimated_gb:.0f} GB of SAE weights to download on first use. Request fewer "
            f"layers, or a smaller codebook_size, to reduce it."
        )
    logger.debug(
        f"Using local for ESM C SAE features: {config.model_checkpoint} @ {config.sae_repo} "
        f"(~{estimated_gb:.2f} GB for layers {config.resolved_layers})"
    )
    outputs = ToolInstance.dispatch(
        "esmc_sae",
        {
            "operation": "sae_features",
            "sequences": inputs.sequences,
            "model_checkpoint": config.model_checkpoint,
            "sae_repo": config.sae_repo,
            "layers": config.resolved_layers,
            "backbone": config.backbone,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    results = [
        SequenceSAEFeatures(
            sequence=sequence,
            layers=[
                SAELayerFeatures(
                    layer=layer,
                    feature_indices=per_sequence[str(layer)]["indices"],
                    feature_magnitudes=per_sequence[str(layer)]["magnitudes"],
                )
                for layer in config.resolved_layers
            ],
        )
        for sequence, per_sequence in zip(inputs.sequences, outputs["features"], strict=True)
    ]

    return ESMCSAEFeaturesOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "sae_repo": config.sae_repo,
            "layers": config.resolved_layers,
            "k": config.k,
            "codebook_size": config.codebook_size,
            "num_sequences": len(inputs.sequences),
        },
        results=results,
    )
