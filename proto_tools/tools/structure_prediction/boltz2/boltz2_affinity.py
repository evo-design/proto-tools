"""Boltz-2 binding-affinity prediction tool."""

import os
import tempfile
from logging import getLogger
from typing import Any, ClassVar

from pydantic import model_validator

from proto_tools.entities.complex import chain_label
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import BFactorType, SingleChainSelection, Structure
from proto_tools.tools.structure_prediction.boltz2.boltz2 import Boltz2Config
from proto_tools.tools.structure_prediction.boltz2.helpers import build_chain_msa_paths, complex_to_yaml
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, InputField, ToolInstance
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = getLogger(__name__)

_AFFINITY_MAX_HEAVY_ATOMS = 128


def _resolve_binder_chain_id(comp: Any, binder_chain: SingleChainSelection | None, index: int) -> str:
    """Resolve and validate one complex's binder ligand chain ID."""
    chain_ids = [chain_label(j) for j in range(len(comp.chains))]
    ligand_ids = [chain_label(j) for j, chain in enumerate(comp.chains) if isinstance(chain, Fragment)]
    if not any(not isinstance(chain, Fragment) and chain.entity_type == "protein" for chain in comp.chains):
        raise ValueError(
            f"Complex {index} must contain at least one protein chain (the affinity target); "
            f"got entity types {comp.entity_types}."
        )

    if binder_chain is None:
        if len(ligand_ids) != 1:
            raise ValueError(
                f"Complex {index} has {len(ligand_ids)} ligand chain(s); affinity needs exactly one. "
                f"Set binder_chain explicitly when a complex has 0 or >1 ligands."
            )
        binder_id = ligand_ids[0]
    else:
        binder_id = binder_chain.chain
        if binder_id not in chain_ids:
            raise ValueError(f"binder_chain {binder_id!r} is not a chain in complex {index} (valid IDs: {chain_ids}).")
        if binder_id not in ligand_ids:
            raise ValueError(
                f"binder_chain {binder_id!r} in complex {index} is not a ligand chain; "
                f"affinity is supported only for small-molecule ligand binders."
            )

    binder = comp.chains[chain_ids.index(binder_id)]
    assert isinstance(binder, Fragment)  # noqa: S101 -- narrowed by the ligand-id check above
    if binder.heavy_atom_count > _AFFINITY_MAX_HEAVY_ATOMS:
        raise ValueError(
            f"Complex {index} binder ligand {binder_id} has {binder.heavy_atom_count} heavy atoms; "
            f"Boltz-2 affinity supports up to {_AFFINITY_MAX_HEAVY_ATOMS}."
        )
    return binder_id


# ============================================================================
# Data Models
# ============================================================================
class Boltz2AffinityInput(StructurePredictionInput):
    """Input for Boltz-2 affinity prediction.

    Attributes:
        complexes (list[Complex]): Each needs >=1 protein target and >=1 ligand chain.
        msas (dict[str, MSA] | None): Inherited pre-computed MSAs keyed by sequence.
        binder_chain (SingleChainSelection | None): Ligand to score; None auto-detects the sole ligand.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    binder_chain: SingleChainSelection | None = InputField(
        default=None,
        title="Binder Chain",
        description="Ligand chain to score for affinity; None auto-detects the single ligand in each complex.",
    )

    @model_validator(mode="after")
    def validate_affinity_binders(self) -> "Boltz2AffinityInput":
        """Validate that each complex has a resolvable ligand binder (fails fast at construction)."""
        for i, comp in enumerate(self.complexes):
            _resolve_binder_chain_id(comp, self.binder_chain, i)
        return self

    @property
    def resolved_binder_chain_ids(self) -> list[str]:
        """Binder chain ID per complex, re-resolved on access so it tracks ToolPool partitioning."""
        return [_resolve_binder_chain_id(comp, self.binder_chain, i) for i, comp in enumerate(self.complexes)]


class Boltz2AffinityMetrics(Metrics):
    """Per-complex affinity metrics emitted by Boltz-2 (see ``metric_spec``)."""

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "affinity_pred_value": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "log10(IC50 μM)",
        },
        "affinity_probability_binary": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "affinity_pred_value1": {
            "availability": "when ensemble emits per-model values",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "log10(IC50 μM)",
        },
        "affinity_probability_binary1": {
            "availability": "when ensemble emits per-model values",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "affinity_pred_value2": {
            "availability": "when ensemble emits per-model values",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "log10(IC50 μM)",
        },
        "affinity_probability_binary2": {
            "availability": "when ensemble emits per-model values",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
    }
    primary_metric: str | None = "affinity_pred_value"


class Boltz2AffinityOutput(StructurePredictionOutput):
    """Boltz-2 affinity output; ``structures`` holds one predicted pose per complex with affinity metrics."""


class Boltz2AffinityConfig(Boltz2Config):
    """Boltz-2 affinity config: all inherited Boltz2Config structure-pass knobs plus the affinity-pass controls.

    Attributes:
        recycling_steps (int): Inherited. Refinement passes for the structure pass. Default: ``3``.
        sampling_steps (int): Inherited. Denoising steps for the structure pass. Default: ``200``.
        diffusion_samples (int): Inherited. Structure samples per complex. Default: ``1``.
        step_scale (float): Inherited. Diffusion step size for the structure pass. Default: ``1.5``.
        max_msa_seqs (int): Inherited. Cap on MSA depth fed into the model. Default: ``8192``.
        subsample_msa (bool): Inherited. Randomly subsample the MSA each run. Default: ``False``.
        num_workers (int): Inherited. Dataloader workers for prediction. Default: ``min(cpu_count, 4)``.
        use_msa (bool): Inherited. Use ColabFold MSAs for protein chains. Default: ``True``.
        colabfold_search_config (ColabfoldSearchConfig | None): Inherited. ColabFold MSA search config. Default: ``None``.
        include_pae_matrix (bool): No-op for affinity; excluded from the cache key. Default: ``False``.
        affinity_mw_correction (bool): Apply molecular-weight correction to the affinity value head. Default: ``False``.
        sampling_steps_affinity (int): Denoising steps for the affinity pass. Default: ``200``.
        diffusion_samples_affinity (int): Diffusion samples per complex for the affinity pass. Default: ``5``.
    """

    include_pae_matrix: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Not applicable to affinity; no PAE matrix is emitted.",
        include_in_key=False,
    )
    affinity_mw_correction: bool = ConfigField(
        title="Affinity MW Correction",
        default=False,
        description="Apply molecular-weight correction to the Boltz-2 affinity value head.",
    )
    sampling_steps_affinity: int = ConfigField(
        title="Affinity Sampling Steps",
        default=200,
        ge=1,
        description="Denoising steps for the affinity prediction pass.",
    )
    diffusion_samples_affinity: int = ConfigField(
        title="Affinity Diffusion Samples",
        default=5,
        ge=1,
        description="Independent diffusion samples per complex for the affinity pass.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Boltz2AffinityInput:
    """Minimal valid input: short protein plus L-tyrosine."""
    return Boltz2AffinityInput(complexes=[["MKTLPGCDA", "c1cc(ccc1C[C@@H](C(=O)O)N)O"]])  # type: ignore[list-item]


@tool(
    key="boltz2-affinity",
    label="Boltz-2 Affinity",
    category="structure_prediction",
    input_class=Boltz2AffinityInput,
    config_class=Boltz2AffinityConfig,
    output_class=Boltz2AffinityOutput,
    metrics_class=Boltz2AffinityMetrics,
    description=(
        "Predicted binding affinity (log10 IC50 μM) and binder probability for a small molecule "
        "against a protein target, via Boltz-2."
    ),
    uses_gpu=True,
    device_count="1-2",
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_boltz2_affinity(
    inputs: Boltz2AffinityInput,
    config: Boltz2AffinityConfig,
    instance: Any = None,
) -> Boltz2AffinityOutput:
    """Predict binding affinity for protein-ligand complexes via Boltz-2 (one ``Structure`` per complex)."""
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advance the seed per complex so duplicate inputs get distinct seeds.
    results = [
        run_boltz2_affinity_on_complex(
            config=config,
            sp_complex=comp,
            binder_chain_id=binder_id,
            msas=inputs.msas,
            instance=instance,
            seed=base_seed + dispatch_idx,
        )
        for dispatch_idx, (comp, binder_id) in enumerate(
            progress_bar(
                list(zip(inputs.complexes, inputs.resolved_binder_chain_ids, strict=True)),
                desc="Scoring affinity (Boltz-2)",
                unit="complex",
                total=len(inputs.complexes),
            )
        )
    ]
    return Boltz2AffinityOutput(structures=results)


def run_boltz2_affinity_on_complex(
    config: Boltz2AffinityConfig,
    sp_complex: Any,
    binder_chain_id: str,
    msas: dict[str, Any] | None = None,
    instance: Any = None,
    seed: int | None = None,
) -> Structure:
    """Run Boltz-2 affinity on a single complex and return the predicted ``Structure``."""
    if seed is None:
        seed = config.seed
    if config.verbose:
        logger.info("Using local GPU for Boltz-2 affinity prediction...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "boltz2_output")
        os.makedirs(output_dir)

        chain_msa_paths: dict[str, str] | None = None
        if config.use_msa:
            chain_msa_paths = build_chain_msa_paths(sp_complex, msas, temp_dir, verbose=config.verbose)

        yaml_content = complex_to_yaml(
            sp_complex.chains,
            chain_msa_paths=chain_msa_paths,
            affinity_binder_chain_id=binder_chain_id,
        )

        input_yaml_path = os.path.join(temp_dir, "boltz2_input.yaml")
        with open(input_yaml_path, "w") as f:
            f.write(yaml_content)

        input_data = {
            "operation": "predict_affinity",
            "input_yaml_path": str(input_yaml_path),
            "output_dir": str(output_dir),
            "recycling_steps": config.recycling_steps,
            "sampling_steps": config.sampling_steps,
            "diffusion_samples": config.diffusion_samples,
            "step_scale": config.step_scale,
            "max_msa_seqs": config.max_msa_seqs,
            "subsample_msa": config.subsample_msa,
            "num_workers": config.num_workers,
            "device": config.device,
            "verbose": config.verbose,
            "seed": seed,
            "sampling_steps_affinity": config.sampling_steps_affinity,
            "diffusion_samples_affinity": config.diffusion_samples_affinity,
            "affinity_mw_correction": config.affinity_mw_correction,
        }

        output_data = ToolInstance.dispatch("boltz2", input_data, instance=instance, config=config)

        cif_output = output_data["structure_cif_output"]
        # Structure-pass confidence metrics are intentionally dropped; affinity output carries Boltz2AffinityMetrics.
        affinity_raw = output_data["affinity_metrics"]

    metrics_dict: dict[str, Any] = {key: float(value) for key, value in affinity_raw.items()}
    return Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=Boltz2AffinityMetrics(**metrics_dict),
        source="boltz2-affinity",
    )
