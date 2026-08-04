"""Source of truth for the per-app Modal split.

Declares which services belong to which app, their timeouts, and their modules.
"""

import logging
import os

logger = logging.getLogger(__name__)

APP_BUCKETS: dict[str, list[str]] = {
    # GPU services.
    "proto-tools-ablang": ["AbLangService"],
    "proto-tools-alphafold2": ["AlphaFold2Service"],
    "proto-tools-alphagenome": ["AlphaGenomeService"],
    "proto-tools-bioemu": ["BioEmuService"],
    "proto-tools-boltz2": ["Boltz2Service"],
    "proto-tools-borzoi": ["BorzoiService"],
    "proto-tools-chai1": ["Chai1Service"],
    "proto-tools-enformer": ["EnformerService"],
    "proto-tools-esm2": ["ESM2Service"],
    "proto-tools-esm3": ["ESM3Service"],
    "proto-tools-esmc": ["ESMCService"],
    "proto-tools-esmfold": ["ESMFoldService"],
    "proto-tools-esmfold2": ["ESMFold2Service"],
    "proto-tools-esmif1": ["ESMIF1Service"],
    "proto-tools-evo1": ["Evo1Service"],
    "proto-tools-evo2": ["Evo2Service"],
    "proto-tools-fampnn": ["FAMPNNService"],
    "proto-tools-freebindcraft": ["FreeBindCraftService"],
    "proto-tools-ligandmpnn": ["LigandMPNNService"],
    "proto-tools-malinois": ["MalinoisService"],
    "proto-tools-metal3d": ["Metal3DService"],
    "proto-tools-opendde": ["OpenDDEService"],
    "proto-tools-pangolin": ["PangolinService"],
    "proto-tools-parade": ["ParadeService"],
    "proto-tools-progen2": ["ProGen2Service"],
    "proto-tools-progen3": ["ProGen3Service"],
    "proto-tools-proteinmpnn": ["ProteinMPNNService"],
    "proto-tools-protenix": ["ProtenixService"],
    "proto-tools-puffin": ["PuffinService"],
    "proto-tools-rf3": ["RF3Service"],
    "proto-tools-rfdiffusion3": ["RFdiffusion3Service"],
    "proto-tools-splice-transformer": ["SpliceTransformerService"],
    "proto-tools-spliceai": ["SpliceAIService"],
    # CPU services — one app each, so deploying one does not build the others.
    "proto-tools-ccd-lookup": ["CcdLookupService"],
    "proto-tools-crispr-tracr-rna": ["CrisprTracrRNAService"],
    "proto-tools-dssp": ["DSSPService"],
    "proto-tools-foldmason": ["FoldmasonService"],
    "proto-tools-foldseek": ["FoldseekService"],
    "proto-tools-ipsae": ["IPSAEService"],
    "proto-tools-mafft": ["MafftAlignService"],
    "proto-tools-minced": ["MincedService"],
    "proto-tools-miranda": ["MirandaService"],
    "proto-tools-orfipy": ["OrfipyService"],
    "proto-tools-pdockq2": ["PDockQ2Service"],
    "proto-tools-prodigal": ["ProdigalService"],
    "proto-tools-promoter-calculator": ["PromoterCalculatorService"],
    "proto-tools-pyhmmer": ["PyHmmerService"],
    "proto-tools-pymol": ["PyMOLService"],
    "proto-tools-pyrosetta": ["PyRosettaService"],
    "proto-tools-segmasker": ["SegmaskerService"],
    "proto-tools-structure-metrics": ["StructureMetricsService"],
    "proto-tools-tmalign": ["TMalignService"],
    "proto-tools-usalign": ["USalignService"],
    "proto-tools-viennarna": ["ViennaRNAService"],
}


def _invert_buckets(buckets: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for app_name, services in buckets.items():
        for service in services:
            if service in out:
                raise ValueError(f"service {service!r} is claimed by both {out[service]!r} and {app_name!r}")
            out[service] = app_name
    return out


SERVICE_TO_APP: dict[str, str] = _invert_buckets(APP_BUCKETS)


# Modal's hard ceiling on a container wall. Not a number we chose, and not one a deploy may exceed:
# a larger ``timeout=`` is rejected at deploy time. Every tier is clamped to it.
MODAL_MAX_TIMEOUT_SECONDS = 86400  # 24 hours

# Container wall tiers, in seconds. A service picks a tier rather than its own number, so the fleet
# runs on a handful of understood budgets instead of one constant per service that nobody can compare.
#
# Tiers are deliberately generous. A wall has to cover the slowest input a tool accepts, under the
# slowest config it accepts, on a cold container — and per-item cost varies by orders of magnitude
# with sequence length. A tier that merely covers the typical case is a tier that fails in production.
TIER_SECONDS: dict[str, int] = {
    "fast": 600,  # 10 min — parsers and short CPU scans
    "medium": 1800,  # 30 min — light model work
    "long": 3600,  # 1 hour — most GPU model work
    "extended": 14400,  # 4 hours — sampling and trajectory work
    "batch": MODAL_MAX_TIMEOUT_SECONDS,  # full design pipelines, at the longest wall Modal permits
}

# The tier for work measured in hours rather than minutes. It is the whole cost policy in one name:
# a service on it gets the longest wall Modal allows, no retries (:func:`retries_for_service`), and
# a warning before the caller's first dispatch (``proto_tools.modal.client``). A run this long is
# expensive enough that billing it twice for the same failure is worse than surfacing the failure.
BATCH_TIER = "batch"

# Wall tier per service. Raising a service to a longer tier is always safe; shortening one can fail
# work that used to complete, so treat a reduction as a change that needs measurement behind it.
SERVICE_TIERS: dict[str, str] = {
    "AbLangService": "long",
    "AlphaFold2Service": "long",
    "AlphaGenomeService": "long",
    "BioEmuService": "extended",
    "Boltz2Service": "long",
    "BorzoiService": "long",
    "CcdLookupService": "fast",
    "Chai1Service": "long",
    "CrisprTracrRNAService": "extended",
    "DSSPService": "fast",
    "ESM2Service": "long",
    "ESM3Service": "long",
    "ESMCService": "long",
    "ESMFold2Service": "long",
    "ESMFoldService": "long",
    "ESMIF1Service": "long",
    "EnformerService": "medium",
    "Evo1Service": "long",
    "Evo2Service": "long",
    "FAMPNNService": "long",
    "FoldmasonService": "medium",
    "FoldseekService": "medium",
    "FreeBindCraftService": "batch",
    "IPSAEService": "fast",
    # Raised from medium: ligandmpnn-score chunks 64 items, the same as proteinmpnn-score, which
    # gets an hour for the same work.
    "LigandMPNNService": "long",
    "MafftAlignService": "medium",
    "MalinoisService": "medium",
    "Metal3DService": "long",
    "MincedService": "medium",
    "MirandaService": "medium",
    "OpenDDEService": "long",
    "OrfipyService": "fast",
    "PDockQ2Service": "fast",
    "PangolinService": "long",
    "ParadeService": "long",
    "ProGen2Service": "long",
    "ProGen3Service": "long",
    "ProdigalService": "medium",
    "PromoterCalculatorService": "long",  # scores every position of every sequence in pure Python
    "ProteinMPNNService": "long",
    "ProtenixService": "long",
    # Both puffin tools chunk 64 items; medium's 30 minutes leaves 28s each, under the per-item
    # floor test_gpu_walls_leave_a_plausible_per_item_budget enforces.
    "PuffinService": "long",
    "PyHmmerService": "medium",
    "PyMOLService": "fast",
    "PyRosettaService": "long",
    "RF3Service": "long",
    "RFdiffusion3Service": "long",
    "SegmaskerService": "fast",
    "SpliceAIService": "long",
    "SpliceTransformerService": "medium",
    "StructureMetricsService": "fast",
    "TMalignService": "fast",
    "USalignService": "fast",
    "ViennaRNAService": "medium",
}


def _resolve_timeout_scale() -> float:
    """Read ``PROTO_MODAL_TIMEOUT_SCALE``, refusing any value that would shorten a wall.

    Lengthening a tier is a deployer's call to make; shortening one kills work that used to
    complete, and would do it from an environment variable that no test covers. A value below 1
    is therefore ignored rather than honoured, as is one that does not parse.

    Returns:
        float: Multiplier to apply to every tier, never less than 1.
    """
    raw = os.getenv("PROTO_MODAL_TIMEOUT_SCALE")
    if raw is None:
        return 1.0
    try:
        scale = float(raw)
    except ValueError:
        logger.warning("PROTO_MODAL_TIMEOUT_SCALE=%r is not a number; using 1.", raw)
        return 1.0
    if scale < 1.0:
        logger.warning("PROTO_MODAL_TIMEOUT_SCALE=%r would shorten every container wall; using 1.", raw)
        return 1.0
    return scale


# Multiplies every tier, for a workload whose inputs are larger than the shipped budgets assume.
# Baked in at deploy time, like PROTO_MODAL_SCALEDOWN_WINDOW:
#
#     PROTO_MODAL_TIMEOUT_SCALE=2 proto-tools deploy --apps esmfold --env proto-env
TIMEOUT_SCALE = _resolve_timeout_scale()

# Per-service Modal container wall, in seconds. This is the SINGLE source of
# truth: each service's ``@app.cls(timeout=...)`` reads its value from here.
#
# Modal restarts the wall on every retry, so a wedged call can bill up to
# ``(1 + max_retries) x`` this value. See SERVICE_RETRIES in app.py — and
# :func:`retries_for_service`, which is why that multiplier is 1 on the batch tier.
#
# Clamped to Modal's ceiling: ``batch`` already sits on it, so any TIMEOUT_SCALE above 1 would
# otherwise push it past what Modal accepts and fail the deploy of a service the scale was never
# aimed at. Scaling still lifts every shorter tier.
SERVICE_MODAL_TIMEOUTS: dict[str, int] = {
    service: min(int(TIER_SECONDS[tier] * TIMEOUT_SCALE), MODAL_MAX_TIMEOUT_SECONDS)
    for service, tier in SERVICE_TIERS.items()
}


# Services whose containers are scheduled with a GPU. Callers translate a
# logical device ("proto"/"modal") into "cuda" for these and "cpu" otherwise —
# proto-tools' BaseConfig defaults to "cpu", which would otherwise run a model
# on the CPU of a GPU container.
GPU_SERVICES: frozenset[str] = frozenset(
    {
        "AbLangService",
        "AlphaFold2Service",
        "AlphaGenomeService",
        "BioEmuService",
        "Boltz2Service",
        "BorzoiService",
        "Chai1Service",
        "ESM2Service",
        "ESM3Service",
        "ESMCService",
        "ESMFold2Service",
        "ESMFoldService",
        "ESMIF1Service",
        "EnformerService",
        "Evo1Service",
        "Evo2Service",
        "FAMPNNService",
        "FreeBindCraftService",
        "LigandMPNNService",
        "MalinoisService",
        "Metal3DService",
        "OpenDDEService",
        "PangolinService",
        "ParadeService",
        "ProGen2Service",
        "ProGen3Service",
        "ProteinMPNNService",
        "ProtenixService",
        "PuffinService",
        "RF3Service",
        "RFdiffusion3Service",
        "SpliceAIService",
        "SpliceTransformerService",
    }
)


def physical_device_for_service(service_class_name: str) -> str:
    """Return the device string a service's container can actually use."""
    return "cuda" if service_class_name in GPU_SERVICES else "cpu"


def runs_for_hours(service_class_name: str) -> bool:
    """Whether a service is on the batch tier, i.e. one call can run for most of a day.

    The single predicate behind the batch-tier cost policy, so retries, the deploy-time wall, and
    the caller-facing warning cannot disagree about which services it covers. An unknown service
    reads as False: it has no tier yet, and guessing "expensive" for it would warn about work that
    finishes in seconds.
    """
    return SERVICE_TIERS.get(service_class_name) == BATCH_TIER


# Service → defining module's import path. Drives generated entrypoints.
SERVICE_TO_MODULE: dict[str, str] = {
    "AbLangService": "proto_tools.modal.masked_models.ablang_deployment.ablang_service",
    "AlphaFold2Service": "proto_tools.modal.structure_prediction.alphafold2_deployment.alphafold2_service",
    "AlphaGenomeService": "proto_tools.modal.sequence_scoring.alphagenome_deployment.alphagenome_service",
    "BioEmuService": "proto_tools.modal.structure_dynamics.bioemu_deployment.bioemu_service",
    "Boltz2Service": "proto_tools.modal.structure_prediction.boltz2_deployment.boltz2_service",
    "BorzoiService": "proto_tools.modal.sequence_scoring.borzoi_deployment.borzoi_service",
    "CcdLookupService": "proto_tools.modal.database_retrieval.ccd_lookup_deployment.ccd_lookup_service",
    "Chai1Service": "proto_tools.modal.structure_prediction.chai1_deployment.chai1_service",
    "CrisprTracrRNAService": "proto_tools.modal.gene_annotation.crispr_tracr_rna_deployment.crispr_tracr_rna_service",
    "DSSPService": "proto_tools.modal.structure_scoring.dssp_deployment.dssp_service",
    "ESM2Service": "proto_tools.modal.masked_models.esm2_deployment.esm2_service",
    "ESM3Service": "proto_tools.modal.masked_models.esm3_deployment.esm3_service",
    "ESMCService": "proto_tools.modal.masked_models.esmc_deployment.esmc_service",
    "ESMFold2Service": "proto_tools.modal.structure_prediction.esmfold2_deployment.esmfold2_service",
    "ESMFoldService": "proto_tools.modal.structure_prediction.esmfold_deployment.esmfold_service",
    "ESMIF1Service": "proto_tools.modal.inverse_folding.esm_if1_deployment.esm_if1_service",
    "EnformerService": "proto_tools.modal.sequence_scoring.enformer_deployment.enformer_service",
    "Evo1Service": "proto_tools.modal.causal_models.evo1_deployment.evo1_service",
    "Evo2Service": "proto_tools.modal.causal_models.evo2_deployment.evo2_service",
    "FAMPNNService": "proto_tools.modal.inverse_folding.fampnn_deployment.fampnn_service",
    "FoldmasonService": "proto_tools.modal.structure_alignment.foldmason_deployment.foldmason_service",
    "FoldseekService": "proto_tools.modal.structure_alignment.foldseek_deployment.foldseek_service",
    "FreeBindCraftService": "proto_tools.modal.binder_design.freebindcraft_deployment.freebindcraft_service",
    "IPSAEService": "proto_tools.modal.structure_scoring.ipsae_deployment.ipsae_service",
    "LigandMPNNService": "proto_tools.modal.inverse_folding.ligandmpnn_deployment.ligandmpnn_service",
    "MafftAlignService": "proto_tools.modal.sequence_alignment.mafft_deployment.mafft_service",
    "MalinoisService": "proto_tools.modal.sequence_scoring.malinois_deployment.malinois_service",
    "Metal3DService": "proto_tools.modal.structure_scoring.metal3d_deployment.metal3d_service",
    "OpenDDEService": "proto_tools.modal.structure_prediction.opendde_deployment.opendde_service",
    "MincedService": "proto_tools.modal.gene_annotation.minced_deployment.minced_service",
    "MirandaService": "proto_tools.modal.gene_annotation.miranda_deployment.miranda_service",
    "OrfipyService": "proto_tools.modal.orf_prediction.orf_deployment.orf_service",
    "PDockQ2Service": "proto_tools.modal.structure_scoring.pdockq2_deployment.pdockq2_service",
    "PangolinService": "proto_tools.modal.rna_splicing.pangolin_deployment.pangolin_service",
    "ParadeService": "proto_tools.modal.sequence_scoring.parade_deployment.parade_service",
    "ProGen2Service": "proto_tools.modal.causal_models.progen2_deployment.progen2_service",
    "ProGen3Service": "proto_tools.modal.causal_models.progen3_deployment.progen3_service",
    "ProdigalService": "proto_tools.modal.orf_prediction.prodigal_deployment.prodigal_service",
    "PromoterCalculatorService": (
        "proto_tools.modal.gene_annotation.promoter_calculator_deployment.promoter_calculator_service"
    ),
    "ProteinMPNNService": "proto_tools.modal.inverse_folding.proteinmpnn_deployment.proteinmpnn_service",
    "ProtenixService": "proto_tools.modal.structure_prediction.protenix_deployment.protenix_service",
    "PuffinService": "proto_tools.modal.sequence_scoring.puffin_deployment.puffin_service",
    "PyHmmerService": "proto_tools.modal.gene_annotation.pyhmmer_deployment.pyhmmer_service",
    "PyMOLService": "proto_tools.modal.structure_alignment.pymol_deployment.pymol_service",
    "PyRosettaService": "proto_tools.modal.structure_scoring.pyrosetta_deployment.pyrosetta_service",
    "RF3Service": "proto_tools.modal.structure_prediction.rf3_deployment.rf3_service",
    "RFdiffusion3Service": "proto_tools.modal.structure_design.rfdiffusion3_deployment.rfdiffusion3_service",
    "SegmaskerService": "proto_tools.modal.sequence_utils.segmasker_deployment.segmasker_service",
    "SpliceAIService": "proto_tools.modal.rna_splicing.spliceai_deployment.spliceai_service",
    "SpliceTransformerService": "proto_tools.modal.rna_splicing.splice_transformer_deployment.splice_transformer_service",
    "StructureMetricsService": (
        "proto_tools.modal.structure_scoring.structure_metrics_deployment.structure_metrics_service"
    ),
    "TMalignService": "proto_tools.modal.structure_alignment.tmalign_deployment.tmalign_service",
    "USalignService": "proto_tools.modal.structure_alignment.usalign_deployment.usalign_service",
    "ViennaRNAService": "proto_tools.modal.structure_prediction.viennarna_deployment.viennarna_service",
}


def _check_manifest_complete() -> None:
    """Fail loudly at import if a service is missing a timeout or module path."""
    for service in SERVICE_TO_APP:
        if service not in SERVICE_MODAL_TIMEOUTS:
            raise ValueError(f"{service} is in APP_BUCKETS but has no SERVICE_MODAL_TIMEOUTS entry")
        if service not in SERVICE_TO_MODULE:
            raise ValueError(f"{service} is in APP_BUCKETS but has no SERVICE_TO_MODULE entry")


_check_manifest_complete()


def get_app_name_for_service(service_class_name: str) -> str:
    """Return the Modal app name owning ``service_class_name``.

    Raises:
        KeyError: If the service is not listed in :data:`APP_BUCKETS`.
    """
    try:
        return SERVICE_TO_APP[service_class_name]
    except KeyError:
        raise KeyError(
            f"{service_class_name} is not listed in APP_BUCKETS — add it to proto_tools/modal/manifest.py"
        ) from None


def app_slug(app_name: str) -> str:
    """Return the short CLI slug for a full app name (``proto-tools-esm2`` → ``esm2``)."""
    return app_name.removeprefix("proto-tools-")


def module_name(app_name: str) -> str:
    """Python module name for an app's entrypoint.

    App names keep hyphens (``proto-tools-splice-transformer``); module names
    cannot, so the filename swaps them for underscores.
    """
    return app_slug(app_name).replace("-", "_")


def app_name_for_slug(slug: str) -> str:
    """Return the full app name for a CLI slug, accepting either form.

    Raises:
        KeyError: If neither ``slug`` nor ``proto-tools-{slug}`` is a known app.
    """
    if slug in APP_BUCKETS:
        return slug
    full = f"proto-tools-{slug}"
    if full in APP_BUCKETS:
        return full
    if (reason := _undeployable_reason(slug)) is not None:
        raise KeyError(reason)
    known = ", ".join(sorted(app_slug(name) for name in APP_BUCKETS))
    raise KeyError(f"unknown app {slug!r} — known apps: {known}")


def _undeployable_reason(slug: str) -> str | None:
    """Explain a name that is a real tool or toolkit but has no deployment, or ``None``.

    Without this, naming one gets "unknown app" plus a list of 50-odd names to scan, which reads as
    a typo. The names most likely to be tried are exactly the ones a user has seen in the catalogue
    — and a tool refused on purpose is not a typo, it is a decision someone recorded a reason for.
    """
    try:
        from proto_tools.tools import ToolRegistry

        specs = ToolRegistry.list_all()
    except Exception:  # a catalogue failure must not replace the plain unknown-app error
        return None

    matches = [s for s in specs if s.key == slug or s.source_file.parent.name == slug.replace("-", "_")]
    if not matches:
        return None
    refused = [s for s in matches if s.local_only]
    if len(refused) == len(matches):
        return f"{slug!r} has no deployment and cannot have one: {refused[0].local_only}"
    return f"{slug!r} is a tool, not a deployable app. Deploy the app that serves it — see `proto-tools deploy --list`."
