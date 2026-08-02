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
    "proto-tools-bioemu": ["BioEmuService"],
    "proto-tools-boltz2": ["Boltz2Service"],
    "proto-tools-borzoi": ["BorzoiService"],
    "proto-tools-chai1": ["Chai1Service"],
    "proto-tools-enformer": ["EnformerService"],
    "proto-tools-esm2": ["ESM2Service"],
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
    "proto-tools-pangolin": ["PangolinService"],
    "proto-tools-progen2": ["ProGen2Service"],
    "proto-tools-proteinmpnn": ["ProteinMPNNService"],
    "proto-tools-protenix": ["ProtenixService"],
    "proto-tools-rf3": ["RF3Service"],
    "proto-tools-rfdiffusion3": ["RFdiffusion3Service"],
    "proto-tools-splice-transformer": ["SpliceTransformerService"],
    # CPU services — one app each, so deploying one does not build the others.
    "proto-tools-crispr-tracr-rna": ["CrisprTracrRNAService"],
    "proto-tools-dssp": ["DSSPService"],
    "proto-tools-ipsae": ["IPSAEService"],
    "proto-tools-mafft": ["MafftAlignService"],
    "proto-tools-minced": ["MincedService"],
    "proto-tools-orfipy": ["OrfipyService"],
    "proto-tools-pymol": ["PyMOLService"],
    "proto-tools-segmasker": ["SegmaskerService"],
    "proto-tools-tmalign": ["TMalignService"],
    "proto-tools-usalign": ["USalignService"],
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
    "batch": 86400,  # 24 hours — full design pipelines
}

# Wall tier per service. Raising a service to a longer tier is always safe; shortening one can fail
# work that used to complete, so treat a reduction as a change that needs measurement behind it.
SERVICE_TIERS: dict[str, str] = {
    "AbLangService": "long",
    "AlphaFold2Service": "long",
    "BioEmuService": "extended",
    "Boltz2Service": "long",
    "BorzoiService": "long",
    "Chai1Service": "long",
    "CrisprTracrRNAService": "extended",
    "DSSPService": "fast",
    "ESM2Service": "long",
    "ESMCService": "long",
    "ESMFold2Service": "long",
    "ESMFoldService": "long",
    "ESMIF1Service": "long",
    "EnformerService": "medium",
    "Evo1Service": "long",
    "Evo2Service": "long",
    "FAMPNNService": "long",
    "FreeBindCraftService": "batch",
    "IPSAEService": "fast",
    # Raised from medium: ligandmpnn-score chunks 64 items, the same as proteinmpnn-score, which
    # gets an hour for the same work.
    "LigandMPNNService": "long",
    "MafftAlignService": "medium",
    "MalinoisService": "medium",
    "Metal3DService": "long",
    "MincedService": "medium",
    "OrfipyService": "fast",
    "PangolinService": "long",
    "ProGen2Service": "long",
    "ProteinMPNNService": "long",
    "ProtenixService": "long",
    "PyMOLService": "fast",
    "RF3Service": "long",
    "RFdiffusion3Service": "long",
    "SegmaskerService": "fast",
    "SpliceTransformerService": "medium",
    "TMalignService": "fast",
    "USalignService": "fast",
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
# ``(1 + max_retries) x`` this value. See SERVICE_RETRIES in app.py.
SERVICE_MODAL_TIMEOUTS: dict[str, int] = {
    service: int(TIER_SECONDS[tier] * TIMEOUT_SCALE) for service, tier in SERVICE_TIERS.items()
}


# Services whose containers are scheduled with a GPU. Callers translate a
# logical device ("proto"/"modal") into "cuda" for these and "cpu" otherwise —
# proto-tools' BaseConfig defaults to "cpu", which would otherwise run a model
# on the CPU of a GPU container.
GPU_SERVICES: frozenset[str] = frozenset(
    {
        "AbLangService",
        "AlphaFold2Service",
        "BioEmuService",
        "Boltz2Service",
        "BorzoiService",
        "Chai1Service",
        "ESM2Service",
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
        "PangolinService",
        "ProGen2Service",
        "ProteinMPNNService",
        "ProtenixService",
        "RF3Service",
        "RFdiffusion3Service",
        "SpliceTransformerService",
    }
)


def physical_device_for_service(service_class_name: str) -> str:
    """Return the device string a service's container can actually use."""
    return "cuda" if service_class_name in GPU_SERVICES else "cpu"


# Service → defining module's import path. Drives generated entrypoints.
SERVICE_TO_MODULE: dict[str, str] = {
    "AbLangService": "proto_tools.modal.masked_models.ablang_deployment.ablang_service",
    "AlphaFold2Service": "proto_tools.modal.structure_prediction.alphafold2_deployment.alphafold2_service",
    "BioEmuService": "proto_tools.modal.structure_dynamics.bioemu_deployment.bioemu_service",
    "Boltz2Service": "proto_tools.modal.structure_prediction.boltz2_deployment.boltz2_service",
    "BorzoiService": "proto_tools.modal.sequence_scoring.borzoi_deployment.borzoi_service",
    "Chai1Service": "proto_tools.modal.structure_prediction.chai1_deployment.chai1_service",
    "CrisprTracrRNAService": "proto_tools.modal.gene_annotation.crispr_tracr_rna_deployment.crispr_tracr_rna_service",
    "DSSPService": "proto_tools.modal.structure_scoring.dssp_deployment.dssp_service",
    "ESM2Service": "proto_tools.modal.masked_models.esm2_deployment.esm2_service",
    "ESMCService": "proto_tools.modal.masked_models.esmc_deployment.esmc_service",
    "ESMFold2Service": "proto_tools.modal.structure_prediction.esmfold2_deployment.esmfold2_service",
    "ESMFoldService": "proto_tools.modal.structure_prediction.esmfold_deployment.esmfold_service",
    "ESMIF1Service": "proto_tools.modal.inverse_folding.esm_if1_deployment.esm_if1_service",
    "EnformerService": "proto_tools.modal.sequence_scoring.enformer_deployment.enformer_service",
    "Evo1Service": "proto_tools.modal.causal_models.evo1_deployment.evo1_service",
    "Evo2Service": "proto_tools.modal.causal_models.evo2_deployment.evo2_service",
    "FAMPNNService": "proto_tools.modal.inverse_folding.fampnn_deployment.fampnn_service",
    "FreeBindCraftService": "proto_tools.modal.binder_design.freebindcraft_deployment.freebindcraft_service",
    "IPSAEService": "proto_tools.modal.structure_scoring.ipsae_deployment.ipsae_service",
    "LigandMPNNService": "proto_tools.modal.inverse_folding.ligandmpnn_deployment.ligandmpnn_service",
    "MafftAlignService": "proto_tools.modal.sequence_alignment.mafft_deployment.mafft_service",
    "MalinoisService": "proto_tools.modal.sequence_scoring.malinois_deployment.malinois_service",
    "Metal3DService": "proto_tools.modal.structure_scoring.metal3d_deployment.metal3d_service",
    "MincedService": "proto_tools.modal.gene_annotation.minced_deployment.minced_service",
    "OrfipyService": "proto_tools.modal.orf_prediction.orf_deployment.orf_service",
    "PangolinService": "proto_tools.modal.rna_splicing.pangolin_deployment.pangolin_service",
    "ProGen2Service": "proto_tools.modal.causal_models.progen2_deployment.progen2_service",
    "ProteinMPNNService": "proto_tools.modal.inverse_folding.proteinmpnn_deployment.proteinmpnn_service",
    "ProtenixService": "proto_tools.modal.structure_prediction.protenix_deployment.protenix_service",
    "PyMOLService": "proto_tools.modal.structure_alignment.pymol_deployment.pymol_service",
    "RF3Service": "proto_tools.modal.structure_prediction.rf3_deployment.rf3_service",
    "RFdiffusion3Service": "proto_tools.modal.structure_design.rfdiffusion3_deployment.rfdiffusion3_service",
    "SegmaskerService": "proto_tools.modal.sequence_utils.segmasker_deployment.segmasker_service",
    "SpliceTransformerService": "proto_tools.modal.rna_splicing.splice_transformer_deployment.splice_transformer_service",
    "TMalignService": "proto_tools.modal.structure_alignment.tmalign_deployment.tmalign_service",
    "USalignService": "proto_tools.modal.structure_alignment.usalign_deployment.usalign_service",
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
    known = ", ".join(sorted(app_slug(name) for name in APP_BUCKETS))
    raise KeyError(f"unknown app {slug!r} — known apps: {known}")
