"""Source of truth for the per-app Modal split.

Declares which services belong to which app, their timeouts, and their modules.
"""

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


# Per-service Modal container wall, in seconds. This is the SINGLE source of
# truth: each service's ``@app.cls(timeout=...)`` reads its value from here.
# Keep these aligned with what each tool actually needs.
SERVICE_MODAL_TIMEOUTS: dict[str, int] = {
    "AbLangService": 3600,
    "AlphaFold2Service": 3600,
    "BioEmuService": 14400,
    "Boltz2Service": 3600,
    "BorzoiService": 3600,
    "Chai1Service": 3600,
    "CrisprTracrRNAService": 14400,
    "DSSPService": 600,
    "ESM2Service": 3600,
    "ESMCService": 3600,
    "ESMFold2Service": 3600,
    "ESMFoldService": 3600,
    "ESMIF1Service": 3600,
    "EnformerService": 1800,
    "Evo1Service": 3600,
    "Evo2Service": 3600,
    "FAMPNNService": 3600,
    "FreeBindCraftService": 86400,
    "IPSAEService": 600,
    "LigandMPNNService": 1800,
    "MafftAlignService": 1800,
    "MalinoisService": 1800,
    "Metal3DService": 3600,
    "MincedService": 1800,
    "OrfipyService": 600,
    "PangolinService": 3600,
    "ProGen2Service": 3600,
    "ProteinMPNNService": 3600,
    "ProtenixService": 3600,
    "PyMOLService": 600,
    "RF3Service": 3600,
    "RFdiffusion3Service": 3600,
    "SegmaskerService": 600,
    "SpliceTransformerService": 1800,
    "TMalignService": 600,
    "USalignService": 600,
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
