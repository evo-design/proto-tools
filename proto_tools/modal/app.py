"""Modal app factory, shared secrets, and the model-weight cache volume."""

import functools
import logging
import os

import modal

from proto_tools.modal.manifest import get_app_name_for_service

logger = logging.getLogger(__name__)

# HuggingFace token for weight downloads. Gated repositories need one, and
# anonymous downloads are rate-limited. Resolution order, highest first.
#
#   PROTO_MODAL_HF_SECRET=<name>  a Modal secret you manage yourself
#   PROTO_MODAL_HF_SECRET=none    anonymous, never send a token
#   (unset)                       the token this machine already has, if any
#
# The last case reads HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or the file written by
# ``hf auth login``, and passes it to your own Modal workspace. A read-scoped
# token is sufficient.
HF_SECRET_NAME = os.getenv("PROTO_MODAL_HF_SECRET")


def _hf_token_secret() -> modal.Secret:
    """Build the HuggingFace secret every service attaches."""
    if HF_SECRET_NAME and HF_SECRET_NAME.lower() != "none":
        return modal.Secret.from_name(HF_SECRET_NAME)
    # Inside a container the token already arrived via this secret; re-reading the
    # environment there would only rewrap it.
    if HF_SECRET_NAME or not modal.is_local():
        return modal.Secret.from_dict({})

    from proto_tools.utils.auth import resolve_hf_token

    token = resolve_hf_token()
    if not token:
        return modal.Secret.from_dict({})
    # Recorded rather than printed. Sending the token to the caller's own workspace is worth an
    # audit trail, but announcing it on every import pushes the deploy result off the end of the
    # output. ``PROTO_MODAL_HF_SECRET`` is the documented way to manage the secret explicitly.
    logger.debug("using this machine's HuggingFace token for weight downloads")
    return modal.Secret.from_dict({"HF_TOKEN": token})


HF_TOKEN_SECRET = _hf_token_secret()

# The Modal environment proto-tools deploys into and dispatches to, unless told otherwise.
#
# Naming one matters more than which name it is. Without it, both sides fall back to whatever
# the Modal profile happens to point at, which is ordinarily production -- so a deploy can land
# on a live app, and a call can reach an app of the same name that some other project deployed.
# That is not hypothetical: a call once reached a protenix deployed months earlier by a different
# project, whose older proto-tools rejected the transport envelope outright.
DEFAULT_ENVIRONMENT = "proto-env"

ENVIRONMENT_VAR = "MODAL_ENVIRONMENT"


def resolve_environment(explicit: str | None = None) -> str:
    """Return the Modal environment to use, most specific source first.

    Args:
        explicit (str | None): A name from ``--env`` or an ``environment=`` argument.

    Returns:
        str: ``explicit``, else ``MODAL_ENVIRONMENT``, else :data:`DEFAULT_ENVIRONMENT`.
    """
    return explicit or os.environ.get(ENVIRONMENT_VAR) or DEFAULT_ENVIRONMENT


# The single persistent volume every service mounts at /weights for model
# weights. Defined once here so the whole deployment shares one cache and can
# never drift apart. Override the name to point at a different volume:
#
#     PROTO_MODAL_CACHE_VOLUME=my-cache python scripts/deploy.py --apps esm2 ...
CACHE_VOLUME_NAME = os.getenv("PROTO_MODAL_CACHE_VOLUME", "proto-cache")
MODEL_CACHE = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

# Shared retries — Modal retries individual inputs on transient failures.
SERVICE_RETRIES = modal.Retries(max_retries=3, initial_delay=2.0, backoff_coefficient=2.0)

# The policy applied to batch-tier services instead. Modal restarts the container wall on every
# retry, so under the batch tier's 24-hour wall the shared policy may bill up to 96 GPU-hours for a
# single call that cannot succeed. Retries also provide the least benefit for a long design
# pipeline, where a failure after several hours indicates a fault in the inputs or the model rather
# than the transient condition retries are intended to absorb.
NO_RETRIES = modal.Retries(max_retries=0)


def retries_for_service(service_class_name: str) -> modal.Retries:
    """Return the retry policy for a service, by tier.

    Batch-tier services get :data:`NO_RETRIES`; everything else gets :data:`SERVICE_RETRIES`, which
    is what those services already pass directly. Adopting this helper in a non-batch service is
    therefore a no-op, and adopting it in a new batch-tier one is what keeps the policy from being
    re-decided per service.
    """
    from proto_tools.modal.manifest import runs_for_hours

    return NO_RETRIES if runs_for_hours(service_class_name) else SERVICE_RETRIES


# Seconds an idle container stays alive before Modal scales it down. A container
# holds its model loaded on the GPU, so a call arriving inside this window skips
# both the container start and the model load.
#
# Thirty seconds covers a burst of calls arriving together while keeping idle GPU time
# short, which is the safer default to bill someone by surprise. Raise it for interactive
# work, where reading output and typing the next call takes longer than the window.
# Baked in at deploy time:
#
#     PROTO_MODAL_SCALEDOWN_WINDOW=300 proto-tools deploy --apps esmc --env proto-env
SCALEDOWN_WINDOW = int(os.getenv("PROTO_MODAL_SCALEDOWN_WINDOW", "30"))


@functools.cache
def get_app(name: str) -> modal.App:
    """Return the memoized ``modal.App`` for ``name``."""
    return modal.App(name)


def get_app_for_service(service_class_name: str) -> modal.App:
    """Resolve a service class to its owning Modal app via the manifest."""
    return get_app(get_app_name_for_service(service_class_name))


__all__ = [
    "CACHE_VOLUME_NAME",
    "HF_TOKEN_SECRET",
    "MODEL_CACHE",
    "NO_RETRIES",
    "SERVICE_RETRIES",
    "get_app",
    "get_app_for_service",
    "retries_for_service",
]
