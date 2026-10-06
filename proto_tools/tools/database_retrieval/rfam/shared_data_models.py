"""proto_tools/tools/database_retrieval/rfam/shared_data_models.py.

Contains the shared family input, the HTTP constants, and private helpers used
by the rfam-family and rfam-regions tool modules.
"""

import logging
from urllib.parse import quote

import requests
from pydantic import field_validator

from proto_tools.utils import BaseToolInput, InputField, request_with_retry

logger = logging.getLogger(__name__)

_RFAM_FAMILY_BASE = "https://rfam.org/family"
_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/rfam-v1"


# ============================================================================
# Data Models
# ============================================================================


class RfamFamilyQuery(BaseToolInput):
    """An Rfam family, named by accession or ID.

    Attributes:
        family (str): Rfam accession (e.g. 'RF01731') or family ID (e.g. 'TwoAYGGAY').
    """

    family: str = InputField(
        title="Family",
        description="Rfam accession (e.g. 'RF01731') or family ID (e.g. 'TwoAYGGAY')",
    )

    @field_validator("family")
    @classmethod
    def _strip_family(cls, value: str) -> str:
        """Reject a blank family before it becomes a request for /family/."""
        value = value.strip()
        if not value:
            raise ValueError("family must be an Rfam accession or ID, e.g. 'RF01731'")
        return value


# ============================================================================
# Private Helpers
# ============================================================================


def _family_url(family: str, *parts: str) -> str:
    """Build an Rfam family endpoint URL, escaping the family name."""
    return "/".join([_RFAM_FAMILY_BASE, quote(family, safe=""), *parts])


def _rfam_get(session: requests.Session, url: str, params: dict[str, str] | None = None) -> requests.Response | None:
    """GET an Rfam endpoint; return None on 404 and raise on any other HTTP error."""
    response = request_with_retry(
        lambda: session.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response


def _not_found(family: str) -> ValueError:
    """Explain an unknown family the same way for every Rfam tool."""
    return ValueError(
        f"No Rfam family matches {family!r}. Pass an accession such as 'RF01731' or an ID such as 'TwoAYGGAY'."
    )
