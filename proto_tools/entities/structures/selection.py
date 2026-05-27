"""Reusable per-chain and per-residue selection primitives for tool inputs."""

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities.structures.structure import Structure
from proto_tools.entities.structures.utils import SUPPORTED_EXTENSIONS
from proto_tools.utils.sequence import validate_positions_list

logger = logging.getLogger(__name__)


class ChainSelection(BaseModel):
    """Selection of one or more whole chains.

    Replaces ``chain_ids: list[str] | None`` fields that select chains without
    per-residue granularity. Empty selections are rejected; the absent-selection
    state is the parent field being ``None``.

    Accepts shorthand at construction:

    - ``"A"``        → ``ChainSelection(chains=["A"])``
    - ``["A", "B"]`` → ``ChainSelection(chains=["A", "B"])``

    Attributes:
        chains (list[str]): Chain IDs in the selection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chains: list[str] = Field(title="Chains", description="Chain IDs in the selection.")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, ChainSelection):
            return {"chains": list(data.chains)}
        if isinstance(data, str):
            return {"chains": [data]}
        if isinstance(data, list):
            if not all(isinstance(c, str) for c in data):
                raise ValueError("List must contain only chain ID strings")
            return {"chains": list(data)}
        if isinstance(data, dict):
            keys = set(data.keys())
            if keys == {"chains"} and isinstance(data["chains"], list):
                return data
            if keys == {"chains"}:
                raise ValueError(
                    f"ChainSelection 'chains' must be a list of chain IDs; got {type(data['chains']).__name__}.",
                )
            if "chains" in keys:
                raise ValueError(
                    f"ChainSelection dict must use only the 'chains' key; got mixed keys {sorted(keys)}.",
                )
        raise ValueError(f"Cannot coerce {type(data).__name__} to ChainSelection")

    @model_validator(mode="after")
    def _reject_empty(self) -> ChainSelection:
        if not self.chains:
            raise ValueError(
                "ChainSelection cannot be empty; use None at the parent field instead.",
            )
        return self

    def validate_against(self, structure: Structure, label: str = "selection") -> None:
        """Raise if any selected chain is missing from ``structure``.

        Args:
            structure (Structure): The structure to check against.
            label (str): Prefix for error messages, typically the parent field name.

        Raises:
            ValueError: If any selected chain ID is absent from ``structure``.
        """
        available = set(structure.get_chain_ids())
        missing = set(self.chains) - available
        if missing:
            raise ValueError(
                f"{label}: chain(s) {sorted(missing)} not in structure (available: {sorted(available)})",
            )


class SingleChainSelection(BaseModel):
    """Selection of exactly one whole chain.

    The single-chain analog of ``ChainSelection``, for tools that operate on one
    chain at a time (e.g. DSSP, which reports percentages for one chain). The
    absent-selection state is the parent field being ``None``.

    Accepts shorthand at construction:

    - ``"A"``          → ``SingleChainSelection(chain="A")``
    - ``{"chain": "A"}`` → ``SingleChainSelection(chain="A")``

    Attributes:
        chain (str): The selected chain ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chain: str = Field(title="Chain", description="Chain ID to select.")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, SingleChainSelection):
            return {"chain": data.chain}
        if isinstance(data, str):
            return {"chain": data}
        if isinstance(data, list):
            raise ValueError(
                "SingleChainSelection takes a single chain ID; use ChainSelection for more than one chain.",
            )
        if isinstance(data, dict):
            keys = set(data.keys())
            if keys == {"chain"}:
                return data
            raise ValueError(
                f"SingleChainSelection dict must use only the 'chain' key; got {sorted(keys)}.",
            )
        raise ValueError(f"Cannot coerce {type(data).__name__} to SingleChainSelection")

    @model_validator(mode="after")
    def _reject_empty(self) -> SingleChainSelection:
        if not self.chain:
            raise ValueError(
                "SingleChainSelection chain must be a non-empty string; use None at the parent field instead.",
            )
        return self

    def validate_against(self, structure: Structure, label: str = "selection") -> None:
        """Raise if the selected chain is missing from ``structure``.

        Args:
            structure (Structure): The structure to check against.
            label (str): Prefix for error messages, typically the parent field name.

        Raises:
            ValueError: If the selected chain ID is absent from ``structure``.
        """
        available = set(structure.get_chain_ids())
        if self.chain not in available:
            raise ValueError(
                f"{label}: chain {self.chain!r} not in structure (available: {sorted(available)})",
            )


class ResidueSelection(BaseModel):
    """Selection of explicit per-chain residue positions.

    Replaces ``*_positions: dict[str, list[int]] | None`` fields. Every chain
    must have a non-empty list of positions; whole-chain semantics are not
    part of this primitive (use ``ChainSelection`` for that).

    **All positions are 1-indexed** (matching biological residue numbering and
    the rest of proto-tools); position ``1`` is the first residue in the chain.

    Accepts at construction (positions are 1-indexed):

    - ``{"A": [1, 2, 3]}`` → ``ResidueSelection(chains={"A": [1, 2, 3]})``
    - ``{"A": [1, 2], "B": [10, 11]}``

    Attributes:
        chains (dict[str, list[int]]): Per-chain residue positions, 1-indexed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chains: dict[str, list[int]] = Field(
        title="Chains",
        description="Per-chain residue positions (1-indexed; position 1 = first residue).",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, ResidueSelection):
            return {"chains": cls._dedup_positions({k: list(v) for k, v in data.chains.items()})}
        if isinstance(data, dict):
            keys = set(data.keys())
            # Canonical kwargs form: {"chains": {...}}.
            if keys == {"chains"} and isinstance(data["chains"], dict):
                return {"chains": cls._dedup_positions(dict(data["chains"]))}
            # Per-chain positions shorthand: {"A": [...], "B": [...]}.
            if "chains" not in keys:
                return {"chains": cls._dedup_positions(dict(data))}
            # Reject mixed/ambiguous dicts up front so the error message is clear.
            raise ValueError(
                "ResidueSelection dict must be either {'chains': {...}} or "
                f"{{chain_id: [positions]}}; got mixed keys {sorted(keys)}.",
            )
        raise ValueError(f"Cannot coerce {type(data).__name__} to ResidueSelection")

    @staticmethod
    def _dedup_positions(chains: dict[str, Any]) -> dict[str, Any]:
        """Validate + dedupe each chain's positions; drop chains with empty lists (with warning)."""
        cleaned: dict[str, Any] = {}
        for chain_id, positions in chains.items():
            # Non-list values pass through; field-type coercion handles the type error.
            if not isinstance(positions, list):
                cleaned[chain_id] = positions
                continue
            if not positions:
                logger.warning(
                    "ResidueSelection chain %r: dropping chain with empty position list.",
                    chain_id,
                )
                continue
            cleaned[chain_id] = validate_positions_list(
                positions,
                label=f"chain {chain_id!r}",
                logger_obj=logger,
            )
        return cleaned

    @model_validator(mode="after")
    def _validate_shape(self) -> ResidueSelection:
        if not self.chains:
            raise ValueError(
                "ResidueSelection cannot be empty; use None at the parent field instead.",
            )
        return self

    @property
    def chain_ids(self) -> list[str]:
        """Chain IDs covered by the selection (preserves insertion order)."""
        return list(self.chains)

    def positions_for(self, chain_id: str) -> list[int]:
        """Selected positions in ``chain_id``."""
        return self.chains[chain_id]

    def validate_against(self, structure: Structure, label: str = "selection") -> None:
        """Raise if any selected chain or position is missing from ``structure``.

        Args:
            structure (Structure): The structure to check against.
            label (str): Prefix for error messages, typically the parent field name.

        Raises:
            ValueError: If any selected chain is absent from the structure, or
                any selected position is not a real residue in its chain.
        """
        available = set(structure.get_chain_ids())
        for chain_id, positions in self.chains.items():
            if chain_id not in available:
                raise ValueError(
                    f"{label}: chain {chain_id!r} not in structure (available: {sorted(available)})",
                )
            valid = set(structure.get_chain_positions(chain_id))
            invalid = set(positions) - valid
            if invalid:
                raise ValueError(
                    f"{label}: invalid positions {sorted(invalid)} for chain {chain_id!r}",
                )


class StructureInputBase(BaseModel):
    """Base for tool inputs that pair a Structure with one or more selection fields.

    Subclasses inherit ``extra="forbid"``, the ``structure`` field, a
    ``mode="before"`` resolver that accepts ``str | Path | Structure | dict``,
    and a ``mode="after"`` validator that finds every direct ChainSelection- or
    ResidueSelection-typed field and validates it against ``self.structure``,
    using the field name as the error label.

    Single-chain shorthand: when the resolved structure has exactly one chain,
    every ``ResidueSelection``-typed field on the subclass also accepts a bare
    ``list[int]`` of positions, automatically coerced to ``{<chain>: [...]}``.
    Passing a bare list against a multi-chain structure raises a clear error
    asking the caller to disambiguate.

    Tools whose inputs don't fit this shape (no structure, multiple structures,
    selections nested in lists/dicts) can opt out by not inheriting.

    Attributes:
        structure (Structure): The structure all selections are validated against.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure = Field(
        title="Input Structure",
        description="Structure (Path | PDB string | Structure | dict).",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_structure(cls, data: Any) -> Any:
        if isinstance(data, (str, Path, Structure)):
            data = {"structure": data}
        if not isinstance(data, dict):
            return data
        structure = data.get("structure")
        if isinstance(structure, (str, Path)):
            # Distinguish file paths from raw PDB/CIF content strings by extension.
            if str(structure).lower().endswith(SUPPORTED_EXTENSIONS):
                data = {**data, "structure": Structure.from_file(structure)}
            else:
                data = {**data, "structure": Structure(structure=str(structure))}
        elif isinstance(structure, dict):
            data = {**data, "structure": Structure(**structure)}

        # Single-chain shorthand: bare list[int] → {chain: [...]} when structure has one chain.
        resolved = data.get("structure")
        if isinstance(resolved, Structure):
            chain_ids = resolved.get_chain_ids()
            for name, field_info in cls.model_fields.items():
                if name == "structure":
                    continue
                if not _annotation_includes(field_info.annotation, ResidueSelection):
                    continue
                value = data.get(name)
                if not _is_position_list(value):
                    continue
                if len(chain_ids) != 1:
                    raise ValueError(
                        f"{name}: bare position list {value} requires a "
                        f"single-chain structure; got chains {chain_ids}. "
                        f"Use {{chain: [positions]}} to disambiguate.",
                    )
                data = {**data, name: {chain_ids[0]: list(value)}}
        return data

    @model_validator(mode="after")
    def _validate_selections(self) -> StructureInputBase:
        for name in type(self).model_fields:
            value = getattr(self, name, None)
            if isinstance(value, (ChainSelection, SingleChainSelection, ResidueSelection)):
                value.validate_against(self.structure, label=name)
        return self


def _annotation_includes(annotation: Any, target: type) -> bool:
    """True if ``target`` is ``annotation`` or one of its union members."""
    if annotation is target:
        return True
    return target in typing.get_args(annotation)


def _is_position_list(value: Any) -> bool:
    """True if ``value`` is a non-empty list of ints (a residue-position list shorthand)."""
    return isinstance(value, list) and bool(value) and all(isinstance(p, int) for p in value)
