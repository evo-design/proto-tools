"""proto_tools/utils/fanout.py.

Splitting one iterable call across several executions and stitching the results back.

Both the local worker pool and a remote device do the same thing: divide a batch, run the
pieces independently, and reassemble one output. Only the per-item list is genuinely split,
so every other output field is carried through from one piece. A field that differs between
pieces is per-item data in disguise, and the merged output would hold one piece's value for
the whole batch, which is why that case is reported rather than passed over.
"""

import functools
import logging
import math
import os
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, MissingAssetError

logger = logging.getLogger(__name__)

# Set PROTO_ON_PARTIAL_FAILURE=return_partial to receive the items that succeeded instead of an
# exception. An environment variable rather than a config field so it never reaches a worker: a
# config crosses to whatever proto-tools an image was built with, and ``extra="forbid"`` means a
# field that version lacks is rejected outright.
_ON_PARTIAL_FAILURE_ENV_VAR = "PROTO_ON_PARTIAL_FAILURE"


class FailedItem(BaseModel):
    """Stands in for one item whose execution failed, keeping the output aligned with the input.

    Occupies the position an item would have held so a caller can still line results up against
    what they sent, and names why that position is empty rather than leaving them to guess.
    """

    error: str = Field(description="Why the execution covering this item failed.")


@functools.cache
def _warn_unrecognised_partial_failure_value(value: str) -> None:
    """Say so once per distinct value, rather than on every dispatch."""
    logger.warning(
        "%s=%r is not recognised; falling back to raising. Did you mean 'return_partial'?",
        _ON_PARTIAL_FAILURE_ENV_VAR,
        value,
    )


def is_fatal_dispatch_error(exc: BaseException) -> bool:
    """Whether a failure condemns the whole call rather than one chunk of it.

    Missing credentials, an undeployed tool, an uninstalled client: none of these are properties of
    the items being sent, so every chunk would fail the same way. Recording them per chunk would
    hand back one placeholder per item, each repeating the same setup problem, and bury the single
    thing the caller has to act on. Retrying them is equally pointless.

    Args:
        exc (BaseException): Failure raised while dispatching.

    Returns:
        bool: True when the failure should surface immediately, whatever the partial-failure
            setting says.
    """
    # MissingAssetError is documented as always propagating, and the pytest skip hook depends on
    # catching the real type to turn an unprovisioned asset into a skip. Absorbing it into a
    # placeholder would break both.
    if isinstance(exc, (ImportError, NotImplementedError, PermissionError, MissingAssetError)):
        return True
    from proto_tools.modal import ModalDispatchError

    return isinstance(exc, ModalDispatchError)


def on_partial_failure() -> Literal["raise", "return_partial"]:
    """How to report a split batch where only some executions succeeded.

    An unrecognised value falls back to raising rather than guessing, but says so: someone who
    wrote the wrong case would otherwise see the default behaviour and no indication why.

    Returns:
        Literal["raise", "return_partial"]: ``"return_partial"`` when the environment asks for the
            surviving items, otherwise ``"raise"``.
    """
    value = os.environ.get(_ON_PARTIAL_FAILURE_ENV_VAR, "")
    if value == "return_partial":
        return "return_partial"
    if value:
        _warn_unrecognised_partial_failure_value(value)
    return "raise"


# Reassembled by the merge itself, so they are expected to differ between pieces.
#
# ``metadata`` is owned in a weaker sense than the rest: no piece's copy is combined with another's,
# so one piece's survives and the others are dropped without a warning. A caller wanting a figure
# that describes the whole batch has to pass it through ``extra_updates``, as both fan-out paths do
# for ``dispatch_stats``. Per-piece telemetry, such as each container's own timing, does not survive.
MERGE_OWNED_FIELDS = frozenset({"tool_id", "success", "execution_time", "timestamp", "warnings", "errors", "metadata"})


def warn_on_piece_disagreement(tool_key: str, source: str, iterable_output_field: str, results: list[Any]) -> None:
    """Warn when a field carried through as invariant differs between pieces of one batch.

    Warns rather than raises because the framework cannot tell a genuine misclassification
    from a benign difference such as an embedded worker identifier.

    Args:
        tool_key (str): Tool whose output is being merged.
        source (str): What produced the pieces, named in the message (e.g. ``"ToolPool"``).
        iterable_output_field (str): The per-item field, which is expected to differ.
        results (list[Any]): One output per piece.
    """
    if len(results) < 2:
        return
    first, *rest = results
    for name in type(first).model_fields:
        if name in MERGE_OWNED_FIELDS or name == iterable_output_field:
            continue
        expected = getattr(first, name, None)
        if any(getattr(other, name, None) != expected for other in rest):
            logger.warning(
                "%s %s: output field %r differs between pieces of one batch but is carried through "
                "as invariant, so the merged result holds one piece's value. If it is per-item "
                "data, move it inside the %r element.",
                source,
                tool_key,
                name,
                iterable_output_field,
            )


def merge_piece_outputs(
    tool_key: str,
    source: str,
    iterable_output_field: str,
    results: list[BaseToolOutput],
    merged_items: list[Any],
    *,
    extra_updates: dict[str, Any] | None = None,
) -> BaseToolOutput:
    """Combine per-piece outputs into one, carrying invariant fields from a piece.

    Copies a piece's output rather than constructing a fresh one, since constructing would
    silently drop every field except the item list.

    Args:
        tool_key (str): Tool whose output is being merged.
        source (str): What produced the pieces, named in any warning.
        iterable_output_field (str): Field holding the per-item list.
        results (list[BaseToolOutput]): One output per piece, in any order.
        merged_items (list[Any]): The per-item values already back in original order.
        extra_updates (dict[str, Any] | None): Further fields to set on the merged output.

    Returns:
        BaseToolOutput: One output carrying the merged items.

    Raises:
        ValueError: If ``results`` is empty, since there is no output to carry fields from.
    """
    if not results:
        raise ValueError(f"{tool_key}: cannot merge an empty set of results")

    warn_on_piece_disagreement(tool_key, source, iterable_output_field, results)

    warnings: list[str] = []
    errors: list[str] = []
    for result in results:
        warnings.extend(result.warnings or [])
        errors.extend(result.errors or [])

    updates: dict[str, Any] = {
        iterable_output_field: merged_items,
        "warnings": warnings,
        "errors": errors,
        **(extra_updates or {}),
    }
    return results[0].model_copy(update=updates)


# ── Packing a batch into pieces ──────────────────────────────────────────────
#
# Both strategies cost items with :func:`item_costs`, then pack differently: ``chunk_indices`` into
# contiguous spans for a remote device, ``lpt_schedule`` into one bin per local device.


def item_costs(input_cls: type[BaseToolInput], items: list[Any]) -> list[float]:
    """Cost of each item, as the tool's own input class reports it.

    The single place either fan-out path asks what an item is worth, so the pool and a remote
    device cut the same batch the same way. A tool that does not override
    :meth:`BaseToolInput.item_cost` reports ``1.0`` for everything, which makes cost and count the
    same measure and leaves its behaviour unchanged.

    Args:
        input_cls (type[BaseToolInput]): Input class declaring the tool's cost function.
        items (list[Any]): Items from the batch's iterable field.

    Returns:
        list[float]: One cost per item, positionally aligned with ``items``.
    """
    return [float(input_cls.item_cost(item)) for item in items]


def chunk_indices(total: int, max_chunk_size: int | None, costs: list[float] | None = None) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` spans covering ``total`` items.

    Cuts so each span carries a comparable share of the batch's cost, never exceeding
    ``max_chunk_size`` items. Cost decides where a cut falls; count caps how wide it can get. A
    batch of uneven items otherwise splits into even counts of wildly different work, and the
    slowest span sets the wall clock.

    Cost can produce **more** pieces than counting alone: peeling an expensive item into its own
    span is the point, and the rest then packs behind it. Every piece still respects
    ``max_chunk_size``, and there are never more pieces than items. ``starmap`` queues pieces and
    reuses warm containers, so the extra ones cost dispatches rather than container starts.

    Spans stay contiguous and in order. The remote path derives a chunk's seed from where it starts
    and merges by concatenation, so reordering to balance further would cost both.

    Every span is computed from the whole batch's costs rather than from a running total, so the
    result does not depend on which end the batch is read from. An earlier attempt derived one
    chunk size from a prefix and applied it to the rest, which gave different splits for the same
    items in a different order.

    A chunk size of ``None`` or one at least as large as the batch yields a single span, so a
    caller can treat chunking as uniform rather than special-casing the whole-batch case.

    Args:
        total (int): Number of items in the batch.
        max_chunk_size (int | None): Largest number of items one execution may receive.
        costs (list[float] | None): Per-item costs, positionally aligned with the batch. ``None``,
            or costs that are all equal, cut exactly as counting alone would.

    Returns:
        list[tuple[int, int]]: Half-open spans in order; empty when ``total`` is zero.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive, which would not terminate, or if
            ``costs`` does not describe every item.
    """
    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
    if total <= 0:
        return []
    if costs is not None and len(costs) != total:
        raise ValueError(f"costs describes {len(costs)} item(s) for a batch of {total}")

    size = total if max_chunk_size is None else min(max_chunk_size, total)
    if costs is None or len(set(costs)) <= 1:
        return [(start, min(start + size, total)) for start in range(0, total, size)]

    # The count ceiling fixes how many spans there must be at minimum; cost then decides where the
    # cuts fall between them. Targeting an equal share per span keeps this independent of order.
    n_spans = math.ceil(total / size)
    target = sum(costs) / n_spans

    spans: list[tuple[int, int]] = []
    start = 0
    carried = 0.0
    for index, cost in enumerate(costs):
        carried += cost
        width = index + 1 - start
        remaining_spans = n_spans - len(spans)
        # Close the span once it has taken its share, or hit the count cap — but never so early
        # that the items left cannot fill the spans still owed.
        must_close = width >= size
        took_share = carried >= target and remaining_spans > 1
        enough_left = total - (index + 1) >= remaining_spans - 1
        if (must_close or took_share) and enough_left:
            spans.append((start, index + 1))
            start = index + 1
            carried = 0.0
    if start < total:
        spans.append((start, total))
    return spans


def derive_chunk_seed(base_seed: int, offset: int, upper_bound: int) -> int:
    """Seed for the chunk starting at ``offset``, distinct from the other chunks of one batch.

    A stochastic tool seeds once per execution and lets its per-item sampling advance the RNG, so
    identical inputs within a batch diverge. Splitting the batch restarts that advancement in each
    piece, and items at matching positions in different pieces would otherwise draw identically.

    Mixes rather than adds, so ``base_seed=1`` at offset 32 does not land on ``base_seed=33`` at
    offset 0. Returns ``base_seed`` unchanged at offset zero, leaving the first chunk, and any
    batch that was never split, exactly as it was.

    This does not reproduce an unsplit run. Matching one would mean knowing how many draws each
    item consumed, which depends on the tool's own sampling. Results stay reproducible for a given
    seed, batch, and chunk size.

    Args:
        base_seed (int): The seed the caller asked for.
        offset (int): Index of the chunk's first item within the original batch.
        upper_bound (int): Exclusive upper bound for the returned seed.

    Returns:
        int: Seed for this chunk.
    """
    if offset == 0:
        return base_seed
    return zlib.crc32(f"{base_seed}:{offset}".encode()) % upper_bound


@dataclass
class DeviceCapability:
    """Describes a device (or device group) available for scheduling.

    Attributes:
        device_id (str): Device string, e.g. ``"cuda:0"`` or ``"cuda:0,cuda:1"``
            for multi-GPU worker slots.
        throughput_weight (float): Relative speed of this device compared to others.
            The scheduler divides a device's accumulated cost by its weight
            to estimate finish time, so a weight of 2.0 means "twice as fast"
            and the device will be assigned roughly twice the work. **Currently
            unused**; all devices default to 1.0 (uniform). Reserved for
            future heterogeneous GPU support (e.g., mixed H100/A100 pools).
        max_item_cost (float | None): Maximum item cost this device can handle, or None for
            no limit. Items whose ``item_cost()`` exceeds this cap are routed
            to other devices that can handle them (falls back to least-loaded
            if no device qualifies). **Currently unused**; all devices
            default to None. Reserved for future VRAM-aware scheduling
            (e.g., a 24 GB GPU cannot fold a 5000-residue protein).
    """

    device_id: str
    throughput_weight: float = 1.0
    max_item_cost: float | None = None

    def __post_init__(self) -> None:
        """Reject a weight that would divide by zero when estimating finish times."""
        if self.throughput_weight <= 0:
            raise ValueError(f"throughput_weight must be positive, got {self.throughput_weight}")


@dataclass
class WorkItem:
    """A single item to be scheduled, with its original position for reassembly."""

    original_index: int
    item: Any
    cost: float


@dataclass
class WorkerAssignment:
    """Items assigned to a specific device after scheduling."""

    device: DeviceCapability
    items: list[WorkItem] = field(default_factory=list)
    total_cost: float = 0.0


def lpt_schedule(
    items: list[WorkItem],
    devices: list[DeviceCapability],
) -> list[WorkerAssignment]:
    """Cost-aware Longest Processing Time (LPT) bin-packing.

    Sorts items by cost descending, then greedily assigns each to the device
    with the lowest estimated finish time (``total_cost / throughput_weight``).
    Gives a 4/3-approximation to optimal makespan.

    With the current defaults (uniform ``throughput_weight=1.0`` and no
    ``max_item_cost`` caps), this reduces to standard LPT, which itself
    degrades to round-robin when all item costs are equal (the common case
    for tools that don't override ``BaseToolInput.item_cost()``).

    Args:
        items (list[WorkItem]): Work items with cost estimates (from ``item_cost()``).
        devices (list[DeviceCapability]): Available devices. ``throughput_weight`` and ``max_item_cost``
            are supported by the algorithm but currently unused (all devices
            get weight 1.0 and no cap).

    Returns:
        list[WorkerAssignment]: List of WorkerAssignments, one per device (devices with no items
            are included but have empty item lists).
    """
    assignments = [WorkerAssignment(device=d) for d in devices]

    # Sort items by cost descending (LPT)
    sorted_items = sorted(items, key=lambda w: w.cost, reverse=True)

    for work_item in sorted_items:
        # Filter devices that can handle this item
        eligible = [
            a for a in assignments if a.device.max_item_cost is None or work_item.cost <= a.device.max_item_cost
        ]
        if not eligible:
            # No device can handle this item; assign to least-loaded anyway
            eligible = assignments

        # Pick device with lowest estimated finish time
        best = min(eligible, key=lambda a: a.total_cost / a.device.throughput_weight)
        best.items.append(work_item)
        best.total_cost += work_item.cost

    return assignments
