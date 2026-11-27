"""tests/tool_infra_tests/test_item_seeds.py.

Stochastic tools that run a batch one item at a time give each item its own seed.
"""

from __future__ import annotations

from unittest.mock import patch

from proto_tools.utils.base_config import RANDOM_SEED_UPPER_BOUND
from proto_tools.utils.fanout import derive_chunk_seed, derive_item_seed


def test_item_seeds_are_distinct_reproducible_and_keep_the_first() -> None:
    seeds = [derive_item_seed(42, index, RANDOM_SEED_UPPER_BOUND) for index in range(4)]
    assert seeds[0] == 42  # a single-item call is seeded exactly as asked
    assert len(set(seeds)) == 4
    assert seeds == [derive_item_seed(42, index, RANDOM_SEED_UPPER_BOUND) for index in range(4)]
    assert all(0 <= seed < RANDOM_SEED_UPPER_BOUND for seed in seeds)
    # The same mixing a split batch's chunks use, so the two schemes agree on what "offset" means.
    assert seeds[3] == derive_chunk_seed(42, 3, RANDOM_SEED_UPPER_BOUND)


def test_an_unseeded_call_stays_unseeded() -> None:
    assert derive_item_seed(None, 0, RANDOM_SEED_UPPER_BOUND) is None
    assert derive_item_seed(None, 3, RANDOM_SEED_UPPER_BOUND) is None


def test_alphafold2_design_gradients_send_one_seed_per_candidate() -> None:
    """Duplicate candidates must not draw identically (the tools are stochastic=True)."""
    from proto_tools.tools.structure_prediction.alphafold2.gradient_helpers import dispatch_design_gradient
    from proto_tools.tools.tool_registry import ToolRegistry

    config = ToolRegistry.get("alphafold2-fixbb-gradient").config_model(seed=7)
    items = [{"prep_kwargs": {}, "logits": [[0.0] * 20]} for _ in range(3)]
    with patch("proto_tools.utils.tool_instance.ToolInstance.dispatch", return_value={"results": []}) as dispatch:
        dispatch_design_gradient(
            protocol="fixbb", items=items, config=config, temperature=1.0, tool_key="alphafold2-fixbb-gradient"
        )
    payload = dispatch.call_args.args[1]
    seeds = [item["seed"] for item in payload["items"]]
    assert seeds == [derive_item_seed(7, index, RANDOM_SEED_UPPER_BOUND) for index in range(3)]
    assert payload["seed"] == 7  # kept for anything in the worker that seeds once per call
    assert all("seed" not in item for item in items)  # the caller's items are not modified
