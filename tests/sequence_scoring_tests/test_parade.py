"""Tests for PARADE UTR activity and mRNA stability prediction."""

import math

import pytest
from pydantic import ValidationError

from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output


@pytest.fixture
def fresh_cache():
    """Install a fresh program-scoped tool cache for the duration of the test."""
    cache = ToolCache()
    token = _program_tool_cache.set(cache)
    yield cache
    _program_tool_cache.reset(token)


def _skip_if_no_gpu() -> None:
    from proto_tools.utils.device import number_of_visible_gpus

    if number_of_visible_gpus() < 1:
        pytest.skip("PARADE GPU test requires a visible CUDA GPU")


def test_parade_input_normalizes_and_validates() -> None:
    """PARADE input maps RNA U to T, allows N, and rejects other characters."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityInput

    inputs = ParadeActivityInput(sequences="acgu N")

    assert inputs.sequences == ["ACGTN"]
    assert len(inputs) == 1
    with pytest.raises(ValueError, match="Invalid nucleotide characters"):
        ParadeActivityInput(sequences="ACGTX")


def test_parade_n_base_encodes_to_zero() -> None:
    """N encodes to an all-zero column, matching upstream (0.25 assigned into an int one-hot truncates)."""
    import sys
    from pathlib import Path

    torch = pytest.importorskip("torch")
    pytest.importorskip("pandas")

    standalone = Path(__file__).resolve().parents[2] / "proto_tools/tools/sequence_scoring/parade/standalone"
    sys.path.insert(0, str(standalone))
    try:
        import utrdata_cl

        encoded = utrdata_cl.Seq2Tensor()("ACGN")  # (4 channels A,C,G,T x 4 positions)
    finally:
        sys.path.remove(str(standalone))
        sys.modules.pop("utrdata_cl", None)

    assert torch.equal(encoded[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))  # A -> one-hot
    assert torch.equal(encoded[:, 3], torch.zeros(4))  # N -> all-zero (NOT 0.25)


@pytest.mark.integration
def test_parade_checkpoint_download_security_runs_in_env() -> None:
    """Checkpoint download/security checks run INSIDE the parade standalone env.

    The logic under test lives in the standalone ``inference`` module (imports
    ``torch``) and is not importable in the base test env, so pytest would skip it.
    ``run_in_env`` runs the plain-assert suite ``_parade_env_security_checks.py``
    inside the built parade env; a non-zero exit surfaces here as ``RuntimeError``.
    """
    from pathlib import Path

    from proto_tools.utils import run_in_env

    checks = Path(__file__).resolve().parent / "_parade_env_security_checks.py"
    output = run_in_env("parade", script=str(checks), timeout=1800)
    assert "ALL CHECKS PASSED" in output, output


def test_parade_custom_checkpoint_rejected_on_proto() -> None:
    """Loading a caller's checkpoint executes its code, which Proto's shared service will not do."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig
    from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
        ParadeGradientConfig,
        ParadeGradientLossTerm,
    )

    for checkpoint in ("https://example.com/evil.ckpt", "checkpoints/model.ckpt"):
        reason = ParadeActivityConfig(checkpoint=checkpoint).remote_unsupported_reason("proto")
        assert reason is not None and "checkpoint" in reason

    assert ParadeActivityConfig().remote_unsupported_reason("proto") is None, "the pinned checkpoint is fine"

    gradient = ParadeGradientConfig(
        loss_terms=[ParadeGradientLossTerm(cell_type="c1", direction="max", weight=1.0)],
        checkpoint="https://example.com/evil.ckpt",
    )
    assert "checkpoint" in (gradient.remote_unsupported_reason("proto") or "")


def test_parade_custom_checkpoint_allowed_on_your_own_deployment() -> None:
    """A Modal workspace belongs to the caller, so running their own checkpoint there is their choice."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig

    config = ParadeActivityConfig(checkpoint="https://example.com/mine.ckpt")
    assert config.remote_unsupported_reason("modal") is None


def test_parade_checkpoint_link_vs_path_and_https_enforcement() -> None:
    """``checkpoint`` accepts local paths and https links (schemeless allowed); non-https links are rejected."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig
    from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
        ParadeGradientConfig,
        ParadeGradientLossTerm,
    )

    # A non-HTTPS link is rejected (the artifact is unpickled).
    with pytest.raises(ValueError, match="https"):
        ParadeActivityConfig(checkpoint="http://example.com/x.ckpt")
    with pytest.raises(ValueError, match="https"):
        ParadeGradientConfig(
            loss_terms=[ParadeGradientLossTerm(cell_type="c1", direction="max", weight=1.0)],
            checkpoint="http://example.com/x.ckpt",
        )
    # A schemeless link (host.tld/path) is accepted and normalized to https://.
    assert (
        ParadeActivityConfig(checkpoint="huggingface.co/org/model.ckpt").checkpoint
        == "https://huggingface.co/org/model.ckpt"
    )
    assert ParadeActivityConfig(checkpoint="https://example.com/x.ckpt").checkpoint == "https://example.com/x.ckpt"
    # Local paths pass through unchanged and are NOT treated as links.
    for path in ("/oak/checkpoints/model.ckpt", "model.ckpt", "sub/model.ckpt", "./model.ckpt"):
        assert ParadeActivityConfig(checkpoint=path).checkpoint == path
    assert ParadeActivityConfig().checkpoint == ""

    # Malformed HTTPS links (no host) are rejected at validation, not after worker startup.
    for malformed in ("https:relative", "https:///no-host", "https://"):
        with pytest.raises(ValueError, match="host"):
            ParadeActivityConfig(checkpoint=malformed)
    # Malformed ports are rejected at validation (else _redact_url crashes the worker on .port).
    for bad_port in ("https://example.com:abc/x.ckpt", "https://example.com:99999/x.ckpt"):
        with pytest.raises(ValueError, match="valid URL"):
            ParadeActivityConfig(checkpoint=bad_port)
    for malformed in (
        "https://exa mple.com/x.ckpt",
        "https://example.com/x y.ckpt",
        "https://example.com/x\x00y.ckpt",
        "https://example.com/café.ckpt",
        "https://example.com\\evil/x.ckpt",
        "https://example%.com/x.ckpt",
        "https://example|evil.com/x.ckpt",
    ):
        with pytest.raises(ValueError):
            ParadeActivityConfig(checkpoint=malformed)


def test_parade_checkpoint_secret_redacted_from_errors() -> None:
    """A credentialed/signed checkpoint link never appears in any error representation."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig
    from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
        ParadeGradientConfig,
        ParadeGradientLossTerm,
    )
    from proto_tools.tools.sequence_scoring.parade.shared_data_models import require_https_checkpoint_url

    secret_url = "https://bob:s3cr3t-token@example.com/x.ckpt"
    # The validator's own message never echoes the secret...
    with pytest.raises(ValueError, match="credentials") as excinfo:
        require_https_checkpoint_url(secret_url)
    assert "s3cr3t-token" not in str(excinfo.value)
    # ...and no public Pydantic error representation retains the rejected URL input.
    constructors = (
        lambda: ParadeActivityConfig(checkpoint=secret_url),
        lambda: ParadeActivityConfig.model_validate({"checkpoint": secret_url}),
        lambda: ParadeActivityConfig.model_validate_json(f'{{"checkpoint": "{secret_url}"}}'),
        lambda: ParadeGradientConfig(
            loss_terms=[ParadeGradientLossTerm(cell_type="c1", direction="max", weight=1.0)],
            checkpoint=secret_url,
        ),
    )
    for constructor in constructors:
        with pytest.raises(ValidationError) as config_exc:
            constructor()
        rendered = str(config_exc.value) + repr(config_exc.value.errors()) + config_exc.value.json()
        assert "s3cr3t-token" not in rendered
        assert secret_url not in rendered

    from pydantic import TypeAdapter

    scalar_or_container_validators = (
        lambda: ParadeActivityConfig.model_validate(secret_url),
        lambda: ParadeActivityConfig.model_validate_json(f'"{secret_url}"'),
        lambda: TypeAdapter(ParadeActivityConfig).validate_python(secret_url),
        lambda: TypeAdapter(ParadeActivityConfig).validate_json(f'"{secret_url}"'),
        lambda: TypeAdapter(ParadeActivityConfig).validate_python([{"checkpoint": secret_url}]),
        lambda: TypeAdapter(ParadeActivityConfig).validate_python([secret_url]),
        lambda: TypeAdapter(ParadeActivityConfig).validate_python([{"other": secret_url}]),
    )
    for validator in scalar_or_container_validators:
        with pytest.raises(ValidationError) as scalar_exc:
            validator()
        rendered = str(scalar_exc.value) + repr(scalar_exc.value.errors()) + scalar_exc.value.json()
        assert "s3cr3t-token" not in rendered

    # Unrelated field errors retain their useful input values.
    with pytest.raises(ValidationError) as unrelated_exc:
        ParadeActivityConfig(construct_type="bogus")  # type: ignore[arg-type]
    assert "bogus" in str(unrelated_exc.value)

    malformed_json = f'{{"checkpoint": "{secret_url}"'
    with pytest.raises(ValidationError) as json_exc:
        ParadeActivityConfig.model_validate_json(malformed_json)
    rendered = str(json_exc.value) + repr(json_exc.value.errors()) + json_exc.value.json()
    assert "s3cr3t-token" not in rendered
    # The redacted parse error still surfaces the parser POSITION so an operator can find the typo.
    assert "line 1" in rendered

    # A bare-string top-level input is redacted under the NEUTRAL label (it may not be a URL).
    with pytest.raises(ValidationError) as bare_exc:
        ParadeActivityConfig.model_validate(secret_url)
    bare_rendered = str(bare_exc.value) + repr(bare_exc.value.errors())
    assert "s3cr3t-token" not in bare_rendered
    assert "<redacted>" in bare_rendered

    activity = ParadeActivityConfig()
    with pytest.raises(ValueError) as assignment_exc:
        activity.checkpoint = secret_url
    assert "s3cr3t-token" not in str(assignment_exc.value)
    assert activity.checkpoint == ""


def test_parade_checkpoint_is_hidden_from_config_repr() -> None:
    """Signed checkpoint query tokens stay out of config repr while remaining available for dispatch."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeGradientConfig,
        ParadeStabilityConfig,
    )

    signed_url = "https://example.com/x.ckpt?X-Amz-Signature=s3cr3t-token"
    for config in (
        ParadeActivityConfig(checkpoint=signed_url),
        ParadeGradientConfig(checkpoint=signed_url),
        ParadeStabilityConfig(checkpoint=signed_url),
    ):
        assert "s3cr3t-token" not in repr(config)
        assert config.model_dump()["checkpoint"] == signed_url


def test_parade_activity_config_resolves_panel() -> None:
    """Empty cell_types resolves to the full panel for the construct type."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig

    assert ParadeActivityConfig(construct_type="utr5").resolved_cell_types == ["c1", "c2", "c4", "c6", "c17"]
    assert ParadeActivityConfig(construct_type="utr3").resolved_cell_types == ["c1", "c2", "c4", "c6", "c17", "c13"]


def test_parade_config_cross_field_assignments_are_atomic() -> None:
    """Default panels follow construct updates and rejected cross-field assignments do not corrupt configs."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeGradientConfig,
        ParadeGradientLossTerm,
    )

    activity = ParadeActivityConfig()
    activity.construct_type = "utr3"
    assert activity.cell_types == []
    assert activity.resolved_cell_types == ["c1", "c2", "c4", "c6", "c17", "c13"]
    activity.cell_types = ["c13"]
    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        activity.construct_type = "utr5"
    assert activity.construct_type == "utr3"
    with pytest.raises(ValidationError, match="unique"):
        activity.cell_types = ["c2", "c2"]
    assert activity.cell_types == ["c13"]

    gradient = ParadeGradientConfig(construct_type="utr3", loss_terms=[ParadeGradientLossTerm(cell_type="c13")])
    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        gradient.construct_type = "utr5"
    assert gradient.construct_type == "utr3"

    # In-place list mutation cannot trigger Pydantic assignment hooks, so execution
    # revalidates a dumped copy before any worker dispatch or scientific calculation.
    mutated_activity = ParadeActivityConfig(construct_type="utr5", cell_types=["c1"])
    mutated_activity.cell_types.append("c13")
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityInput, run_parade_activity

    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        run_parade_activity(ParadeActivityInput(sequences=["A" * 50]), mutated_activity)

    gradient.loss_terms.clear()
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientInput, run_parade_gradient

    with pytest.raises(ValidationError, match="cannot be empty"):
        run_parade_gradient(ParadeGradientInput(logits=[[[0.0] * 4] * 50]), gradient)


def test_parade_activity_config_rejects_offpanel_and_duplicate_cells() -> None:
    """c13 is 3'UTR-only, and requested codes must be unique."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig

    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        ParadeActivityConfig(construct_type="utr5", cell_types=["c13"])
    with pytest.raises(ValidationError, match="cell_types must be unique"):
        ParadeActivityConfig(construct_type="utr3", cell_types=["c2", "c2"])


def test_parade_activity_run_dispatches_and_maps_scores(monkeypatch) -> None:
    """run_parade_activity dispatches to the worker and pivots per-cell scores."""
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {"scores": [{"c1": 2.4, "c2": 2.5}, {"c1": 1.3, "c2": 1.4}]}

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_activity(
        ParadeActivityInput(sequences=["A" * 50, "C" * 50]),
        ParadeActivityConfig(construct_type="utr5", cell_types=["c1", "c2"], batch_size=2, device="cuda"),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "activity"
    assert payload["construct_type"] == "utr5"
    assert payload["cell_types"] == ["c1", "c2"]
    assert payload["checkpoint_url"].endswith("model-utr5-deltas-epoch%3D9-step%3D840.ckpt")
    assert payload["checkpoint_md5"] == "a48aeffc516e32f4d8780b855bbcd849"
    assert payload["checkpoint_filename"] == "parade-model-utr5-deltas-epoch9-step840.ckpt"
    assert result.cell_types == ["c1", "c2"]  # derived from results, survives cache reconstruction
    assert dict(result.results[0].scores.items()) == {"c1": 2.4, "c2": 2.5}
    assert result.results[1].scores.c1 == 1.3
    assert_metrics_in_spec(result)


def test_parade_activity_full_cache_hit_reconstructs_valid_output(monkeypatch, fresh_cache, tmp_path) -> None:
    """A full iterable-cache hit rebuilds a usable output (regression for the model_construct path).

    The framework's full-hit path reconstructs the output from only the iterable ``results``
    field, so ``cell_types`` must be derivable and export must not break.
    """
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    dispatch_calls = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        dispatch_calls.append(payload["sequences"])
        return {"scores": [{"c1": 1.0, "c2": 2.0} for _ in payload["sequences"]]}

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    inputs = ParadeActivityInput(sequences=["A" * 50, "C" * 50])
    config = ParadeActivityConfig(construct_type="utr5", cell_types=["c1", "c2"])

    first = run_parade_activity(inputs, config)
    second = run_parade_activity(inputs, config)  # every item cached -> full-hit reconstruction

    assert len(dispatch_calls) == 1  # the second call skipped dispatch (full cache hit)
    assert len(second.results) == 2
    assert second.cell_types == ["c1", "c2"]  # derived field is valid after reconstruction
    assert [r.sequence for r in second.results] == [r.sequence for r in first.results]
    # export(name, export_path=...): name is the file stem, export_path the directory.
    second.export("parade_activity_cached", export_path=tmp_path, file_format="json")
    second.export("parade_activity_cached", export_path=tmp_path, file_format="csv")
    assert (tmp_path / "parade_activity_cached.json").exists()
    assert (tmp_path / "parade_activity_cached.csv").exists()


def test_parade_activity_captured_error_output_serializes(monkeypatch) -> None:
    """A captured-error output (no tool fields) still serializes: the computed fields must not raise.

    Under PROTO_CAPTURE_ERRORS the framework builds the output via ``model_construct`` without
    tool-specific fields; ``results`` defaults to ``[]`` and the computed fields guard on it.
    """
    monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        ParadeActivityOutput,
        run_parade_activity,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(boom))

    result = run_parade_activity(
        ParadeActivityInput(sequences=["A" * 50]),
        ParadeActivityConfig(construct_type="utr5", cell_types=["c1"]),
    )
    assert isinstance(result, ParadeActivityOutput)
    assert result.success is False
    assert result.results == []
    assert result.construct_type == ""  # computed field guards on empty results
    assert result.cell_types == []
    result.model_dump()
    result.model_dump_json()
    repr(result)


def test_parade_activity_accepts_mixed_lengths(monkeypatch) -> None:
    """Mixed-length input is accepted (the standalone batches per length group)."""
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        return {"scores": [{"c1": 1.0, "c2": 2.0} for _ in payload["sequences"]]}

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_activity(
        ParadeActivityInput(sequences=["A" * 50, "C" * 49]),  # different lengths
        ParadeActivityConfig(construct_type="utr5", cell_types=["c1", "c2"]),
    )
    assert [r.sequence_length for r in result.results] == [50, 49]


def test_parade_stability_run_dispatches_and_maps_log_ratios(monkeypatch) -> None:
    """run_parade_stability dispatches to the worker and maps per-sequence log-ratios."""
    import proto_tools.tools.sequence_scoring.parade.parade_stability as parade_stability
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {"log_ratios": [-1.64, -0.86]}

    monkeypatch.setattr(parade_stability.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_stability(
        ParadeStabilityInput(sequences=["A" * 50, "G" * 50]),
        ParadeStabilityConfig(device="cuda"),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "stability"
    assert payload["checkpoint_url"].endswith("stability-epoch%3D24-step%3D725.ckpt")
    assert payload["checkpoint_md5"] == "511c0b4d794f948708ab1e6fa866734b"
    assert result.results[0].log_ratio == -1.64
    assert result.results[1].sequence_length == 50


def test_parade_stability_accepts_mixed_lengths(monkeypatch) -> None:
    """Mixed-length input is accepted (the standalone batches per length group)."""
    import proto_tools.tools.sequence_scoring.parade.parade_stability as parade_stability
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        return {"log_ratios": [float(len(s)) for s in payload["sequences"]]}

    monkeypatch.setattr(parade_stability.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_stability(
        ParadeStabilityInput(sequences=["A" * 50, "C" * 51]),  # different lengths
        ParadeStabilityConfig(),
    )
    assert [r.sequence_length for r in result.results] == [50, 51]
    assert result.results[0].log_ratio == 50.0


def test_parade_gradient_input_validates_logits() -> None:
    """PARADE gradient input requires a B x L x 4 logit tensor."""
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientInput

    inputs = ParadeGradientInput(logits=[[[0.0] * 4] * 50])
    assert len(inputs.logits) == 1
    assert len(inputs.logits[0]) == 50
    with pytest.raises(ValidationError, match="must have 4 columns"):
        ParadeGradientInput(logits=[[[0.0] * 3]])
    with pytest.raises(ValidationError):
        ParadeGradientInput(logits=[[0.0] * 4] * 50)


def test_parade_gradient_rejects_nonfinite_values() -> None:
    """PARADE rejects NaN/Infinity before they can silently contaminate a gradient result."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
    )

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError, match="finite number"):
            ParadeGradientInput(logits=[[[value, 0.0, 0.0, 0.0]]])
        with pytest.raises(ValidationError, match="finite number"):
            ParadeGradientInput(logits=[[[0.0, 0.0, 0.0, 0.0]]], temperature=value)
        with pytest.raises(ValidationError, match="finite number"):
            ParadeGradientConfig(loss_terms=[ParadeGradientLossTerm(sigmoid_center=value)])

    term = ParadeGradientLossTerm()
    with pytest.raises(ValidationError, match="frozen"):
        term.sigmoid_center = math.nan
    with pytest.raises(ValidationError, match="frozen"):
        term.sigmoid_scale = 0.0
    with pytest.raises(ValidationError, match="frozen"):
        term.direction = "invalid"  # type: ignore[assignment]


def test_parade_gradient_output_accepts_empty_metrics() -> None:
    """A valid forward-only/empty metrics payload does not recurse during output validation."""
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientOutput

    output = ParadeGradientOutput(gradient=None, loss=0.0, metrics={}, vocab=["A", "C", "G", "T"])
    assert output.sample_metrics == []
    output.model_dump()


def test_parade_run_boundaries_revalidate_mutated_inputs(monkeypatch) -> None:
    """In-place input mutation cannot bypass sequence or finite-logit validation before dispatch."""
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    import proto_tools.tools.sequence_scoring.parade.parade_gradient as parade_gradient
    import proto_tools.tools.sequence_scoring.parade.parade_stability as parade_stability
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_activity,
        run_parade_gradient,
        run_parade_stability,
    )

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("invalid mutated input reached worker dispatch")

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(unexpected_dispatch))
    monkeypatch.setattr(parade_gradient.ToolInstance, "dispatch", staticmethod(unexpected_dispatch))
    monkeypatch.setattr(parade_stability.ToolInstance, "dispatch", staticmethod(unexpected_dispatch))

    sequences = ParadeActivityInput(sequences=["A" * 50])
    sequences.sequences.append("!")
    with pytest.raises(ValidationError, match="Invalid nucleotide"):
        run_parade_activity(sequences, ParadeActivityConfig())

    gradient = ParadeGradientInput(logits=[[[0.0] * 4]])
    gradient.logits[0][0][0] = math.nan
    with pytest.raises(ValidationError, match="finite"):
        run_parade_gradient(gradient, ParadeGradientConfig())

    invalid_stability_config = ParadeStabilityConfig().model_copy(update={"batch_size": 0})
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        run_parade_stability(ParadeStabilityInput(sequences=["A" * 186]), invalid_stability_config)


def test_parade_gradient_config_rejects_offpanel_cell() -> None:
    """A loss term cell code must belong to the construct-type panel."""
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientConfig, ParadeGradientLossTerm

    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        ParadeGradientConfig(construct_type="utr5", loss_terms=[ParadeGradientLossTerm(cell_type="c13")])
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParadeGradientConfig(loss_terms=[{"cell_type": "c2", "weigth": 9}])  # type: ignore[list-item]


def test_parade_gradient_run_dispatches_loss_terms(monkeypatch) -> None:
    """run_parade_gradient dispatches relaxed logits, construct type, and objectives."""
    import proto_tools.tools.sequence_scoring.parade.parade_gradient as parade_gradient
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {
            "gradient": [[[0.1, 0.0, 0.0, -0.1]] * 50],
            "loss": 0.25,
            "metrics": {"raw_scores": [{"c2": 2.5}], "loss_terms": [[]], "losses": [0.25]},
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(parade_gradient.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_gradient(
        ParadeGradientInput(logits=[[[0.0] * 4] * 50], temperature=0.7),
        ParadeGradientConfig(
            construct_type="utr5",
            loss_terms=[
                ParadeGradientLossTerm(cell_type="c2", direction="max", weight=2.0),
                ParadeGradientLossTerm(cell_type="c6", direction="min", weight=0.5),
            ],
            soft=1.0,
            hard=0.5,
            device="cuda",
        ),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "gradient"
    assert payload["construct_type"] == "utr5"
    assert payload["temperature"] == 0.7
    assert payload["soft"] == 1.0
    assert payload["hard"] == 0.5
    assert payload["checkpoint_url"].endswith("model-utr5-deltas-epoch%3D9-step%3D840.ckpt")
    assert payload["loss_terms"][0]["cell_type"] == "c2"
    assert payload["loss_terms"][1]["direction"] == "min"
    assert result.gradient is not None
    assert len(result.gradient[0]) == 50
    assert len(result.sample_metrics) == 1
    assert result.sample_metrics[0]["loss"] == 0.25
    assert result.sample_metrics[0]["c2"] == 2.5
    assert result.vocab == ["A", "C", "G", "T"]
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_activity_real_gpu() -> None:
    """Real GPU smoke test for PARADE UTR activity through the tool worker."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    result = run_parade_activity(
        ParadeActivityInput(sequences=["ACGT" * 12 + "AC", "TGCA" * 12 + "GT"]),
        ParadeActivityConfig(construct_type="utr5", batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert result.cell_types == ["c1", "c2", "c4", "c6", "c17"]
    for sequence_result in result.results:
        assert set(sequence_result.scores) == {"c1", "c2", "c4", "c6", "c17"}
        assert all(math.isfinite(score) for score in sequence_result.scores.values())
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_activity_real_gpu_utr3() -> None:
    """Real GPU smoke test for the distinct 3'UTR checkpoint (6-cell panel incl. c13, ~240 nt)."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    result = run_parade_activity(
        ParadeActivityInput(sequences=["ACGT" * 60, "TGCA" * 60]),  # 240 nt: the 3'UTR training length
        ParadeActivityConfig(construct_type="utr3", batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert result.cell_types == ["c1", "c2", "c4", "c6", "c17", "c13"]
    for sequence_result in result.results:
        assert set(sequence_result.scores) == {"c1", "c2", "c4", "c6", "c17", "c13"}
        assert all(math.isfinite(score) for score in sequence_result.scores.values())
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_gradient_real_gpu_utr3() -> None:
    """Real GPU smoke test for the 3'UTR differentiable path (c13 objective, ~240 nt)."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )

    result = run_parade_gradient(
        ParadeGradientInput(logits=[[[0.0] * 4] * 240], temperature=1.0),  # 240 nt
        ParadeGradientConfig(
            construct_type="utr3",
            loss_terms=[
                ParadeGradientLossTerm(cell_type="c2", direction="max"),
                ParadeGradientLossTerm(cell_type="c13", direction="min"),
            ],
            device="cuda",
        ),
    )

    assert result.gradient is not None
    assert len(result.gradient[0]) == 240
    assert math.isfinite(result.loss)
    assert math.isfinite(result.sample_metrics[0]["c13"])


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_real_gpu() -> None:
    """Real GPU smoke test for PARADE 3' UTR stability through the tool worker."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    result = run_parade_stability(
        ParadeStabilityInput(sequences=["ACGT" * 46 + "AC", "TGCA" * 46 + "GT"]),  # 186 nt: the training length
        ParadeStabilityConfig(batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert all(math.isfinite(sequence_result.log_ratio) for sequence_result in result.results)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_gradient_real_gpu() -> None:
    """Real GPU smoke test for differentiable PARADE UTR activity."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )

    result = run_parade_gradient(
        ParadeGradientInput(logits=[[[0.0] * 4] * 50, [[0.1, 0.2, 0.3, 0.4]] * 50], temperature=1.0),
        ParadeGradientConfig(
            construct_type="utr5",
            loss_terms=[
                ParadeGradientLossTerm(cell_type="c2", direction="max"),
                ParadeGradientLossTerm(cell_type="c6", direction="min"),
            ],
            device="cuda",
        ),
    )

    assert result.gradient is not None
    assert len(result.gradient) == 2
    assert all(len(matrix) == 50 for matrix in result.gradient)
    assert any(value != 0.0 for matrix in result.gradient for row in matrix for value in row)
    assert math.isfinite(result.loss)
    assert len(result.sample_metrics) == 2


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_mixed_length_grouping_preserves_order() -> None:
    """Interleaved-length input is grouped per length and reassembled in original order.

    Exercises the real standalone grouping path (which the mocked unit tests don't): scores
    an interleaved 50/30/50/30 batch and checks each result matches the same sequence scored
    within its own length group, position for position.
    """
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    config = ParadeActivityConfig(construct_type="utr5", cell_types=["c1", "c2"], device="cuda")
    interleaved = ["A" * 50, "C" * 30, "GC" * 25, "AT" * 15]  # lengths 50, 30, 50, 30
    mixed = run_parade_activity(ParadeActivityInput(sequences=interleaved), config)
    assert [r.sequence_length for r in mixed.results] == [50, 30, 50, 30]

    ref_50 = run_parade_activity(ParadeActivityInput(sequences=["A" * 50, "GC" * 25]), config)
    ref_30 = run_parade_activity(ParadeActivityInput(sequences=["C" * 30, "AT" * 15]), config)
    comparisons = (
        (mixed.results[0], ref_50.results[0]),
        (mixed.results[2], ref_50.results[1]),
        (mixed.results[1], ref_30.results[0]),
        (mixed.results[3], ref_30.results[1]),
    )
    for actual, expected in comparisons:
        assert dict(actual.scores.items()) == pytest.approx(dict(expected.scores.items()), rel=1e-6, abs=1e-6)


@pytest.mark.benchmark("parade-activity")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_activity_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-activity on 25000 length-50 UTRs across the 5'UTR panel (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    inputs = ParadeActivityInput(sequences=random_dna_sequences(n=25000, length=50, seed=0))
    config = ParadeActivityConfig(construct_type="utr5", batch_size=64)

    result = benchmark_twice(request, "parade", lambda: run_parade_activity(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)
    assert result.tool_id == "parade-activity"
    assert len(result.results) == 25000


@pytest.mark.benchmark("parade-stability")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_stability_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-stability on 25000 length-186 3'UTRs (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    inputs = ParadeStabilityInput(sequences=random_dna_sequences(n=25000, length=186, seed=0))
    config = ParadeStabilityConfig(batch_size=64)

    result = benchmark_twice(request, "parade", lambda: run_parade_stability(inputs, config))
    validate_output(result)
    assert result.tool_id == "parade-stability"
    assert len(result.results) == 25000


@pytest.mark.benchmark("parade-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-gradient: 5 loops of a 256x50x4 batched UTR logit backward pass (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )
    from proto_tools.utils import DNA_NUCLEOTIDES

    base_index = {nucleotide: column for column, nucleotide in enumerate(DNA_NUCLEOTIDES)}
    sequences = random_dna_sequences(n=256, length=50, seed=0)
    logits = [
        [[1.0 if column == base_index[base] else 0.0 for column in range(len(DNA_NUCLEOTIDES))] for base in sequence]
        for sequence in sequences
    ]
    inputs = ParadeGradientInput(logits=logits, temperature=1.0)
    config = ParadeGradientConfig(
        construct_type="utr5",
        loss_terms=[
            ParadeGradientLossTerm(cell_type="c2", direction="max"),
            ParadeGradientLossTerm(cell_type="c6", direction="min"),
        ],
    )

    def run_batch() -> object:
        last = None
        for _ in range(5):
            last = run_parade_gradient(inputs, config)
        return last

    result = benchmark_twice(request, "parade", run_batch)
    validate_output(result)
    assert_metrics_in_spec(result)
    assert result.tool_id == "parade-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == 256
