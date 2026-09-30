"""Shared numerical checks for the two mixed protein/DNA model backends."""

import importlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from standalone_helpers.mixed_sequence import tokenize_mixed_sequence

from proto_tools.tools.masked_models.mixed_data_models import MIXED_VOCAB, one_hot_mixed_logits
from proto_tools.utils import ToolInstance
from proto_tools.utils.persistent_worker import _build_subprocess_env
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

# Long-form markers, since upstream_reference passes this string verbatim to the upstream tokenizer.
LOCUS = "<+>MA<+>ac<->X"


def call_model(toolkit, operation, checkpoint, inputs, **settings):
    """Call the public tool using its toolkit-specific input/config classes."""
    api = importlib.import_module("proto_tools.tools")
    prefix = "GLM2" if toolkit == "glm2" else "Minerva"
    kind = {
        "embeddings": "Embeddings",
        "score": "Scoring",
        "sample": "Sample",
        "gradient": "Gradient",
        "interactions": "Interactions",
    }[operation]
    input_value = getattr(api, f"{prefix}{kind}Input")(**inputs)
    config = getattr(api, f"{prefix}{kind}Config")(model_checkpoint=checkpoint, **settings)
    result = getattr(api, f"run_{toolkit}_{operation}")(input_value, config)
    assert result.success
    key = "embedding" if operation == "embeddings" else operation
    assert result.tool_id == f"{toolkit}-{key}"
    assert_metrics_in_spec(result)
    return result


def check_embeddings_score_gradient(toolkit, checkpoint):
    """Verify token alignment, exact-length batching, and inference-to-backward persistence."""
    sequences = [LOCUS, "<->W<+>g", "<+>AC<+>gt<->X"]
    batch = call_model(toolkit, "embeddings", checkpoint, {"sequences": sequences}, return_logits=True, batch_size=3)
    singles = call_model(toolkit, "embeddings", checkpoint, {"sequences": sequences}, return_logits=True, batch_size=1)
    for sequence, result, single in zip(sequences, batch.results, singles.results, strict=True):
        tokens = tokenize_mixed_sequence(sequence)
        assert result.tokens == tokens
        assert result.vocab == MIXED_VOCAB
        assert result.attention_mask == [1] * len(tokens)
        assert np.asarray(result.logits).shape == (len(tokens), 24)
        assert np.isfinite(result.mean_embedding).all()
        np.testing.assert_allclose(result.mean_embedding, single.mean_embedding, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(result.logits, single.logits, rtol=1e-4, atol=1e-4)
    score = call_model(toolkit, "score", checkpoint, {"sequences": LOCUS}, return_logits=True, batch_size=2).scores[0]
    assert score.tokens == tokenize_mixed_sequence(LOCUS)
    assert score.scored_positions == [2, 3, 5, 6]
    assert score.log_likelihood == pytest.approx(4 * score.avg_log_likelihood)
    assert score.perplexity == pytest.approx(math.exp(-score.avg_log_likelihood))
    assert np.asarray(score.logits).shape == (8, 24)
    np.testing.assert_array_equal(np.asarray(score.logits)[[0, 3, 6, 7]], 0.0)
    inputs = {"sequence": LOCUS, "logits": one_hot_mixed_logits(LOCUS, sharpness=1.0), "temperature": None}
    backward = call_model(toolkit, "gradient", checkpoint, inputs, batch_size=2)
    forward = call_model(toolkit, "gradient", checkpoint, inputs, batch_size=2, compute_gradient=False)
    gradient = np.asarray(backward.gradient)
    assert gradient.shape == (8, 24)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[[0, 3, 6, 7]], 0.0)
    np.testing.assert_array_equal(gradient[[1, 2], 20:], 0.0)
    np.testing.assert_array_equal(gradient[[4, 5], :20], 0.0)
    assert np.any(gradient[[1, 2, 4, 5]] != 0.0)
    assert backward.loss == pytest.approx(-score.avg_log_likelihood, rel=2e-5, abs=2e-5)
    assert forward.loss == pytest.approx(backward.loss, rel=2e-5, abs=2e-5)
    assert forward.gradient is None
    validate_output(backward)


def check_sampling(toolkit, checkpoint, sampling_method):
    """Keep markers and alphabets fixed while advancing RNG across duplicate loci."""
    # Long-form markers in the input come back as one-character markers.
    sequence = "<+>____<+>____<->X"
    mapping = {**dict.fromkeys(range(2, 6), "protein"), **dict.fromkeys(range(7, 11), "dna")}
    inputs = {"sequences": [sequence] * 3, "mask_modalities": [mapping] * 3}
    settings = {
        "seed": 7,
        "temperature": 2.0,
        "sampling_method": sampling_method,
        "num_steps": 3,
        "return_logits": True,
    }
    output = call_model(toolkit, "sample", checkpoint, inputs, **settings)
    repeated = call_model(toolkit, "sample", checkpoint, inputs, **settings)
    assert output.sequences == repeated.sequences
    assert len(set(output.sequences)) > 1
    for result in output.results:
        assert result.tokens == tokenize_mixed_sequence(result.sequence)
        assert [result.tokens[i] for i in [0, 5, 10, 11]] == ["+", "+", "-", "X"]
        assert result.sequence == "".join(result.tokens)
        assert all(token in MIXED_VOCAB[:20] for token in result.tokens[1:5])
        assert all(token in MIXED_VOCAB[20:] for token in result.tokens[6:10])
        assert np.asarray(result.logits).shape == (12, 24)
        assert result.vocab == MIXED_VOCAB


def check_gradient_finite_difference(toolkit, checkpoint):
    """Compare a real model's relaxed masked-PLL derivative with a central difference."""
    sequence = "<+>M<+>a"
    logits = np.asarray(one_hot_mixed_logits(sequence, sharpness=2.0))
    inputs = {"sequence": sequence, "logits": logits.tolist(), "temperature": 0.9}
    result = call_model(toolkit, "gradient", checkpoint, inputs, batch_size=2)
    epsilon = 0.02
    losses = []
    for delta in [epsilon, -epsilon]:
        perturbed = logits.copy()
        perturbed[3, 21] += delta
        losses.append(
            call_model(
                toolkit,
                "gradient",
                checkpoint,
                {**inputs, "logits": perturbed.tolist()},
                batch_size=2,
                compute_gradient=False,
            ).loss
        )
    derivative = (losses[0] - losses[1]) / (2 * epsilon)
    assert result.gradient[3][21] == pytest.approx(derivative, rel=0.03, abs=3e-4)


_UPSTREAM_REFERENCE = """
import importlib.util,json,sys,torch
path,checkpoint,sequence,depth=sys.argv[1:]
spec=importlib.util.spec_from_file_location("mixed_reference_adapter",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
adapter=module._MODEL
adapter.load(checkpoint,"cuda")
tokenizer=adapter.tokenizer
encoded=tokenizer(sequence,add_special_tokens=False,return_tensors="pt")
ids=encoded["input_ids"].to("cuda")
mask=torch.ones_like(ids)
tokens=tokenizer.convert_ids_to_tokens(ids[0].tolist())
vocab=list("ACDEFGHIKLMNPQRSTVWYacgt")
biological_ids=[tokenizer.convert_tokens_to_ids(t) for t in vocab]
positions=[i for i,t in enumerate(tokens) if t in vocab]
rows=[];logprobs=[]
with torch.no_grad():
 for position in positions:
  masked=ids.clone();masked[0,position]=tokenizer.mask_token_id
  pred=adapter.model(input_ids=masked,attention_mask=mask).logits[0,position].float()
  rows.append(pred[biological_ids].cpu().tolist())
  logprobs.append(torch.log_softmax(pred,dim=-1)[ids[0,position]].item())
 result={"positions":[i+1 for i in positions],"logits":rows,"avg_log_likelihood":sum(logprobs)/len(logprobs)}
 if depth!="0":
  raw=adapter.model(input_ids=ids,attention_mask=mask,output_interactions=True,interaction_layers=int(depth))
  result["maps"]={head:value[0].float().cpu().tolist() for head,value in raw.interactions.items()}
print("MIXED_REFERENCE="+json.dumps(result))
"""


def upstream_reference(toolkit, checkpoint, sequence, interaction_layers=0):
    """Run native upstream forwards in the managed environment, outside the tool algorithms."""
    instance = ToolInstance.get(toolkit)
    instance.ensure_ready()
    inference = (
        Path(__file__).parents[2] / "proto_tools" / "tools" / "masked_models" / toolkit / "standalone" / "inference.py"
    )
    process = subprocess.run(
        [
            str(instance.env_path / "bin" / "python"),
            "-c",
            _UPSTREAM_REFERENCE,
            str(inference),
            checkpoint,
            sequence,
            str(interaction_layers),
        ],
        env=_build_subprocess_env(device="cuda", tool_env_path=instance.env_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    reference = next(
        line.removeprefix("MIXED_REFERENCE=")
        for line in process.stdout.splitlines()
        if line.startswith("MIXED_REFERENCE=")
    )
    return json.loads(reference)


def check_upstream_pll(toolkit, checkpoint):
    """Validate canonical target selection and the full-vocabulary PLL normalization."""
    reference = upstream_reference(toolkit, checkpoint, LOCUS)
    result = call_model(toolkit, "score", checkpoint, {"sequences": LOCUS}, batch_size=2, return_logits=True).scores[0]
    assert result.scored_positions == reference["positions"]
    assert result.avg_log_likelihood == pytest.approx(reference["avg_log_likelihood"], rel=2e-5, abs=2e-5)
    np.testing.assert_allclose(
        np.asarray(result.logits)[np.asarray(result.scored_positions) - 1], reference["logits"], rtol=1e-4, atol=1e-4
    )


def benchmark_operation(request, toolkit, checkpoint, operation):
    """Benchmark each public operation with a representative, bounded mixed locus."""
    sequence = "<+>" + "MKTLACDE" * 4 + "<+>" + "acgt" * 8
    inputs = {"sequences": [sequence] * (4 if operation == "sample" else 1)}
    if operation == "embeddings":
        inputs = {"sequences": [sequence[:-1] + base for base in "acgt"]}
    settings = {"batch_size": 4}
    if operation == "gradient":
        inputs = {"sequence": sequence, "logits": one_hot_mixed_logits(sequence, sharpness=2.0)}
    if operation == "sample":
        settings["seed"] = 0
    result = benchmark_twice(request, toolkit, lambda: call_model(toolkit, operation, checkpoint, inputs, **settings))
    validate_output(result)
    assert result.success is not False
    return result
