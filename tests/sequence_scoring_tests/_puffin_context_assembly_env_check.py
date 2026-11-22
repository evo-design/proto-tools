"""In-env check that the Puffin gradient's fixed context is the input upstream predict() builds.

``compute_gradient`` cannot call ``Puffin.predict`` (it takes a string and detaches), so it
encodes ``context_left``/``context_right`` itself and places them around the relaxed candidate.
If that encoding drifted from upstream's (a different base order, ``N`` read as zeros or as
``A``) the gradient would stay finite and self-consistent while scoring an input the forward
tools never see. This hands upstream's own ``predict`` a stand-in module that records its input,
so no weights are loaded and no GPU is needed; the cloned Puffin repo supplies ``predict``.

Run by ``test_puffin_gradient_context_matches_upstream_encoding`` in ``test_puffin_gradient.py``.
Plain asserts; a non-zero exit surfaces as ``RuntimeError``. The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

import random

import inference  # type: ignore[import-not-found]
import torch

inference.PuffinModel()._ensure_repo()
from puffin import Puffin  # type: ignore[import-not-found]  # noqa: E402


class _Recorder:
    """Stands in for the loaded module: records the tensor predict() feeds it."""

    def __init__(self) -> None:
        """No tracks are named, so predict() only builds and feeds its input."""
        self.use_cuda = False
        self.targeti_map: dict[str, int] = {}
        self.targeti_rev_map: dict[str, int] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return torch.zeros(1, 10, x.shape[2])


rng = random.Random(0)
context_left = "".join(rng.choice("ACGTN") for _ in range(inference.PUFFIN_PADDING))
context_right = "N" * 40 + "".join(rng.choice("ACGT") for _ in range(inference.PUFFIN_PADDING - 40))
design = "".join(rng.choice("ACGT") for _ in range(60))

recorder = _Recorder()
Puffin.predict(recorder, context_left + design + context_right)

one_hot = torch.tensor([[1.0 if base == b else 0.0 for b in "ACGT"] for base in design], requires_grad=True)
left = inference._encode_fixed_dna(context_left, "cpu")
right = inference._encode_fixed_dna(context_right, "cpu")
built = inference._model_input(one_hot, left, right)

assert built.shape == recorder.x.shape, f"{tuple(built.shape)} vs upstream {tuple(recorder.x.shape)}"
assert torch.equal(built.detach(), recorder.x), "fixed context or layout differs from upstream predict()"
assert torch.equal(right[:40], torch.full((40, 4), 0.25)), "N must read as 0.25 per base"

# Only the candidate carries gradient, and each candidate row maps back to its own column.
weights = torch.randn_like(built)
(built * weights).sum().backward()
assert one_hot.grad is not None and one_hot.grad.shape == (len(design), 4)
start = len(context_left)
assert torch.equal(one_hot.grad, weights[0, :, start : start + len(design)].t())

# Without context the candidate is the whole input, laid out as before.
assert torch.equal(inference._model_input(one_hot, None, None), one_hot.t().unsqueeze(0))

print("PUFFIN_CONTEXT_ASSEMBLY_OK")
