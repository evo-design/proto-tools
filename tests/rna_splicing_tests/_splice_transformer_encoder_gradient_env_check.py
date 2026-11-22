"""In-env check that SpliceTransformer input gradients reach the conv encoders.

Upstream runs ``SpTransformer.encoder`` under ``torch.no_grad()`` to freeze it during training.
The gradient tool differentiates ``SpTransformer.forward`` directly, so that block silently cut
the encoders out of every input gradient while the forward score still used them. This builds
the model with random weights on CPU (no checkpoint, no GPU) and backpropagates one splice
probability: every encoder parameter the forward uses must receive a gradient.

Run by ``test_splice_transformer_gradient_reaches_the_encoders`` in
``test_splice_transformer_gradient.py``. Plain asserts; a non-zero exit surfaces as
``RuntimeError``. The leading ``_`` keeps pytest from collecting this file in the base process.
"""

import torch
from model import SpTransformer  # type: ignore[import-not-found]

torch.manual_seed(0)
model = SpTransformer(context_len=4000).eval()
x = torch.softmax(torch.randn(1, 4, 2 * 4000 + 1000), dim=1).requires_grad_(True)

out = model(x)
torch.softmax(out[:, :3, :], dim=1)[0, 2].mean().backward()

assert x.grad is not None and torch.isfinite(x.grad).all()
# The encoders run with feature=True, which skips their own output heads.
used = {name: p for name, p in model.encoder.named_parameters() if "_output." not in name}
missing = [name for name, p in used.items() if p.grad is None]
assert not missing, f"no gradient reached {len(missing)} of {len(used)} encoder parameters, e.g. {missing[:3]}"
assert any(p.grad.abs().sum() > 0 for p in used.values()), "encoder gradients are all zero"
print("SPLICE_TRANSFORMER_ENCODER_GRADIENT_OK")
