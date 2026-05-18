# ruff: noqa
# mypy: ignore-errors
"""
Standalone Malinois MPRA scoring module.

Derived from the CODA/BODA2 package (https://github.com/sjgosai/boda2) so that
scoring can run in a standalone proto-tools environment with only torch + numpy.
Per the CODA/BODA2 README, the Malinois model, model weights, and code defining
the model architecture are covered under the MIT license.

Only the inference-time components are included: model architecture, one-hot
encoding, flanking logic, and checkpoint loading.  Training-only pieces
(LightningModule hooks, data modules, CLI argument parsers) are omitted.
"""

import math
import os
import sys
import shutil
import tarfile
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from standalone_helpers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_NT = ["A", "C", "G", "T"]

MPRA_UPSTREAM = (
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGG"
    "TTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAG"
    "GCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGA"
    "TAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGC"
    "TGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
)

MPRA_DOWNSTREAM = (
    "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCA"
    "AGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCC"
    "AGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTT"
    "CACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAG"
    "CGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"
)

# ---------------------------------------------------------------------------
# DNA ↔ tensor helpers
# ---------------------------------------------------------------------------


def dna2tensor(sequence_str, vocab_list=STANDARD_NT):
    """One-hot encode a DNA string → tensor of shape (4, L)."""
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for idx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), idx] = 1
    return torch.Tensor(seq_tensor)


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------


class Conv1dNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=True,
        weight_norm=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        try:
            return self.bn_layer(self.conv(x))
        except AttributeError:
            return self.conv(x)


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, weight_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features)

    def forward(self, x):
        try:
            return self.bn_layer(self.linear(x))
        except AttributeError:
            return self.linear(x)


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)
        return reorg


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self, in_features, hidden_group_size, out_group_size, n_branches=1, n_layers=1, activation="ReLU", dropout_p=0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f"branched_layer_{i + 1}", GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f"branched_layer_{i + 1}", GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)
        i = -1
        for i in range(self.n_layers - 1):
            hook = getattr(self, f"branched_layer_{i + 1}")(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f"branched_layer_{i + 2}")(hook)
        return hook


# ---------------------------------------------------------------------------
# Loss function (needed by model constructor, unused at inference)
# ---------------------------------------------------------------------------


class L1KLmixed(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.MSE = nn.L1Loss(reduction=reduction.replace("batch", ""))
        self.KL = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, preds, targets):
        preds_log_prob = preds - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        MSE_loss = self.MSE(preds, targets)
        KL_loss = self.KL(preds_log_prob, target_log_prob)
        combined_loss = MSE_loss.mul(self.alpha) + KL_loss.mul(self.beta)
        return combined_loss.div(self.alpha + self.beta)


_LOSS_REGISTRY = {
    "L1KLmixed": L1KLmixed,
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


class BassetBranched(nn.Module):
    def __init__(
        self,
        input_len=600,
        conv1_channels=300,
        conv1_kernel_size=19,
        conv2_channels=200,
        conv2_kernel_size=11,
        conv3_channels=200,
        conv3_kernel_size=7,
        n_linear_layers=2,
        linear_channels=1000,
        linear_activation="ReLU",
        linear_dropout_p=0.3,
        n_branched_layers=1,
        branched_channels=250,
        branched_activation="ReLU6",
        branched_dropout_p=0.0,
        n_outputs=280,
        use_batch_norm=True,
        use_weight_norm=False,
        loss_criterion="L1KLmixed",
        loss_args={},
    ):
        super().__init__()

        self.input_len = input_len

        self.conv1_channels = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = _get_padding(conv1_kernel_size)

        self.conv2_channels = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = _get_padding(conv2_kernel_size)

        self.conv3_channels = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = _get_padding(conv3_kernel_size)

        self.n_linear_layers = n_linear_layers
        self.linear_channels = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p = linear_dropout_p

        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p = branched_dropout_p

        self.n_outputs = n_outputs

        self.loss_criterion = loss_criterion
        self.loss_args = loss_args

        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.pad1 = nn.ConstantPad1d(self.conv1_pad, 0.0)
        self.conv1 = Conv1dNorm(
            4,
            self.conv1_channels,
            self.conv1_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad2 = nn.ConstantPad1d(self.conv2_pad, 0.0)
        self.conv2 = Conv1dNorm(
            self.conv1_channels,
            self.conv2_channels,
            self.conv2_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad3 = nn.ConstantPad1d(self.conv3_pad, 0.0)
        self.conv3 = Conv1dNorm(
            self.conv2_channels,
            self.conv3_channels,
            self.conv3_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )

        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)
        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        next_in_channels = self.conv3_channels * self._get_flatten_factor(self.input_len)

        for i in range(self.n_linear_layers):
            setattr(
                self,
                f"linear{i + 1}",
                LinearNorm(
                    next_in_channels,
                    self.linear_channels,
                    bias=True,
                    batch_norm=self.use_batch_norm,
                    weight_norm=self.use_weight_norm,
                ),
            )
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(
            next_in_channels,
            self.branched_channels,
            self.branched_channels,
            self.n_outputs,
            self.n_branched_layers,
            self.branched_activation,
            self.branched_dropout_p,
        )

        self.output = GroupedLinear(self.branched_channels, 1, self.n_outputs)
        self.nonlin = getattr(nn, self.linear_activation)()
        self.dropout = nn.Dropout(p=self.linear_dropout_p)

        loss_cls = _LOSS_REGISTRY.get(self.loss_criterion)
        if loss_cls is not None:
            self.criterion = loss_cls(**self.loss_args)

    @staticmethod
    def _get_flatten_factor(input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0
        return (hook + 2) // 4

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        return hook

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(self.nonlin(getattr(self, f"linear{i + 1}")(hook)))
        hook = self.branched(hook)
        return hook

    def classify(self, x):
        return self.output(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return self.classify(decoded)


# ---------------------------------------------------------------------------
# Model registry (for checkpoint loading)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "BassetBranched": BassetBranched,
}

# ---------------------------------------------------------------------------
# FlankBuilder
# ---------------------------------------------------------------------------


class FlankBuilder(nn.Module):
    """Pads input one-hot sequences with fixed flanking regions."""

    def __init__(self, left_flank=None, right_flank=None, batch_dim=0, cat_axis=-1):
        super().__init__()
        self.register_buffer("left_flank", left_flank.detach().clone())
        self.register_buffer("right_flank", right_flank.detach().clone())
        self.batch_dim = batch_dim
        self.cat_axis = cat_axis

    def forward(self, my_sample):
        *batch_dims, channels, length = my_sample.shape
        pieces = []
        if self.left_flank is not None:
            pieces.append(self.left_flank.expand(*batch_dims, -1, -1))
        pieces.append(my_sample)
        if self.right_flank is not None:
            pieces.append(self.right_flank.expand(*batch_dims, -1, -1))
        return torch.cat(pieces, axis=self.cat_axis)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _unpack_artifact(artifact_path, download_path="./"):
    if "gs" in artifact_path:
        raise RuntimeError("GCS paths not supported in standalone mode")
    assert os.path.isfile(artifact_path), f"Could not find file at {artifact_path}"
    assert tarfile.is_tarfile(artifact_path), f"Expected a tarfile at {artifact_path}"
    shutil.unpack_archive(artifact_path, download_path)
    print(f"archive unpacked in {download_path}", file=sys.stderr)


def _model_fn(model_dir):
    checkpoint = torch.load(os.path.join(model_dir, "torch_checkpoint.pt"), weights_only=False)
    model_name = checkpoint["model_module"]
    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls(**vars(checkpoint["model_hparams"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint['timestamp']} in eval mode")
    model.eval()
    return model


def load_model(artifact_path):
    """Load a Malinois model from a tar.gz artifact (same as boda.common.utils.load_model)."""
    USE_CUDA = torch.cuda.device_count() >= 1
    if os.path.isdir("./artifacts"):
        shutil.rmtree("./artifacts")
    _unpack_artifact(artifact_path)
    my_model = _model_fn("./artifacts")
    my_model.eval()
    if USE_CUDA:
        my_model.cuda()
    return my_model


# ---------------------------------------------------------------------------
# High-level scoring API
# ---------------------------------------------------------------------------


def load_malinois(artifact_path, malinois_dir=None, seq_len=200):
    """
    Load model and build the FlankBuilder for a given insert length.

    Returns (model, flank_builder) both on GPU if available.
    """
    model = load_model(artifact_path)

    if malinois_dir is None:
        malinois_dir = os.path.dirname(artifact_path)

    input_len = torch.load(os.path.join(malinois_dir, "artifacts/torch_checkpoint.pt"), weights_only=False)[
        "model_hparams"
    ].input_len

    left_pad_len = (input_len - seq_len) // 2
    right_pad_len = (input_len - seq_len) - left_pad_len

    left_flank = dna2tensor(MPRA_UPSTREAM[-left_pad_len:]).unsqueeze(0)
    right_flank = dna2tensor(MPRA_DOWNSTREAM[:right_pad_len]).unsqueeze(0)

    flank_builder = FlankBuilder(left_flank=left_flank, right_flank=right_flank)

    if torch.cuda.device_count() >= 1:
        flank_builder.cuda()

    return model, flank_builder


def score_sequences(sequences: List[str], model, flank_builder, batch_size=512) -> np.ndarray:
    """
    Score a list of DNA strings.

    Returns an array of shape (N, n_outputs) with the averaged
    forward + reverse-complement predictions.
    """
    seq_tensor = torch.stack([dna2tensor(s) for s in sequences], dim=0)
    dataset = torch.utils.data.TensorDataset(seq_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    results = []
    with torch.no_grad():
        for (batch,) in loader:
            prepped = flank_builder(batch.cuda() if next(model.parameters()).is_cuda else batch)
            preds = model(prepped) + model(prepped.flip(dims=[1, 2]))
            preds = preds.div(2.0)
            results.append(preds.detach().cpu())
    return torch.cat(results, dim=0).numpy()


if __name__ == "__main__":
    raise SystemExit("malinois_scoring.py is a library module; run inference.py through proto-tools instead.")
