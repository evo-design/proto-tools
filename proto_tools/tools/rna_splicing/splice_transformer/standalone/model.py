"""Copied from https://github.com/ShenLab-Genomics/SpliceTransformer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from sinkhorn_transformer import SinkhornTransformer


class ResBlock(nn.Module):  # type: ignore[misc]
    """Dilated convolutional residual block with batch normalization."""

    def __init__(self, L: Any, W: Any, AR: Any, pad: Any = True) -> None:
        """Initialize ResBlock."""
        super().__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1
        # padding calculation:
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s)) if pad else 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x: Any) -> Any:
        """Run the forward pass."""
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return out + x


class SpEncoder(nn.Module):  # type: ignore[misc]
    """SpliceAI-style convolutional encoder for splice site prediction."""

    def __init__(self, L: Any, tissue_cnt: Any = 11, context_len: Any = 4000) -> None:
        """Initialize SpEncoder."""
        super().__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 3, 1)
        self.tissue_output = nn.Conv1d(L, tissue_cnt, 1)
        self.context_len = context_len

    def forward(self, x: Any, use_usage_head: Any = True) -> Any:  # noqa: ARG002 — kept for API compatibility with upstream
        """Run the forward pass."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out_splice = self.splice_output(skip)
        out_usage = self.tissue_output(skip)
        return torch.concat([out_splice, out_usage], dim=1)

    def forward_feature(self, x: Any) -> Any:
        """Run the forward pass returning intermediate features."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        return skip

    def random_mask(self, seq: Any, label: Any) -> Any:
        """Randomly mask flanking context regions for data augmentation."""
        if (torch.rand(1)) < 0.05:
            seq[:, :, : self.context_len] *= 0
            seq[:, :, -self.context_len :] *= 0
        return seq, label


class SpEncoder_4tis(nn.Module):  # type: ignore[misc]
    """Four-tissue variant of SpEncoder with separate output heads per tissue pair."""

    def __init__(self, L: Any, tissue_cnt: Any = 11, context_len: Any = 4000):  # noqa: ARG002 — kept for API compatibility with upstream
        """Initialize SpEncoder_4tis."""
        super().__init__()
        #
        # convolution window size in residual units
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        # atrous rate in residual units
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        #
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)
        self.context_len = context_len

    def forward(self, x: Any, use_usage_head: Any = True) -> Any:  # noqa: ARG002 — kept for API compatibility with upstream
        """Run the forward pass."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out = [
            F.softmax(self.conv_last1(skip), dim=1),
            torch.sigmoid(self.conv_last2(skip)),
            F.softmax(self.conv_last3(skip), dim=1),
            torch.sigmoid(self.conv_last4(skip)),
            F.softmax(self.conv_last5(skip), dim=1),
            torch.sigmoid(self.conv_last6(skip)),
            F.softmax(self.conv_last7(skip), dim=1),
            torch.sigmoid(self.conv_last8(skip)),
        ]
        return torch.cat(out, 1)

    def forward_feature(self, x: Any) -> Any:
        """Run the forward pass returning intermediate features."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        return skip


class AttnBlock(nn.Module):  # type: ignore[misc]
    """Sinkhorn transformer attention block with axial positional embedding."""

    def __init__(
        self, dim: Any = 256, depth: Any = 6, causal: Any = False, max_seq_len: Any = 8192, reversible: Any = False
    ) -> None:
        """Initialize AttnBlock."""
        super().__init__()
        bucket_size = 64
        axial_position_shape = ((max_seq_len // bucket_size), bucket_size)
        self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
        self.attn = SinkhornTransformer(
            dim,
            depth,
            heads=8,
            n_local_attn_heads=2,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.1,
            layer_dropout=0.1,
            ff_dropout=0.1,
            ff_chunks=10,
            causal=causal,
            reversible=reversible,
            non_permutative=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Any) -> Any:
        """Run the forward pass."""
        x = torch.transpose(x, 1, 2).contiguous()
        x = x + self.pos_emb(x)
        x = self.attn(x)
        x = self.norm(x)
        return torch.transpose(x, 1, 2).contiguous()


class SpEncoder_L(nn.Module):  # type: ignore[misc]
    """Large-channel SpEncoder variant with configurable tissue count."""

    def __init__(self, L: Any, tissue_cnt: Any = 15, context_len: Any = 4000) -> None:
        """Initialize SpEncoder_L."""
        super().__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 3, 1)
        self.tissue_output = nn.Conv1d(L, tissue_cnt, 1)
        self.context_len = context_len

    def forward(self, x: Any, feature: Any = False) -> Any:
        """Run the forward pass."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        if feature:
            return skip
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out_splice = self.splice_output(skip)
        out_usage = self.tissue_output(skip)
        return torch.concat([out_splice, out_usage], dim=1)


class SpEncoder2_L(nn.Module):  # type: ignore[misc]
    """Stage-2 convolutional encoder for multi-tissue splice site scoring."""

    def __init__(self, L: Any, tissue_cnt: Any = 4, context_len: Any = 4000):  # noqa: ARG002 — kept for API compatibility with upstream
        """Initialize SpEncoder2_L."""
        super().__init__()
        #
        # convolution window size in residual units
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21])
        # atrous rate in residual units
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20])
        #
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.context_len = context_len

    def forward(self, x: Any, feature: Any = False) -> Any:
        """Run the forward pass."""
        x = x[:, 0:4, :]
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.W)):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        if feature:
            return skip
        skip = F.pad(skip, (-self.context_len, -self.context_len))
        out = [
            F.softmax(self.conv_last0(skip), dim=1),
            torch.sigmoid(self.conv_last1(skip)),
        ]
        return torch.cat(out, 1)


class SpTransformer(nn.Module):  # type: ignore[misc]
    """Splice Transformer combining convolutional encoders with Sinkhorn attention."""

    def __init__(
        self,
        dim: Any = 32,
        context_len: Any = 4000,
        tissue_num: Any = 15,
        max_seq_len: Any = 8192,
        attn_depth: Any = 6,
        training: Any = False,
    ) -> None:
        """Initialize SpTransformer."""
        super().__init__()
        self.context_len = context_len
        self.encoderL = 128 + 64
        self.tissue_num = tissue_num
        self.encoder = self.load_pretrain(training)
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, dim, 1),
            nn.Conv1d(dim, dim, 1),
        )
        self.conv2 = nn.Conv1d(dim + self.encoderL, dim * 2, 1)
        #
        self.max_seq_len = max_seq_len
        self.attn = AttnBlock(dim * 2, depth=attn_depth, max_seq_len=self.max_seq_len)
        self.splice = nn.Conv1d(dim * 2, 3, 1)
        self.usage = nn.Conv1d(dim * 2, tissue_num, 1)

    def load_pretrain(self, training: Any = False) -> Any:
        """Load pretrained encoder weights."""
        model1 = SpEncoder_L(128, tissue_cnt=self.tissue_num, context_len=self.context_len)
        model2 = SpEncoder2_L(64, tissue_cnt=self.tissue_num, context_len=self.context_len)
        if training:
            print("Load previous model SpEncoder1")
            WEIGHTS = "/public/home/shenninggroup/yny/code/CellSplice/runs/T-EncoderL1-1000-128_3/best.ckpt"
            save_dict = torch.load(WEIGHTS, map_location=torch.device("cpu"))
            model1.load_state_dict(save_dict["state_dict"], strict=True)
            print("Load previous model SpEncoder2")
            WEIGHTS = "/public/home/shenninggroup/yny/code/Splice-Pytorch/model/stage1.ckpt"
            model2.load_state_dict(torch.load(WEIGHTS, map_location=torch.device("cpu")), strict=False)
            print("Done")
        return nn.ModuleList([model1, model2])

    def forward(self, x: Any) -> Any:
        """Run the forward pass."""
        target_output_len = x.size(2) - 2 * self.context_len
        target_mid_len = self.max_seq_len
        odd_fix = x.size(2) & 1
        #
        with torch.no_grad():
            feat1 = [self.encoder[i](x, feature=True) for i in [0, 1]]
        feat1 = torch.concat(feat1, dim=1)
        #
        feat2 = self.conv1(x)
        # clip 1
        seq_len = x.size(2)
        feat1 = F.pad(feat1, (-(seq_len - target_mid_len) // 2, -(seq_len - target_mid_len) // 2 + odd_fix))
        feat2 = F.pad(feat2, (-(seq_len - target_mid_len) // 2, -(seq_len - target_mid_len) // 2 + odd_fix))
        #
        emb = torch.concat([feat1, feat2], dim=1)
        emb = self.conv2(emb)
        attn = self.attn(emb)

        splice_out = self.splice(attn)
        usage_out = self.usage(attn)

        out = torch.concat([splice_out, usage_out], dim=1)

        seq_len = out.size(2)
        return F.pad(out, (-(seq_len - target_output_len) // 2 + odd_fix, -(seq_len - target_output_len) // 2))
