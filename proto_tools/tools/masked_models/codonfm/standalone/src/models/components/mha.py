# ruff: noqa
# Vendored verbatim from NVIDIA-BioNeMo/CodonFM @ 0561907ca25adae7ecbe685e8fc2882e634b1f91 (src/models/components/mha.py); Apache-2.0.
# Only this header is prepended; the body is unchanged so the published Encodon
# checkpoints load and run exactly as upstream. Do not edit — re-vendor to update.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from einops import rearrange

import xformers.ops as xops
from transformers.pytorch_utils import Conv1D

from src.models.components.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

class MultiHeadAttention(nn.Module):
    """
    Multi-Headed Self Attention module using xformers for memory-efficient attention
    and Rotary Positional Embeddings.

    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply to the attention scores.
        rotary_theta (float, default=10000.0): The base for the geometric progression
            used to compute the rotation angles for Rotary Positional Embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_theta: float = 1e4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.rotary_emb = RotaryEmbedding(
            dim=embed_dim // num_heads,
            theta=rotary_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask (torch.Tensor): Mask to prevent attention to certain positions.
                It can be a padding mask or a causal mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # These assertions are likely due to hardware constraints or specific optimizations
        # in the version of xformers being used, which may require sequence lengths
        # to be a multiple of 8 for optimal performance with Tensor Cores.
        assert attention_mask.shape[-1] % 8 == 0, "attention_mask must be divisible by 8"

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = rearrange(q, "b q (h d) -> b q h d", h=self.num_heads)
        k = rearrange(k, "b k (h d) -> b k h d", h=self.num_heads)
        v = rearrange(v, "b v (h d) -> b v h d", h=self.num_heads)
        
        # Apply rotary positional embeddings to query and key.
        cos, sin = self.rotary_emb(q)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # The attention_mask is transformed into a bias tensor suitable for xformers.
        # The repeat calls ensure the mask is broadcastable across all attention heads
        # and reshapes a padding mask into a full attention matrix mask.
        attn_bias = attention_mask.repeat(1, 1, attention_mask.size(-1), 1)
        attn_bias = attn_bias.to(q.dtype)
        attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)

        # Memory-efficient attention from xformers.
        x = xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            op=None,
            attn_bias=attn_bias,
            p=self.dropout_rate if self.training else 0.0,
        )

        # x: (batch_size, query_seq_len, n_head, head_dim)
        x = rearrange(x, "b q h d -> b q (h d)", h=self.num_heads)

        return x
