#!/usr/bin/env python
# ruff: noqa
# Vendored from https://github.com/autosome-ru/parade @ f8e3e02 (predictor/model/stability_data.py).
# Kept byte-for-byte (this header and a trailing newline are the only additions) so PARADE's
# published checkpoints load against the exact featurization they were trained with.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


CODES = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


class Seq2Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def n2id(n):
        return CODES[n.upper()]

    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [self.n2id(x) for x in seq.upper()]
        code = torch.tensor(seq)
        code = F.one_hot(code, num_classes=5)

        code[code[:, 4] == 1] = 0.25
        code = code[:, :4].float()
        return code.transpose(0, 1)


class StabilityData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        predict_cols: list = ['RNA/gDNA'],
        celltype_codes: list = None,
        features: tuple = ("sequence",),
    ):
        """
        :param: features (tuple): a tuple of features in selected order. Possible values:
            * 'sequence': the channels for A, T, G, and C.
            * 'positional': positional encoding feature.
        """
        self.data = df
        self.predict_cols = predict_cols

        self.features = features
        self.num_channels = self.calculate_num_channels()

        # Data preparation
        self.s2t = Seq2Tensor()

        self.prepare_data(df)

    def calculate_num_channels(self):
        n_ch = 0
        options = {"sequence": 4,
                   "positional": 1}
        for k in self.features:
            n_ch += options[k]
        return n_ch

    def prepare_data(self, df: pd.DataFrame):
        self.seqs = df['seq']
        self.values = df[self.predict_cols].to_numpy(dtype=np.float32)  # IMPORTANT!!

    @staticmethod
    def revcomp_seq_tensor(seq, batch=False):
        if batch:
            return torch.flip(seq, (1, 2))
        return torch.flip(seq, (0, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq_str = self.seqs.iloc[index]
        seq = self.s2t(seq_str)
        values = self.values[index]

        positional_ch = torch.arange(0, len(seq_str))[None, :] % 3 == 0
        elements = {
            'sequence': seq,
            'positional': positional_ch,
        }
        to_concat = [elements[k] for k in self.features]
        compiled_seq = torch.concat(to_concat)
        return compiled_seq, values


class DataLoaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        batch_per_epoch: int,
    ):
        self.dataloader = dataloader
        self.batch_per_epoch = batch_per_epoch
        self.iterator = iter(self.dataloader)

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __iter__(self):
        return self

    def reset(self):
        self.iterator = iter(self.dataloader)