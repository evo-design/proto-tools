# ruff: noqa
# Vendored from https://github.com/autosome-ru/parade @ f8e3e02 (predictor/model/pl_regressor.py).
# Kept byte-for-byte (this header and a trailing newline are the only additions) so PARADE's
# published checkpoints load against the exact classes they were trained with.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import math


def torch_pearsonr(
    x: torch.Tensor,
    y: torch.Tensor
):
    vx = x - torch.mean(x, axis=0)
    vy = y - torch.mean(y, axis=0)

    corr = torch.sum(vx * vy, axis=0) * torch.rsqrt(torch.sum(vx ** 2, axis=0)) * torch.rsqrt(torch.sum(vy ** 2, axis=0))
    return corr


LEGNET_DEFAULTS = dict(
    seqsize=50,
    ks=3,
    in_channels=9,
    out_channels=4,
    conv_sizes=(128, 64, 64),
    mapper_size=128,
    linear_sizes=None,
    use_max_pooling=False,
    final_activation=nn.Identity
)

CRITERION_DEFAULTS = dict()

OPTIMIZER_DEFAULTS = dict(
    lr=0.01,
    weight_decay=0.1,
)

SCHEDULER_DEFAULTS = dict(
    max_lr=0.01,
    steps_per_epoch=128,
    epochs=40,
    pct_start=0.3,
    three_phase=False,
    cycle_momentum=True
)


class RNARegressor(pl.LightningModule):
    def __init__(
        self,
        # train_params=TRAIN_DEFAULTS.copy(),
        model_class=None,
        model_kws=LEGNET_DEFAULTS.copy(),
        criterion_class=nn.BCEWithLogitsLoss,
        criterion_kws=CRITERION_DEFAULTS.copy(),
        optimizer_class=torch.optim.AdamW,
        optimizer_kws=OPTIMIZER_DEFAULTS.copy(),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_kws=SCHEDULER_DEFAULTS.copy(),
        test_time_validation=False,
        initialize_weights=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        # self.automatic_optimization = False

        # self.model_class = model_class
        if self.hparams.model_class.__name__ == "LegNetClassifier":
            self.model_name = "_".join([
                f"{self.hparams.model_class.__name__}",
                f"C{'-'.join(map(str, self.hparams.model_kws['conv_sizes']))}",
                f"M{self.hparams.model_kws['mapper_size']}",
                f"L{1 if self.hparams.model_kws['linear_sizes'] is None else '-'.join(map(str, self.hparams.model_kws['linear_sizes']))}"])
        else:
            self.model_name = self.hparams.model_class.__name__

        self.model = self.hparams.model_class(**self.hparams.model_kws)

        if initialize_weights:
            self.model.apply(self.initialize_weights)

        # self.criterion_class = criterion_class
        self.criterion = self.hparams.criterion_class(**self.hparams.criterion_kws)

        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def compute_loss(self, batch, calculate_metrics=True):
        seqs, real = batch
        pred = self.model(seqs)
        #pred = torch.squeeze(pred)
        loss = self.criterion(pred, real)
        if calculate_metrics:
            with torch.no_grad():
                pearson_r = torch_pearsonr(pred, real)
            return loss, pearson_r
        else:
            return loss

    def compute_loss_smallbatch(self, batch, calculate_metrics=True):
        batched_seqs, real = batch
        smallbatch_size = batched_seqs.shape[1]
        seqs = batched_seqs.reshape((-1,) + batched_seqs.shape[-2:])
        pred = self.model(seqs)
        pred = pred.reshape((-1, smallbatch_size) + pred.shape[1:]).mean(axis=1)
        #pred = torch.squeeze(pred)
        loss = self.criterion(pred, real)
        if calculate_metrics:
            with torch.no_grad():
                pearson_r = torch_pearsonr(pred, real)
            return loss, pearson_r
        else:
            return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss, pearson_r = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for i in range(pearson_r.shape[0]):
            self.log(f"train_pearson_r_{i}", pearson_r[i].item(), on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        if self.hparams.test_time_validation:
            batched_seqs, real = batch
            smallbatch_size = batched_seqs.shape[1]
            seqs = batched_seqs.reshape((-1,) + batched_seqs.shape[-2:])
            pred = self.model(seqs)
            pred = pred.reshape((-1, smallbatch_size) + pred.shape[1:]).mean(axis=1)
            #pred = torch.squeeze(pred)
        else:
            seqs, real = batch
            pred = self.model(seqs)
            #pred = torch.squeeze(pred)
        return pred, real

    def validation_step(self, batch, batch_idx):
        if self.hparams.test_time_validation:
            loss, pearson_r = self.compute_loss_smallbatch(batch)
        else:
            loss, pearson_r = self.compute_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        for i in range(pearson_r.shape[0]):
            self.log(f"val_pearson_r_{i}", pearson_r[i].item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(
            self.model.parameters(),
            **self.hparams.optimizer_kws
        )
        # optimizer = Lion(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler_class is None:
            return {
                "optimizer": optimizer,
            }
        else:
            lr_scheduler = self.hparams.lr_scheduler_class(
                optimizer,
                **self.hparams.lr_scheduler_kws
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss",
                    "strict": True,
                    "name": self.hparams.lr_scheduler_class.__name__,
                },
            }