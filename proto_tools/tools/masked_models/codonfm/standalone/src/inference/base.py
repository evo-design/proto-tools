# ruff: noqa
# Vendored verbatim from NVIDIA-BioNeMo/CodonFM @ 0561907ca25adae7ecbe685e8fc2882e634b1f91 (src/inference/base.py); Apache-2.0.
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
from lightning import LightningModule
from abc import ABC, abstractmethod

class BaseInference(LightningModule, ABC):
    """Abstract base class for inference modules.

    Provides a standard interface for configuring a model and running predict
    steps. Subclasses must implement `configure_model` and `_predict_step`.

    Args:
        model_path: Path to the model checkpoint to load.
        task_type: String identifier for the task (e.g., "mutation_prediction").
    """

    def __init__(self, model_path: str, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.model = None
        self.prediction_counter = 0
        self.save_hyperparameters()

    @abstractmethod
    def configure_model(self):
        """Configure the underlying model for inference.

        Must be implemented by subclasses to initialize and load weights.
        """
        pass

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step and increment an internal counter.

        Args:
            batch: A batch of input tensors.
            batch_idx: Batch index within the epoch.

        Returns:
            The output produced by `_predict_step`.
        """
        self.prediction_counter += 1
        return self._predict_step(batch, batch_idx)

    @abstractmethod
    def _predict_step(self, batch, batch_idx):
        """Perform the actual prediction step.

        Must be implemented by subclasses.

        Args:
            batch: A batch of input tensors.
            batch_idx: Batch index within the epoch.
        """
        pass