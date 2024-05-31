from typing import *

import torch
import torchtnt

Batch = Tuple[torch.tensor, torch.tensor]

class MyTrainUnit(torchtnt.framework.unit.TrainUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: torch.nn,
    ):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

    def train_step(self, state: torchtnt.framework.state.State, data: Batch) -> None:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_epoch_end(self, state: torchtnt.framework.state.State) -> None:
        self.lr_scheduler.step()

class MyPredictUnit(torchtnt.framework.unit.PredictUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.module = module

    def predict_step(self, state: torchtnt.framework.state.State, data: Batch) -> torch.tensor:
        inputs, targets = data
        outputs = self.module(inputs)
        return outputs, targets