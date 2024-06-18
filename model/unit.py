import torch
import torchtnt

import numpy as np

from tqdm import *
from typing import *
from torchtnt import framework

Batch = Tuple[torch.tensor, torch.tensor]

class MyTrainUnit(torchtnt.framework.unit.TrainUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: torch.nn, totalSteps = None,
        verbose = 1, totalEpochs = 20, device = "gpu",
    ):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.totalSteps = totalSteps
        self.currentEpoch = 0
        self.totalEpochs = totalEpochs
        self.verbose = verbose
        self.tqdm = None
        self.device = device

    def train_step(self, state: torchtnt.framework.state.State, data: Batch) -> None:
        if self.verbose == 1:
            self.tqdm.update(1)
        print("HERE")
        inputs, targets = data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.module(inputs)
        print("REALLY")
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_epoch_start(self, state: torchtnt.framework.state.State) -> None:
        self.currentEpoch += 1
        if self.verbose == 1:
            self.tqdm = tqdm(total=self.totalSteps, desc=f"Epoch {self.currentEpoch}/{self.totalEpochs}")
        elif self.verbose == 2 and not self.tqdm:
            self.tqdm = tqdm(total=self.totalEpochs)

    def on_train_epoch_end(self, state: torchtnt.framework.state.State) -> None:
        self.lr_scheduler.step()
        if self.verbose==1:
            if not self.totalSteps:
                self.totalSteps = self.tqdm.n
            self.tqdm.close()
        elif self.verbose==2:
            self.tqdm.update(1)

class MyPredictUnit(torchtnt.framework.unit.PredictUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        totalSteps = None,
    ):
        super().__init__()
        self.module = module
        self.outputs = np.array([])
        self.labels = np.array([])
        self.totalSteps = totalSteps

    def on_predict_epoch_start(self, state: torchtnt.framework.state.State) -> None:
        self.tqdm = tqdm(total=self.totalSteps)

    def predict_step(self, state: torchtnt.framework.state.State, data: Batch) -> torch.tensor:
        self.tqdm.update(1)
        inputs, targets = data
        outputs = self.module(inputs)
        if len(self.outputs) == 0:
            self.outputs = outputs.detach().cpu().numpy()
            self.labels = targets.detach().cpu().numpy()
        else:
            self.outputs = np.append(self.outputs, outputs.detach().cpu().numpy(), axis=0)
            self.labels = np.append(self.labels, targets.detach().cpu().numpy(), axis=0)
        return outputs
    
    def on_predict_epoch_end(self, state: torchtnt.framework.state.State) -> None:
        self.tqdm.close()
