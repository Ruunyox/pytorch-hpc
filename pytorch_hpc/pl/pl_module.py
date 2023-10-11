import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torch
from typing import Tuple


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        optimizer_class: OptimizerCallable,
        lr_scheduler_class: LRSchedulerCallable,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the module"""
        return self.model(x)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        return loss

    def configure_optimizers(self):
        if self.optimizer_class is not None:
            optimizer = self.optimizer_class(self.parameters())
        else:
            optimizer = None
        return optimizer

    def validation_step(self, batch: otrch.utils.data.Data) -> torch.Tensor:
        self.model.eval()
        x, y = batch
        loss = self.model.training_step(x)
        model.train()
        return loss
