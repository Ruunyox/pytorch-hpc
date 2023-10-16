import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torch
from typing import Tuple, Callable, Type, Union

OptimType = torch.optim.Optimizer
SchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]

__all__ = ["LightningModel"]


class LightningModel(pl.LightningModule):
    """Model for PyTorch Lightning training

    Parameters
    ----------
    model:
        `torch.nn.Module` instance for taking in inputs and outputing predictions
    loss_function:
        `torch.nn.Module` instance for calculating deviations from labels
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the module

        Parameters
        ----------
        x:
            `torch.Tensor` input to the model

        Returns
        -------
        y:
            `torch.Tensor` output predicted by the model
        """
        return self.model(x)

    def step(self, batch: Tuple, stage: str) -> torch.tensor:
        """Computes the loss for a single batch passed through the model

        Parameters
        ----------
        batch:
            `Tuple` of input `torch.Tensors` representing the input and the label
        stage:
            `str` denoting the stage of training. Can be `train`, `validation`, or `test`

        Returns
        -------
        loss:
            `torch.Tensor` loss computed over the input batch
        """
        x, y = batch
        with torch.set_grad_enabled(stage == "train"):
            output = self.forward(x)
            loss = self.loss_function(output, y)
        return loss

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """training step over one batch"""
        loss = self.step(batch, "train")
        training_step_output = self.trainer.strategy.reduce(loss)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """validation step over one batch"""
        loss = self.step(batch, "validation")
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        """Logs the average training loss over the entire epoch"""
        with torch.no_grad():
            epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log(
            f"training_loss",
            epoch_average,
            sync_dist=True,
            prog_bar=True,
        )
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Logs the average validation loss over the entire epoch"""
        with torch.no_grad():
            epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss",
            epoch_average,
            sync_dist=True,
            prog_bar=True,
        )
        self.validation_step_outputs.clear()
