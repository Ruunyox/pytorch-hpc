import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torch
import torch_geometric
from typing import Tuple, Callable, Type, Union, Dict, Optional

OptimType = torch.optim.Optimizer
SchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]


class StandardGraphExpander(torch.nn.Module):
    """Wrapper for standard graph input expansion

    Parameters
    ----------
    scalar:
        If `True`, single channel `Data.x` inputs will be reshaped as
        `(num_nodes, -1)`
    """

    def __init__(
        self, scalar: bool = False, input_cast_type: str = "torch.FloatTensor"
    ):
        super(StandardGraphExpander, self).__init__()
        self.scalar = scalar
        self.input_cast_type = input_cast_type

    def forward(self, data: torch_geometric.data.Data) -> Dict[str, torch.Tensor]:
        return {
            "x": data.x.view(data.x.shape[0], -1).type(self.input_cast_type)
            if self.scalar
            else data.x.type(self.input_cast_type),
            "edge_index": data.edge_index,
            "edge_weight": data.edge_weight,
            "edge_attr": data.edge_attr,
            "batch": data.batch,
        }


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


class LightningGraphModel(pl.LightningModule):
    """Model for PyTorch Lightning training with Pytorch Geometric

    Parameters
    ----------
    model:
        `torch.nn.Module` instance for taking in inputs and outputing predictions
    loss_function:
        `torch.nn.Module` instance for calculating deviations from labels
    data_expansion:
        `torch.nn.Module` that expands an incoming `torch_geometric.data.Data` object
        into a `Dict[str, torch.Tensor]` of direct model inputs
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        data_expansion: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.data_expansion = data_expansion
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, data_batch: torch_geometric.data.Data) -> torch.Tensor:
        """Forward pass through the module

        Parameters
        ----------
        data_batch:
            `torch_geometric.data.Data` input to the model

        Returns
        -------
        y:
            `torch.Tensor` output predicted by the model
        """
        expanded_data = self.data_expansion(data_batch)
        return self.model(**expanded_data)

    def step(self, data_batch: torch_geometric.data.Data, stage: str) -> torch.tensor:
        """Computes the loss for a single batch passed through the model

        Parameters
        ----------
        DataBatch:
            `torch.data.Data` instance containg graph-batched data and targets
        stage:
            `str` denoting the stage of training. Can be `train`, `validation`, or `test`

        Returns
        -------
        loss:
            `torch.Tensor` loss computed over the input batch
        """
        with torch.set_grad_enabled(stage == "train"):
            output = self.forward(data_batch)
            loss = self.loss_function(output.squeeze(), data_batch.y)
        return loss

    def training_step(
        self, data_batch: torch_geometric.data.Data, batch_idx: int
    ) -> torch.Tensor:
        """training step over one batch"""
        loss = self.step(data_batch, "train")
        training_step_output = self.trainer.strategy.reduce(loss)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(
        self, data_batch: torch_geometric.data.Data, batch_idx: int
    ) -> torch.Tensor:
        """validation step over one batch"""
        loss = self.step(data_batch, "validation")
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
