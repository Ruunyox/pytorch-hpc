import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torch
import torch_geometric
from typing import Tuple, Callable, Type, Union, Dict, Optional
from torchmetrics import Accuracy

OptimType = torch.optim.Optimizer
SchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]


class LightningModel(pl.LightningModule):
    """Model for PyTorch Lightning training

    Parameters
    ----------
    model:
        `torch.nn.Module` instance for taking in inputs and outputing predictions
    loss_function:
        `torch.nn.Module` instance for calculating deviations from labels
    task: Training task. Must be one of `LightningModel.tasks`
    """

    tasks = ["regression", "multiclass"]

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        task: str = "multiclass",
        jit_compile: bool = False,
    ):
        super().__init__()
        self.model = model
        self.jit_compile = jit_compile
        assert task in LightningModel.tasks
        self.task = task
        self.loss_function = (
            torch.jit.script(loss_function) if self.jit_compile else loss_function
        )
        if self.task == "multiclass":
            self.accuracy = Accuracy(self.task, num_classes=model.out_dim)
            self.accuracy_step_outputs = []
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
        return loss, output

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """training step over one batch"""
        loss, _ = self.step(batch, "train")
        training_step_output = self.trainer.strategy.reduce(loss)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """validation step over one batch"""
        loss, output = self.step(batch, "validation")
        if self.task == "multiclass":
            acc = self.accuracy(output, batch[1])
            self.accuracy_step_outputs.append(acc)
        self.validation_step_outputs.append(loss)
        return loss

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
            self.validation_step_outputs.clear()
            if self.task == "multiclass":
                acc_average = torch.stack(self.accuracy_step_outputs).mean()
                self.accuracy_step_outputs.clear()
        self.log(
            f"validation_loss",
            epoch_average,
            sync_dist=True,
            prog_bar=True,
        )
        if self.task == "multiclass":
            self.log(
                f"validation_accuracy",
                acc_average,
                sync_dist=True,
                prog_bar=True,
            )


class LightningGraphModel(LightningModel):
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
    task:
        `str` specifying the training task. Must be one of `LightningGraphModel.tasks`.
    """

    tasks = ["regression", "multiclass"]

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        data_expansion: torch.nn.Module,
        task: str = "regression",
        jit_compile: bool = False,
    ):
        super(LightningGraphModel, self).__init__(
            model=model, loss_function=loss_function, task=task, jit_compile=jit_compile
        )
        self.data_expansion = data_expansion

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

    def step(
        self, data_batch: torch_geometric.data.Data, stage: str
    ) -> Tuple[torch.tensor, torch.tensor]:
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
        output:
            `torch.tensor` of raw model outputs
        """
        with torch.set_grad_enabled(stage == "train"):
            output = self.forward(data_batch)
            loss = self.loss_function(output.squeeze(), data_batch.y)
        return loss, output

    def training_step(
        self, data_batch: torch_geometric.data.Data, batch_idx: int
    ) -> torch.Tensor:
        """training step over one batch"""
        loss, _ = self.step(data_batch, "train")
        training_step_output = self.trainer.strategy.reduce(loss)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(
        self, data_batch: torch_geometric.data.Data, batch_idx: int
    ) -> torch.Tensor:
        """validation step over one batch"""
        loss, _ = self.step(data_batch, "validation")
        if self.task == "multiclass":
            acc = self.accuracy(output, data_batch.y)
            self.accuracy_step_outputs.append(acc)
        self.validation_step_outputs.append(loss)
        return loss
