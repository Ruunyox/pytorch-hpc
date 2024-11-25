import torch
import lightning.pytorch as pl
from datetime import datetime, date
import numpy as np


class BasicRunStats(pl.callbacks.Callback):
    def __init__(self):
        super(BasicRunStats, self).__init__()

        self.start_time = None
        self.end_time = None
        self.train_start_time = None
        self.train_end_time = None
        self.val_start_time = None
        self.val_end_time = None
        self.train_epoch_times = []
        self.val_epoch_times = []

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_rank == 0:
            self.train_start_time = datetime.now()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_rank == 0:
            self.train_end_time = datetime.now()
            duration = (
                self.train_end_time.timestamp() - self.train_start_time.timestamp()
            )
            self.train_epoch_times.append(duration)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_rank == 0:
            self.start_time = datetime.now()

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_rank == 0:
            self.end_time = datetime.now()
            duration = self.end_time.timestamp() - self.start_time.timestamp()
            avg_train = np.average(self.train_epoch_times)
            std_train = np.std(self.train_epoch_times)
            # avg_val = np.average(self.val_epoch_times)
            print("")
            print(">>> TRAINING SUMMARY <<<")
            print("===========================================================")
            print(f"total time       :  {duration:.4f}")
            print(f"avg train epoch  :  {avg_train:.4f} +/- {std_train:.4f}")
            print("===========================================================")
            print("")
