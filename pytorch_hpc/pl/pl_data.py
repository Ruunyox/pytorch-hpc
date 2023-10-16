import sys
import builtins
import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torchvision.datasets
from typing import Tuple, Optional, Dict, List, Any
from torchvision.datasets import *
from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

__all__ = ["TorchvisionDataModule"]

transform_map = {"ToTensor": torchvision.transforms.ToTensor}


class TorchvisionDataModule(pl.LightningDataModule):
    """Wraps commonly available Torchvsion Datasets for quick
    inspection with Pytorch Lighting.
    Downloads and splits into train/val/test sets according
    to user specified options.

    Parameters
    ----------
    dataset_name:
        String specifying the name of the torchvision dataset to use/download
    root_dir:
        String specifying where the data should be downloaded to
    splits_fn:
        .npz filename containtin a dictionary of train/val splits of the form

            {"train_idx": np.ndarray, "val_idx": np.ndarray}

        If None, a default train/validation split will be set as the first 4/5th /
        last 1/5th of the original train dataset.
    val_size:
        float between 0.0 and 1.0 determing the size of the validation
        percentage take from the full, original training set
    train_dataloader_opts:
        Dict of kawrgs for train DataLoader
    val_dataloader_opts:
        Dict of kawrgs for validation DataLoader
    test_dataloader_opts:
        Dict of kawrgs for test DataLoader
    transform
        Optional list of torchvision.transform.Transforms that are applied
        to the raw dataset. Eg, for image datasets stored in PIL format:

            transform = [torchvision.transform.ToTensor()]
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = ".",
        splits_fn: Optional[str] = None,
        train_dataloader_opts: Optional[Dict] = None,
        val_dataloader_opts: Optional[Dict] = None,
        test_dataloader_opts: Optional[Dict] = None,
        transform: Optional[List[str]] = None,
        target_transform: Optional[List[str]] = None,
    ):
        super().__init__()
        if dataset_name not in torchvision.datasets.__all__:
            raise ValueError(
                f"dataset {dataset_name} not in torchvision datasets. Must be one of {tv_datasets}."
            )
        else:
            self.dataset_name = dataset_name
            self.dataset_class = getattr(
                sys.modules["torchvision.datasets"], dataset_name
            )
        self.root_dir = root_dir

        full_train_len = len(
            self.dataset_class(self.root_dir, train=True, download=False)
        )

        if splits_fn is not None:
            self.splits = np.load(splits_fn)
        else:
            self.splits = {
                "train_idx": np.arange(0, int(full_train_len * 4.0 / 5.0)),
                "val_idx": np.arange(
                    int(full_train_len * 4.0 / 5.0), full_train_len
                ),
            }

        if train_dataloader_opts is None:
            self.train_dataloader_opts = {}
        else:
            self.train_dataloader_opts = train_dataloader_opts

        if val_dataloader_opts is None:
            self.val_dataloader_opts = {}
        else:
            self.val_dataloader_opts = val_dataloader_opts

        if test_dataloader_opts is None:
            self.test_dataloader_opts = {}
        else:
            self.test_dataloader_opts = test_dataloader_opts

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.transform = (
            Compose([transform_map[t]() for t in transform])
            if transform is not None
            else None
        )
        self.target_transform = (
            Compose([transform_map[t]() for t in transform])
            if target_transform is not None
            else None
        )

    def _get_dataset(
        self, train: bool = True, return_dataset: bool = True, download: bool = True
    ) -> Optional[torch.utils.data.Dataset]:
        """Helper method to download and optionally return
        dataset objects.
        """

        dataset = self.dataset_class(
            self.root_dir,
            train=train,
            download=download,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        if return_dataset is True:
            return dataset

    def prepare_data(self):
        """Download the (full) train and test datasets.
        For DDP, we don't want to assign/store variables here.
        This method is only called by the Trainer on the first
        rank in order to prevent multiple calls on each process.
        """

        self._get_dataset(
            train=True,
            return_dataset=False,
            download=True,
        )
        self._get_dataset(
            train=False,
            return_dataset=False,
            download=True,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset, applying train/val/test splits. Keep in mind that under a DDP strategy,
        train/validation splitting should NOT be done here - unless `seed_everything` is not None, then
        leakage between the training and validation sets can occur as the different processes may have
        different runtime seeds.
        """

        if stage in ["fit", "validate"] or stage is None:
            dataset_train = self._get_dataset(
                train=True, return_dataset=True, download=False
            )
            self.dataset_val = Subset(dataset_train, self.splits["val_idx"])
            self.dataset_train = Subset(dataset_train, self.splits["train_idx"])
        if stage == "test":
            self.dataset_test = self._get_dataset(
                train=False, return_dataset=True, download=False
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.dataset_train, **self.train_dataloader_opts)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.dataset_val, **self.val_dataloader_opts)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.dataset_test, **self.test_dataloader_opts)

    def predict_dataloader(
        self, x: torch.utils.data.Dataset, dataloader_opts: Optional[Dict] = None
    ) -> torch.utils.data.DataLoader:
        if dataloader_opts is None:
            dataloader_opts = {}
        return DataLoader(x, **dataloader_opts)
