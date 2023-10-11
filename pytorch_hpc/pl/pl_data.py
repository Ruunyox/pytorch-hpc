import sys
import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torchvision.datasets
from typing import Tuple, Optional, Dict, List
from torchvision.datasets import *
from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

tv_datasets = torchvision.datasets.__all__


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
    splits:
        Dictionary of train/val splits of the form

            {"train_idx": np.ndarray, "val_idx": np.ndarray}

    val_size:
        float between 0.0 and 1.0 determing the size of the validation
        percentage take from the full, original training set
    shuffle_split:
        Bool telling whether or not the full training data shoudl be
        shuffled before splitting into the final train/validation sets
    train_dataloader_opts:
        Dict of kawrgs for train DataLoader
    val_dataloader_opts:
        Dict of kawrgs for validation DataLoader
    test_dataloader_opts:
        Dict of kawrgs for test DataLoader
    transforms
        Optional list of torchvision.transforms.Transforms that are applied
        to the raw dataset. Eg, for image datasets stored in PIL format:

            transforms = [torchvision.transforms.ToTensor()]
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = ".",
        splits: Optional[Dict] = None,
        shuffle_split: bool = True,
        val_size: float = 0.2,
        train_dataloader_opts: Dict = None,
        val_dataloader_opts: Dict = None,
        test_dataloader_opts: Dict = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        target_transforms: Optional[List[torch.nn.Module]] = None,
    ):
        super().__init__()
        if dataset_name not in tv_datasets:
            raise ValueError(
                f"dataset {dataset_name} not in torchvision datasets. Must be one of {tv_datasets}."
            )
        else:
            self.dataset_name = dataset_name
            self.dataset_class = getattr(
                sys.modules["torchvision.datasets"], dataset_name
            )
        self.root_dir = root_dir
        if splits is not None:
            if val_size is not None:
                warnings.warn(
                    "splits have been explicitly specified. Therefore, the additionally specified val_ratio of {val_ratio} will be overriden by the ratio implied by the splits"
                )
            assert all(k in splits for k in ["train_idx", "val_idx"])
            self.val_size = (len(splits["val_idx"])) / (
                len(splits["train_idx"]) + len(splits["val_idx"])
            )
        else:
            self.val_size = val_size
        self.splits = splits
        self.shuffle_split = True

        if train_dataloader_opts is None:
            self.train_dataloader_opts = {}
        if val_dataloader_opts is None:
            self.val_dataloader_opts = {}
        if test_dataloader_opts is None:
            self.test_dataloader_opts = {}

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None

        if target_transforms is not None:
            self.target_transforms = Compose(target_transforms)
        else:
            self.target_transforms = None

    def prepare_data(self):
        """Download the (full) train and test datasets"""

        self.dataset_train = self.dataset_class(
            self.root_dir,
            train=True,
            download=True,
            transform=self.transforms,
            target_transform=self.target_transforms,
        )
        self.dataset_test = self.dataset_class(
            self.root_dir,
            train=False,
            download=True,
            transform=self.transforms,
            target_transform=self.target_transforms,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset, applying train/val/test splits"""

        if stage == "fit" or stage is None:
            if self.splits is None:
                full_train_idx = np.arange(len(self.dataset_train))
                train_idx, val_idx = train_test_split(
                    full_train_idx, test_size=0.2, shuffle=self.shuffle_split
                )
                self.splits = {}
                self.splits["train_idx"] = train_idx
                self.splits["val_idx"] = val_idx

            self.dataset_val = Subset(self.dataset_train, self.splits["val_idx"])
            self.dataset_train = Subset(
                self.dataset_train, self.splits["train_idx"]
            )  # re-assign the train dataset after shaving off some validation data
        if stage == "test":
            if self.dataset_test is None:
                self.dataset_test = self.dataset_class(
                    self.root_dir,
                    train=False,
                    download=True,
                    transform=self.transforms,
                    target_transform=self.target_transforms,
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
