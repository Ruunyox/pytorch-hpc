import sys
import builtins
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torchvision.datasets
from typing import Tuple, Optional, Dict, List, Any, Callable
import torch
from torchvision.datasets import *
from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

try:
    import torch_geometric
    from torch_geometric.loader import DataLoader as GeomDataLoader
except:
    warnings.warn(
        "Failed `from torch_geometric.loader import DataLoader as GeomDataLoader`"
    )


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
    train_dataloader_opts:
        Dict of kawrgs for train DataLoader
    val_dataloader_opts:
        Dict of kawrgs for validation DataLoader
    test_dataloader_opts:
        Dict of kawrgs for test DataLoader
    transform
        Optional list of torchvision.transform.Transform class names are applied
        to the raw inputs of dataset. Eg, for image datasets stored in PIL format:

            transform = ["ToTensor"]
    target_transform
        Optional list of torchvision.transform.Transform class names are applied
        to the raw labels of the dataset. Eg, for image datasets stored in PIL format:

            transform = ["ToTensor"]
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
                f"dataset {dataset_name} not in torchvision datasets. Must be one of {torchvision.datasets.__all__}."
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
                "val_idx": np.arange(int(full_train_len * 4.0 / 5.0), full_train_len),
            }

        self._initialize_loaders(train_dataloader_opts, val_dataloader_opts)

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

    def _initialize_loaders(self, train_dataloader_opts, val_dataloader_opts):
        if train_dataloader_opts is None:
            self.train_dataloader_opts = {}
        else:
            self.train_dataloader_opts = train_dataloader_opts

        if val_dataloader_opts is None:
            self.val_dataloader_opts = {}
        else:
            self.val_dataloader_opts = val_dataloader_opts

    def _get_dataset(
        self, train: bool = True, return_dataset: bool = True, download: bool = True
    ) -> Optional[torch.utils.data.Dataset]:
        """Helper method to download and optionally return
        dataset objects. All transforms specified in self.__init__() are applied.

        Paramters
        ---------
        train:
            if True, the predetermined train set of the dataset will be grabbed
        return_dataset:
            if True, the `Dataset` object is returned
        download:
            if True, the dataset will be downloaded and stored locally in `self.root_dir`

        Returns
        -------
        dataset:
            If `return_dataset` is `True`, the `Dataset` instance with the specified
            options.
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
        """Returns the train dataloader with the options specified in self.__init__()"""
        return DataLoader(self.dataset_train, **self.train_dataloader_opts)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the validation dataloader with the options specified in self.__init__()"""
        return DataLoader(self.dataset_val, **self.val_dataloader_opts)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the test dataloader with the options specified in self.__init__()"""
        return DataLoader(self.dataset_test, **self.test_dataloader_opts)

    def predict_dataloader(
        self, dataset: torch.utils.data.Dataset, dataloader_opts: Optional[Dict] = None
    ) -> torch.utils.data.DataLoader:
        """Returns a prediction dataloader

        Parameters
        ----------
        dataset:
            `Dataset` instance over which predictions will be made
        dataloader_opts:
            Dictionary of kwargs for the prediction dataloader
        """

        if dataloader_opts is None:
            dataloader_opts = {}
        return DataLoader(dataset, **dataloader_opts)


class GeometricDataModule(TorchvisionDataModule):
    """Wrapper for `torch_geometric.datasets` datasets

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
    train_dataloader_opts:
        Dict of kawrgs for train DataLoader
    val_dataloader_opts:
        Dict of kawrgs for validation DataLoader
    test_dataloader_opts:
        Dict of kawrgs for test DataLoader
    transform:
        Optional Callable to transform a `torch_geometric.data.Data`
        after saving to disk
    pre_transform:
        Optional Callable to transform a `torch_geometric.data.Data`
        before saving to disk
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = ".",
        splits_fn: Optional[str] = None,
        train_dataloader_opts: Optional[Dict] = None,
        val_dataloader_opts: Optional[Dict] = None,
        test_dataloader_opts: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super(TorchvisionDataModule, self).__init__()
        if dataset_name not in torch_geometric.datasets.__all__:
            raise ValueError(
                f"dataset {dataset_name} not in torch_geometric datasets. Must be one of {torch_geometric.datasets.__all__}."
            )
        else:
            self.dataset_name = dataset_name
            self.dataset_class = getattr(
                sys.modules["torch_geometric.datasets"], dataset_name
            )
        self.root_dir = root_dir

        full_train_len = len(self.dataset_class(self.root_dir))

        if splits_fn is not None:
            self.splits = np.load(splits_fn)
        else:
            self.splits = {
                "train_idx": np.arange(0, int(full_train_len * 9.0 / 10.0)),
                "val_idx": np.arange(int(full_train_len * 9.0 / 10.0), full_train_len),
            }

        self._initialize_loaders(train_dataloader_opts, val_dataloader_opts)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    def _get_dataset(
        self, split: str = "train", return_dataset: bool = True
    ) -> Optional[torch.utils.data.Dataset]:
        """Helper method to download and optionally return
        dataset objects. All transforms specified in self.__init__() are applied.

        Paramters
        ---------
        train:
            One of 'train' or 'test'
        return_dataset:
            if True, the `Dataset` object is returned

        Returns
        -------
        dataset:
            If `return_dataset` is `True`, the `Dataset` instance with the specified
            options.
        """

        dataset = self.dataset_class(
            self.root_dir,
            split=split,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

        if return_dataset is True:
            return dataset

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset, applying train/val/test splits. Keep in mind that under a DDP strategy,
        train/validation splitting should NOT be done here - unless `seed_everything` is not None, then
        leakage between the training and validation sets can occur as the different processes may have
        different runtime seeds.
        """

        if stage in ["fit", "validate"] or stage is None:
            dataset_train = self._get_dataset(split="train", return_dataset=True)
            self.dataset_val = Subset(dataset_train, self.splits["val_idx"])
            self.dataset_train = Subset(dataset_train, self.splits["train_idx"])
        if stage == "test":
            self.dataset_test = self._get_dataset(
                split="test",
                return_dataset=True,
            )

    def prepare_data(self):
        """Download the (full) train and test datasets.
        For DDP, we don't want to assign/store variables here.
        This method is only called by the Trainer on the first
        rank in order to prevent multiple calls on each process.
        """

        self._get_dataset(
            split="train",
            return_dataset=False,
        )
        self._get_dataset(
            split="test",
            return_dataset=False,
        )

    def train_dataloader(self) -> GeomDataLoader:
        """Returns the train dataloader with the options specified in self.__init__()"""
        return GeomDataLoader(self.dataset_train, **self.train_dataloader_opts)

    def val_dataloader(self) -> GeomDataLoader:
        """Returns the validation dataloader with the options specified in self.__init__()"""
        return GeomDataLoader(self.dataset_val, **self.val_dataloader_opts)

    def test_dataloader(self) -> GeomDataLoader:
        """Returns the test dataloader with the options specified in self.__init__()"""
        return GeomDataLoader(self.dataset_test, **self.test_dataloader_opts)

    def predict_dataloader(
        self, dataset: torch.utils.data.Dataset, dataloader_opts: Optional[Dict] = None
    ) -> GeomDataLoader:
        """Returns a prediction dataloader

        Parameters
        ----------
        dataset:
            `Dataset` instance over which predictions will be made
        dataloader_opts:
            Dictionary of kwargs for the prediction dataloader
        """

        if dataloader_opts is None:
            dataloader_opts = {}
        return GeomDataLoader(dataset, **dataloader_opts)
