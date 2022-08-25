from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.utils.data
from torch.utils.data.dataset import T_co

from dataset_format_benchmark.storages import ImageFileStorage

USE_CUDA = torch.cuda.is_available()


class BaseDataset(torch.utils.data.Dataset, ABC):
    DATASET_DIR_NAME = None
    METADATA_FILE_NAME = None

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.data_length = 0
        self.dataset_path = Path(root, self.DATASET_DIR_NAME)
        self.metadata_file_path = Path(self.dataset_path, 'metadata_fs.json')
        self.storages: list[ImageFileStorage] = []

    def add_storage(self, storage: ImageFileStorage):
        self.storages.append(storage)

    def get_storages(self) -> list[ImageFileStorage]:
        return self.storages

    @staticmethod
    @abstractmethod
    def iter_images(root: Path, min_size: Optional[int] = None) -> Iterator:
        pass

    def _download(self, force: bool = False):
        pass

    @abstractmethod
    def _prepare(self, force: bool = False):
        pass

    def load(self, force_download: bool = False, force_prepare: bool = False):
        self._download(force_download or force_prepare)
        self._prepare(force_prepare)

    def __len__(self):
        return self.data_length


class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, storage: ImageFileStorage, transform=None, target_transform=None):
        super().__init__()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.storage = storage

    def __getitem__(self, index) -> T_co:
        return self.storage[index]

    def __len__(self):
        return len(self.storage)
