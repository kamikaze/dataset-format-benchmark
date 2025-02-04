from abc import ABC
from pathlib import Path
from typing import Generator, Sequence, Mapping

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from imageio.v3 import imread
from torch import Tensor
from torchvision import transforms

from dataset_format_benchmark.storages import ImageFileStorage

USE_CUDA = torch.cuda.is_available()


class BaseDataset(torch.utils.data.Dataset, ABC):
    DATASET_DIR_NAME = None
    METADATA_FILE_NAME = None

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.data_length = 0
        self.dataset_root_path = Path(root, self.DATASET_DIR_NAME)
        self.metadata_file_path = Path(self.dataset_root_path, 'metadata_fs.json')
        self.metadata: Sequence[Mapping] | None = None
        self.filenames: Sequence[str] = []
        self.storages: list[ImageFileStorage] = []

    def add_storage(self, storage: ImageFileStorage):
        self.storages.append(storage)

    def get_storages(self) -> Sequence[ImageFileStorage]:
        return self.storages

    def iter_files(self, root: Path, recursive: bool = True) -> Generator[Path, None, None]:
        for item in root.iterdir():
            if item.is_file():
                yield item
            elif item.is_dir() and recursive and not item.name.startswith('.'):
                yield from self.iter_files(item)

    def _download(self, force: bool = False):
        pass

    def _load_metadata(self):
        pass

    def load(self, force_download: bool = False, force_prepare: bool = False):
        self._download(force_download or force_prepare)
        self._load_metadata()

    def __len__(self):
        return self.data_length


class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, storage: ImageFileStorage, transform=None, target_transform=None):
        super().__init__()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.dataset = dataset
        self.storage = storage

    def __getitem__(self, index):
        return self.storage[self.dataset[index]]

    def __len__(self):
        return len(self.dataset)
