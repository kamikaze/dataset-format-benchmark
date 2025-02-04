import json
import logging
from enum import auto, StrEnum
from pathlib import Path
from typing import Sequence

import msgspec
from msgspec import Struct

from dataset_format_benchmark.datasets import BaseDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VehicleKind(StrEnum):
    CAR = auto()
    MOTORCYCLE = auto()


class VehicleBody(StrEnum):
    SEDAN = auto()
    HATCHBACK = auto()
    WAGON = auto()
    MINIVAN = auto()
    SUV = auto()
    TRUCK = auto()
    BUS = auto()


class LabelSet(Struct):
    kind: VehicleKind
    body: VehicleBody
    color: str
    make: str
    model: str


class DatasetItem(Struct):
    image: str
    labels: LabelSet


class OwnTransportDataset(BaseDataset):
    DATASET_DIR_NAME = 'own_transport'
    IMAGE_HEIGHT = 6336
    IMAGE_WIDTH = 9504
    METADATA_FILE_NAME = 'metadata.json'

    LABELS = {
        0: None,
        1: 'car',
        2: 'motorcycle',
    }

    CAR_TYPE_LABELS = {
        0: None,
        1: 'sedan',
        2: 'hatchback',
        3: 'wagon',
        4: 'minivan',
        5: 'suv',
        6: 'truck',
        7: 'bus',
    }

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.metadata: Sequence[DatasetItem] | None = None

        with open(self.metadata_file_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.dataset_root_path / item['image']
        image = self._load_image(image_path)
        image_tensor = self._to_tensor(image)

        if self.transform:
            image = self.transform(image_tensor)

        labels = item.get('labels')

        return image, labels

    def _load_metadata(self):
        with open(self.metadata_file_path, 'rb') as f:
            self.metadata = msgspec.json.decode(f.read(), type=Sequence[DatasetItem])

    # def load(self, force_download: bool = False, force_prepare: bool = False):
    #     super().load(force_download, force_prepare)
    #
    #     with open(self.dataset.metadata_file_path) as f:
    #         metadata = json.load(f)
    #
    #     self.dataset_shape = metadata['shape']
    #     self.x = metadata['filenames']
    #     self.y = metadata['labels']
    #     self.data_length = len(self.y)
    #
    #     self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))
