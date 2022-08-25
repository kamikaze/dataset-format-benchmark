import json
from pathlib import Path
from typing import Optional, Generator

from PIL import Image, UnidentifiedImageError

from dataset_format_benchmark.datasets import BaseDataset


class OwnTransportDataset(BaseDataset):
    DATASET_DIR_NAME = 'own_transport'
    IMAGE_HEIGHT = 6336
    IMAGE_WIDTH = 9504
    METADATA_FILE_NAME = 'metadata.json'
    LABELS = {
        0: 'Car',
        1: 'Truck',
        2: 'Airplane',
        3: 'Train',
        4: 'Bicycle',
        5: 'Scooter',
        6: 'Ship',
    }

    @staticmethod
    def iter_images(root: Path, min_size: Optional[int] = None) -> Generator:
        for file_item in root.iterdir():
            if file_item.is_file():
                try:
                    with Image.open(file_item) as image:
                        if min_size and min(image.size) < min_size:
                            continue

                        _type = file_item.name.split('_', maxsplit=2)[1]

                        yield _type, image
                except UnidentifiedImageError:
                    pass

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            metadata = {
                'shape': (3, self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
                'items': {
                    'file.arw': {
                        'type': 'car',
                        'make': 'audi',
                        'model': 'a6',
                        'body': 'sedan',
                        'color': 'black',
                    }
                }
            }

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)
