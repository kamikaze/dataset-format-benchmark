import json
import logging
from abc import ABC
from pathlib import Path
from typing import Optional, Sequence

import imageio as iio
import numpy as np
import torch
from PIL import Image
from torch.nn import functional

from dataset_format_benchmark.datasets.utils import adjust_image


logger = logging.getLogger(__name__)


class ImageFileStorage(ABC):
    IMAGE_FILE_EXTENSION = ''
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS: tuple[int] = ()

    def __init__(self, color_spaces: Sequence):
        self.color_spaces = color_spaces

    def __str__(self):
        return self.DATASET_SUBDIR_NAME

    def _get_full_dst_file_path(self, target_dir_path: Path, file_name: str, bits: int, color_space: str):
        dst_dir_path = Path(target_dir_path, f'.{self.IMAGE_FILE_EXTENSION}{str(bits)}{color_space}')
        dst_dir_path.mkdir(exist_ok=True)

        return Path(dst_dir_path, f'{file_name}.{self.IMAGE_FILE_EXTENSION}')

    def _save_image(self, dst_path: Path, image: np.ndarray):
        iio.imwrite(dst_path, image)

    def save_image(self, target_dir_path: Path, file_name: str, bits: int, color_space: str, image: np.ndarray):
        dst_path = self._get_full_dst_file_path(target_dir_path, file_name, bits, color_space)
        self._save_image(dst_path, image)

        logger.info(f'Saved converted image in: {str(dst_path)}')

    def _resize_images(self, size: Optional[int] = None):
        item_count = self.dataset._count_images(size)
        print(f'Item count: {item_count}')

        self.dataset.dataset_subdir_path.mkdir(exist_ok=True)

        x = []
        y = []

        for idx, (label, image) in enumerate(self.dataset._iter_images(self.dataset.image_dir_path, size)):
            file_name = Path(image.filename).name
            print(f'{idx}/{item_count}: Processing {file_name}')

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                label_dir_path = Path(self.dataset.dataset_subdir_path, label)
                label_dir_path.mkdir(exist_ok=True)
                out_file_path = Path(label_dir_path, file_name)
                self._save_image(out_file_path, image)

                x.append(f'{label}/{file_name}')
                y.append(int(label) - 1)

        metadata = {
            'shape': (3, size, size),
            'filenames': x,
            'labels': y,
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.dataset.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self.dataset._get_min_size(self.dataset.IMAGE_WIDTH)
            print(f'Found minimum size: {min_size}px')

            metadata = self._resize_images(min_size)

            with open(self.dataset.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.dataset.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.x = metadata['filenames']
        self.y = metadata['labels']
        self.data_length = len(self.y)

        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def __getitem__(self, index):
        x_path = str(Path(self.dataset.dataset_subdir_path, self.x[index]))

        with Image.open(x_path) as image:
            x = np.asarray(image)

        x = torch.from_numpy(x)

        return x, self.y[index]