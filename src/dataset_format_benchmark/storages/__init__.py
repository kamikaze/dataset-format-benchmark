import json
import logging
from abc import ABC
from pathlib import Path
from typing import Optional, Sequence

import imageio as iio
import numpy as np
import torch
from PIL import Image
from imageio.v3 import imread
from torch import Tensor
from torch.nn import functional
from torchvision import transforms

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

    @staticmethod
    def _load_image(image_path: Path) -> Image:
        """Loads an image while preserving its original bit depth."""
        image = imread(image_path)

        # Handle multi-channel images (e.g., RGB) vs single-channel (grayscale)
        if image.ndim == 2:  # Grayscale image
            # 16-bit or 8-bit grayscale
            mode = 'I;16' if image.dtype == np.uint16 else 'L'
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
            mode = None  # Keep default
        else:
            raise ValueError(f'Unsupported image shape: {image.shape}')

        return Image.fromarray(image, mode=mode) if mode else Image.fromarray(image)

    @staticmethod
    def _to_tensor(image: Image) -> Tensor:
        if image.mode == 'I;16':
            return torch.tensor(np.array(image), dtype=torch.float32)  # Keep 16-bit precision
        else:
            return transforms.ToTensor()(image)  # Standard 8-bit conversion

    def load_from_dataset(self, dataset, index):
        x_path = str(Path(dataset.dataset_subdir_path, dataset.x[index]))

        with Image.open(x_path) as image:
            x = np.asarray(image)

        x = torch.from_numpy(x)

        return x, self.y[index]
