from pathlib import Path
from typing import Sequence

import imageio as iio
import numpy as np

from dataset_format_benchmark.storages import ImageFileStorage


class JPEGImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'jpeg'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS = (8, )

    def __init__(self, color_spaces: Sequence, quality: int = 100):
        super().__init__(color_spaces)
        self.quality = quality

    def __str__(self):
        return f'{self.DATASET_SUBDIR_NAME}_{self.quality}'

    def _get_full_dst_file_path(self, target_dir_path: Path, file_name: str, bits: int, color_space: str):
        dst_dir_path = Path(target_dir_path, f'.{self.IMAGE_FILE_EXTENSION}{str(bits)}{color_space}{self.quality}')
        dst_dir_path.mkdir(exist_ok=True)

        return Path(dst_dir_path, f'{file_name}.{self.IMAGE_FILE_EXTENSION}')

    def _save_image(self, dst_path: Path, image: np.ndarray):
        iio.imwrite(dst_path, image, quality=self.quality)


class PNGImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'png'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS = (8, )

    def _save_image(self, dst_path: Path, image: np.ndarray):
        iio.imwrite(dst_path, image, optimize=True)


class BMPImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'bmp'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS = (8, )


class TIFFImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'tiff'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS = (8, 16, )


class WebPImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'webp'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
    SUPPORTED_BPS = (8, )
