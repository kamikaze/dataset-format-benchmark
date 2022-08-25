import csv
import json
from operator import itemgetter
from pathlib import Path
from shutil import rmtree
from typing import Iterator

import numpy as np
import torch
import torch.utils.data
import zarr
from PIL import Image
from PIL.Image import Resampling
from numpy.lib.format import open_memmap
from torch.nn import functional
from torchvision.io import read_image

from dataset_format_benchmark.datasets.kaggle import KaggleDataset


class DatasetFlickrImage(KaggleDataset):
    BYTES_PER_VALUE = 16 / 8
    DATASET_NAME = 'hsankesara/flickr-image-dataset'
    DATASET_DIR_NAME = 'flickr30k_images'
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    RESULT_FILE_NAME = 'results.csv'

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

    def _download(self, force: bool = False):
        if force or not self.dataset_path.exists():
            super()._download(force)
            # Removing duplicated data
            rmtree(Path(self.image_dir_path, self.DATASET_DIR_NAME), ignore_errors=True)
            Path(self.image_dir_path, self.RESULT_FILE_NAME).unlink(missing_ok=True)

    def _prepare(self, force: bool = False):
        pass


class DatasetFlickrImageFilesystem(DatasetFlickrImage):
    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            with open(self.result_file_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                # Skipping header
                next(reader)

                results = tuple(reader)
                get_filename = itemgetter(0)
                filenames = set(map(get_filename, results))

                with open(self.metadata_file_path, 'w') as fm:
                    # TODO: what to use???
                    metadata = {filename: 0 for filename in filenames}
                    json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.x = tuple(metadata.keys())
        self.y = tuple(metadata.values())
        self.data_length = len(self.y)

        # How do I know class count when passing transformation? idk...
        if self.target_transform:
            self.y = self.target_transform(self.y)

        # So I transform it manually inside
        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def __getitem__(self, index):
        image_path = Path(self.image_dir_path, self.x[index])
        x = read_image(str(image_path))
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        x = x / 255.0

        return x, y


class DatasetFlickrImageZarr(DatasetFlickrImage):
    def __init__(self, root: Path, transform=None, target_transform=None, chunks=None):
        super().__init__(root, transform, target_transform)
        self.dataset_file_path = Path(self.dataset_path, 'data.zarr')
        self.chunks = chunks

    def _prepare(self, force: bool = True):
        if force or not self.dataset_file_path.exists():
            with open(self.result_file_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                # Skipping header
                next(reader)

                results = tuple(reader)
                get_filename = itemgetter(0)
                filenames = set(map(get_filename, results))

            data_length = len(filenames)
            dataset_shape = (data_length, 3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

            dataset = zarr.open(str(self.dataset_file_path), mode='w')
            x = dataset.zeros('samples', shape=dataset_shape, chunks=self.chunks, dtype='float16')
            y = dataset.zeros('labels', dtype='int8', shape=data_length)
            labels = []

            for idx, filename in enumerate(filenames):
                if idx % 100 == 0:
                    print(f'Preparing {idx}/{data_length}')

                image_file_path = Path(self.image_dir_path, filename)
                image = Image.open(image_file_path)
                width, height = image.size  # Get dimensions
                new_size = min(width, height)

                left = int((width - new_size) / 2)
                top = int((height - new_size) / 2)
                right = int((width + new_size) / 2)
                bottom = int((height + new_size) / 2)

                # Crop the center of the image
                image = image.crop((left, top, right, bottom))

                image = image.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), resample=Resampling.LANCZOS)
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image

                # TODO: what to use???
                labels.append(0)

            y[:] = np.array(labels)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        self.root = zarr.open(str(self.dataset_file_path), mode='r')
        self.x = self.root['samples']
        self.y = functional.one_hot(torch.tensor(self.root['labels'], dtype=torch.long))
        self.data_length = len(self.y)

    def __getitem__(self, index):
        # Getting a single ndarray from Zarr Array
        x = self.x[index]
        # Creating a Tensor from ndarray
        x = torch.from_numpy(x)

        return x, self.y[index]


class DatasetFlickrImageNumpyMmap(DatasetFlickrImage):
    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataset_file_name = 'data.npy'
        self.dataset_path = Path(Path(__file__).parent, root, self.DATASET_DIR_NAME)
        self.dataset_file_path = Path(self.dataset_path, self.dataset_file_name)
        self.metadata_file_path = Path(self.dataset_path, 'metadata_numpy_mmap.json')

    def _prepare(self, force: bool = True):
        if force or not self.dataset_file_path.exists() or not self.metadata_file_path.exists():
            with open(self.result_file_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                # Skipping header
                next(reader)

                results = tuple(reader)
                get_filename = itemgetter(0)
                filenames = set(map(get_filename, results))

            item_count = len(filenames)
            print(f'Item count: {item_count}')
            dataset_shape = (item_count, 3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

            x = open_memmap(
                str(self.dataset_file_path), mode='w+', dtype='float16', shape=dataset_shape
            )
            y = []

            for idx, filename in enumerate(filenames):
                if idx % 100 == 0:
                    print(f'Preparing {idx}/{item_count}')

                image_file_path = Path(self.image_dir_path, filename)
                image = Image.open(image_file_path)
                width, height = image.size  # Get dimensions
                new_size = min(width, height)

                left = int((width - new_size) / 2)
                top = int((height - new_size) / 2)
                right = int((width + new_size) / 2)
                bottom = int((height + new_size) / 2)

                # Crop the center of the image
                image = image.crop((left, top, right, bottom))

                image = image.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), resample=Resampling.LANCZOS)
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image[:, :]

                # TODO: what to use???
                y.append(0)

            x.flush()

            with open(self.metadata_file_path, 'w') as f:
                metadata = {
                    'shape': dataset_shape,
                    'labels': y
                }
                json.dump(metadata, f)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.data_length = self.dataset_shape[0]
        self.y = metadata['labels']

        self.x = open_memmap(
            str(self.dataset_file_path), mode='r', dtype='float16', shape=self.dataset_shape
        )
        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def iter_images(self) -> Iterator:
        pass

    def __getitem__(self, index):
        x = self.x[index]
        x = np.array(x)
        x = torch.from_numpy(x)

        return x, self.y[index]


class DatasetFlickrImageHDF5(KaggleDataset):
    pass


class DatasetFlickrImageCuPy(KaggleDataset):
    pass
