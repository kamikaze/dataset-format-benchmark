import csv
import json
from operator import itemgetter
from pathlib import Path
from typing import Optional, Mapping, Iterator, Generator

import numpy as np
import torch
import torch.utils.data
import zarr
from PIL import Image, UnidentifiedImageError
from numpy.lib.format import open_memmap
from torch.nn import functional
from torchvision.io import read_image

from dataset_format_benchmark.datasets.kaggle import KaggleDataset
from dataset_format_benchmark.datasets.utils import adjust_image


class WikiArtDataset(KaggleDataset):
    DATASET_NAME = 'ipythonx/wikiart-gangogh-creating-art-gan'
    BYTES_PER_VALUE = 16 / 8
    DATASET_DIR_NAME = 'wikiart'
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        if self.METADATA_FILE_NAME:
            self.metadata_file_path = Path(self.dataset_path, self.METADATA_FILE_NAME)
        else:
            self.metadata_file_path = None

    def _download(self, force: bool = False):
        if force or not self.dataset_path.exists():
            super()._download(force)

    @staticmethod
    def iter_images(root: Path, min_size: Optional[int] = None) -> Generator:
        dir_names = sorted(
            file_item.name for file_item in filter(lambda i: i.is_dir() and i.name[0] != '.', root.iterdir())
        )

        for idx, dir_name in enumerate(dir_names):
            for file_item in Path(root, dir_name).iterdir():
                if file_item.is_file():
                    try:
                        with Image.open(file_item) as image:
                            if min_size and min(image.size) < min_size:
                                continue

                            yield idx, image
                    except UnidentifiedImageError:
                        pass

    def _get_min_size(self, limit_size: Optional[int] = None):
        min_size = 999999

        for _, image in self.iter_images(self.dataset_path, limit_size):
            width, height = image.size

            # Filter out images with any dimension smaller than self.IMAGE_WIDTH px
            min_size = max(limit_size, min(min_size, min(height, width)))

        return min_size

    @staticmethod
    def _resize_images(size: int):
        pass

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            min_size = self._get_min_size()
            print(f'Found minimum size: {min_size}px')

            raise NotImplementedError

            image_set = self._resize_images(min_size)

            directories = sorted(
                file_item.name
                for file_item in filter(lambda i: i.is_dir() and i.name[0] != '.', self.dataset_path.iterdir())
            )

            metadata = {
                'labels': {idx: dir_name for idx, dir_name in enumerate(directories)},
                'images': image_set
            }

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)


class DatasetWikiArtFilesystem(WikiArtDataset):
    IMAGE_DIR_NAME = '.crops'
    METADATA_FILE_NAME = 'metadata_fs.json'

    def _resize_images(self, size: Optional[int] = None) -> Mapping[str, int]:
        image_set = {}
        dir_indexes = set()

        self.image_dir_path.mkdir(exist_ok=True)

        for idx, image in self.iter_images(self.dataset_path, size):
            crop_dir_path = Path(self.image_dir_path, str(idx))
            file_name = Path(image.filename).name

            if idx not in dir_indexes:
                crop_dir_path.mkdir(exist_ok=True)
                dir_indexes.add(idx)

            image_file_path = crop_dir_path / file_name

            if image_file_path.exists():
                print(f'File exists: {image_file_path}, skipping')
                continue

            try:
                image = adjust_image(image, size, size)
            except OSError:
                print(f'Failed reading image file: {file_name}')
            else:
                image.save(image_file_path, quality=100, subsampling=0)
                image_set[f'{idx}/{file_name}'] = idx

        return image_set

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        images = metadata['images']

        self.x = tuple(images.keys())
        self.y = tuple(images.values())
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


class DatasetWikiArtZarr(WikiArtDataset):
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

                try:
                    image = adjust_image(image, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
                except OSError:
                    print(f'Failed reading image file: {filename}')
                else:
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


class DatasetWikiArtNumpyMmap(WikiArtDataset):
    DATASET_DTYPE = 'float16'
    METADATA_FILE_NAME = 'metadata_numpy_mmap.json'

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataset_file_path = Path(self.dataset_path, 'data.npy')

    def __count_images(self, min_size: Optional[int] = None) -> int:
        cnt = sum(
            int(min(image.width, image.height) >= min_size)
            for _, image in self.iter_images(self.dataset_path)
        )

        return cnt

    def _resize_images(self, size: Optional[int] = None):
        item_count = self.__count_images(size)
        print(f'Item count: {item_count}')
        dataset_shape = (item_count, 3, size, size)

        x = open_memmap(
            str(self.dataset_file_path), mode='w+', dtype=self.DATASET_DTYPE, shape=dataset_shape
        )
        y = []

        for idx, (dir_idx, image) in enumerate(self.iter_images(self.dataset_path, size)):
            file_name = Path(image.filename).name

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {dir_idx}/{file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image[:, :]
                y.append(dir_idx)

        x.flush()

        metadata = {
            'shape': (len(y), 3, size, size),
            'labels': y
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.dataset_file_path.exists() or not self.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self._get_min_size(self.IMAGE_WIDTH)
            metadata = self._resize_images(min_size)

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.data_length = self.dataset_shape[0]
        self.y = metadata['labels']

        self.x = open_memmap(
            str(self.dataset_file_path), mode='r', dtype=self.DATASET_DTYPE, shape=self.dataset_shape
        )
        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def __getitem__(self, index):
        x = self.x[index]
        x = np.array(x)
        x = torch.from_numpy(x)

        return x, self.y[index]


class DatasetWikiArtHDF5(WikiArtDataset):
    pass


class DatasetWikiArtCuPyMmap(WikiArtDataset):
    DATASET_DTYPE = 'float16'
    METADATA_FILE_NAME = 'metadata_cupy_mmap.json'

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataset_file_path = Path(self.dataset_path, 'cupy_data.npz')

    def __count_images(self, min_size: Optional[int] = None) -> int:
        cnt = sum(
            int(min(image.width, image.height) >= min_size)
            for _, image in self.iter_images(self.dataset_path)
        )

        return cnt

    def _resize_images(self, size: Optional[int] = None):
        item_count = self.__count_images(size)
        print(f'Item count: {item_count}')
        dataset_shape = (item_count, 3, size, size)

        x = open_memmap(
            str(self.dataset_file_path), mode='w+', dtype=self.DATASET_DTYPE, shape=dataset_shape
        )
        y = []

        for idx, (dir_idx, image) in enumerate(self.iter_images(self.dataset_path, size)):
            file_name = Path(image.filename).name

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {dir_idx}/{file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image[:, :]
                y.append(idx)

        x.flush()

        metadata = {
            'shape': dataset_shape,
            'labels': y
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.dataset_file_path.exists() or not self.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self._get_min_size(self.IMAGE_WIDTH)
            metadata = self._resize_images(min_size)

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.data_length = self.dataset_shape[0]
        self.y = metadata['labels']

        self.x = open_memmap(
            str(self.dataset_file_path), mode='r', dtype=self.DATASET_DTYPE, shape=self.dataset_shape
        )
        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def iter_images(self) -> Iterator:
        pass

    def __getitem__(self, index):
        x = self.x[index]
        x = np.array(x)
        x = torch.from_numpy(x)

        return x, self.y[index]
