import json
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
import torch.utils.data
from PIL import Image, UnidentifiedImageError
from numpy.lib.format import open_memmap
from torch.nn import functional

from dataset_format_benchmark.datasets.kaggle import KaggleDataset
from dataset_format_benchmark.datasets.utils import adjust_image


class FaceMasksDataset(KaggleDataset):
    DATASET_NAME = [
        'tapakah68/medical-masks-part1',
        'tapakah68/medical-masks-part2',
    ]
    BYTES_PER_VALUE = 16 / 8
    DATASET_DIR_NAME = 'masks'
    DATASET_SUBDIR_NAME = None
    DATASET_FILE_NAME = None
    IMAGE_DIR_NAME = 'images'
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    METADATA_FILE_NAME = 'metadata.json'
    RESULT_FILE_NAME = 'df.csv'
    LABELS = {
        0: 'The mask is worn correctly, covers the nose and mouth.',
        1: 'The mask covers the mouth, but does not cover the nose.',
        2: 'The mask is on, but does not cover the nose or mouth.',
        3: 'There is no mask on the face.',
    }

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        if self.DATASET_SUBDIR_NAME:
            self.dataset_subdir_path = Path(self.dataset_path, self.DATASET_SUBDIR_NAME)
        else:
            self.dataset_subdir_path = self.dataset_path

        if self.DATASET_FILE_NAME:
            self.dataset_file_path = Path(self.dataset_subdir_path, self.DATASET_FILE_NAME)
        else:
            self.dataset_file_path = None

        if self.METADATA_FILE_NAME:
            self.metadata_file_path = Path(self.dataset_subdir_path, self.METADATA_FILE_NAME)
        else:
            self.metadata_file_path = None

    def _download(self, force: bool = False):
        if force or not self.dataset_path.exists():
            super()._download(force)

            second_file_path = Path(self.dataset_path, 'df_part_2.csv')

            with open(second_file_path) as fi, open(Path(self.dataset_path, 'df.csv'), 'a') as fo:
                next(fi)
                fo.writelines(fi)

            second_file_path.unlink(missing_ok=True)

    @staticmethod
    def iter_images(root: Path, min_size: Optional[int] = None) -> Generator:
        # ID, TYPE, USER_ID, GENDER, AGE, name, size_mb

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

    def _get_min_size(self, limit_size: Optional[int] = None):
        min_size = 999999

        for _, image in self.iter_images(self.image_dir_path, limit_size):
            width, height = image.size

            # FIXME: fails when limit_size is None and it is
            # Filter out images with any dimension smaller than self.IMAGE_WIDTH px
            min_size = max(limit_size, min(min_size, min(height, width)))

        return min_size

    def _count_images(self, min_size: Optional[int] = None) -> int:
        cnt = sum(
            int(min(image.width, image.height) >= min_size)
            for _, image in self.iter_images(self.image_dir_path)
        )

        return cnt

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
                self._save_image(image, out_file_path)

                x.append(f'{label}/{file_name}')
                y.append(int(label) - 1)

        metadata = {
            'shape': (3, size, size),
            'filenames': x,
            'labels': y,
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            min_size = self._get_min_size()
            print(f'Found minimum size: {min_size}px')

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


class DatasetMasksImageFiles(FaceMasksDataset):
    IMAGE_FILE_EXTENSION = None
    DATASET_SUBDIR_NAME = 'image_files'
    METADATA_FILE_NAME = 'metadata.json'

    def _save_image(self, image, out_file_path: Path):
        image.save(out_file_path.with_suffix(self.IMAGE_FILE_EXTENSION))

    def _resize_images(self, size: Optional[int] = None):
        item_count = self._count_images(size)
        print(f'Item count: {item_count}')

        self.dataset_subdir_path.mkdir(exist_ok=True)

        x = []
        y = []

        for idx, (label, image) in enumerate(self.iter_images(self.image_dir_path, size)):
            file_name = Path(image.filename).name
            print(f'{idx}/{item_count}: Processing {file_name}')

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                label_dir_path = Path(self.dataset_subdir_path, label)
                label_dir_path.mkdir(exist_ok=True)
                out_file_path = Path(label_dir_path, file_name)
                self._save_image(image, out_file_path)

                x.append(f'{label}/{file_name}')
                y.append(int(label) - 1)

        metadata = {
            'shape': (3, size, size),
            'filenames': x,
            'labels': y,
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self._get_min_size(self.IMAGE_WIDTH)
            print(f'Found minimum size: {min_size}px')

            metadata = self._resize_images(min_size)

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.x = metadata['filenames']
        self.y = metadata['labels']
        self.data_length = len(self.y)

        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def __getitem__(self, index):
        x_path = str(Path(self.dataset_subdir_path, self.x[index]))

        with Image.open(x_path) as image:
            x = np.asarray(image)

        x = torch.from_numpy(x)

        return x, self.y[index]


class DatasetMasksNumpyZipFiles(FaceMasksDataset):
    DATASET_DTYPE = 'float16'
    DATASET_SUBDIR_NAME = 'numpy_zip_files'
    METADATA_FILE_NAME = 'metadata.json'

    def _resize_images(self, size: Optional[int] = None):
        item_count = self._count_images(size)
        print(f'Item count: {item_count}')

        self.dataset_subdir_path.mkdir(exist_ok=True)

        x = []
        y = []

        for idx, (label, image) in enumerate(self.iter_images(self.image_dir_path, size)):
            label = int(label) - 1
            file_name = Path(image.filename).name
            print(f'{idx}/{item_count}: Processing {file_name}')

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                file_name_base = file_name.split('.', maxsplit=1)[0]
                label_dir_path = Path(self.dataset_subdir_path, str(label))
                label_dir_path.mkdir(exist_ok=True)
                out_file_path = Path(label_dir_path, f'{file_name_base}.npz')
                np.savez_compressed(out_file_path, value=image)
                x.append(f'{label}/{file_name_base}.npz')
                y.append(label)

        metadata = {
            'shape': (3, size, size),
            'filenames': x,
            'labels': y
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self._get_min_size(self.IMAGE_WIDTH)
            print(f'Found minimum size: {min_size}px')

            metadata = self._resize_images(min_size)

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.x = metadata['filenames']
        self.y = metadata['labels']
        self.data_length = len(self.y)

        self.y = functional.one_hot(torch.tensor(self.y, dtype=torch.long))

    def __getitem__(self, index):
        x_path = str(Path(self.dataset_subdir_path, self.x[index]))

        with np.load(x_path) as f:
            x = torch.from_numpy(f)

        return x, self.y[index]


class DatasetMasksNumpyMmap(FaceMasksDataset):
    DATASET_DTYPE = 'float16'
    DATASET_SUBDIR_NAME = 'numpy_mmap'
    DATASET_FILE_NAME = 'data.npy'
    METADATA_FILE_NAME = 'metadata.json'

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

    def _resize_images(self, size: Optional[int] = None):
        item_count = self._count_images(size)
        print(f'Item count: {item_count}')

        dataset_shape = (item_count, 3, size, size)

        self.dataset_file_path.parent.mkdir(exist_ok=True)

        x = open_memmap(
            str(self.dataset_file_path), mode='w+', dtype=self.DATASET_DTYPE, shape=dataset_shape
        )
        y = []

        for idx, (label, image) in enumerate(self.iter_images(self.image_dir_path, size)):
            file_name = Path(image.filename).name
            print(f'{idx}/{item_count}: Processing {file_name}')

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image[:, :]
                # Decreasing by 1 due to indexes start from 1
                y.append(int(label) - 1)

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
            print(f'Found minimum size: {min_size}px')

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

#    def __len__(self):
#        return 10000

    def __getitem__(self, index):
        x = self.x[index]
        x = np.array(x)
        x = torch.from_numpy(x)

        return x, self.y[index]
