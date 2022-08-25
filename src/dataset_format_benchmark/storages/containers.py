import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from meeting06.datasets.utils import adjust_image
from meeting06.storages import ImageFileStorage
from numpy.lib.format import open_memmap
from torch.nn import functional


class NumpyZipImageStorage(ImageFileStorage):
    DATASET_DTYPE = 'float16'
    DATASET_SUBDIR_NAME = 'numpy_zip_files'
    METADATA_FILE_NAME = 'metadata.json'

    def _resize_images(self, size: Optional[int] = None):
        item_count = self._count_images(size)
        print(f'Item count: {item_count}')

        self.dataset_subdir_path.mkdir(exist_ok=True)

        x = []
        y = []

        for idx, (label, image) in enumerate(self._iter_images(self.image_dir_path, size)):
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


class DatasetMasksNumpyMmap(ImageFileStorage):
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

        for idx, (label, image) in enumerate(self._iter_images(self.image_dir_path, size)):
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
