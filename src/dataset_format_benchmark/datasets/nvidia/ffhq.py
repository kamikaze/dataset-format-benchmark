import json
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from dataset_format_benchmark.datasets import BaseDataset
from dataset_format_benchmark.datasets.nvidia import downloader
from dataset_format_benchmark.datasets.utils import adjust_image


class NvidiaFFHQDataset(BaseDataset):
    DATASET_NAME = [
        'tapakah68/medical-masks-part1',
        'tapakah68/medical-masks-part2',
    ]
    BYTES_PER_VALUE = 16 / 8
    DATASET_DIR_NAME = 'nvidia_ffhq'
    DATASET_SUBDIR_NAME = None
    DATASET_FILE_NAME = None
    IMAGE_DIR_NAME = 'images'
    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 1024
    RESULT_FILE_NAME = 'df.csv'

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
            tasks = ('json', 'stats', 'images',)
            downloader.run(tasks)

    @staticmethod
    def iter_images(root: Path, min_size: Optional[int] = None):
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
