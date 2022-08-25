from abc import ABC
from pathlib import Path

from dataset_format_benchmark.datasets import BaseDataset


class KaggleDataset(BaseDataset, ABC):
    BYTES_PER_VALUE = None
    DATASET_NAME = None
    IMAGE_DIR_NAME = None
    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None
    RESULT_FILE_NAME = None

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        if self.IMAGE_DIR_NAME:
            self.image_dir_path = Path(self.dataset_path, self.IMAGE_DIR_NAME)
        else:
            self.image_dir_path = None

        if self.RESULT_FILE_NAME:
            self.result_file_path = Path(self.dataset_path, self.RESULT_FILE_NAME)
        else:
            self.result_file_path = None

        self.transform = transform
        self.target_transform = target_transform
        self.x = None
        self.y = None

    def _download(self, force: bool = False):
        import kaggle

        kaggle.api.authenticate()

        if isinstance(self.DATASET_NAME, list):
            for dataset_name in self.DATASET_NAME:
                kaggle.api.dataset_download_files(dataset_name, path=self.dataset_path, quiet=False, unzip=True,
                                                  force=force)
        else:
            kaggle.api.dataset_download_files(self.DATASET_NAME, path=self.dataset_path, quiet=False, unzip=True,
                                              force=force)
