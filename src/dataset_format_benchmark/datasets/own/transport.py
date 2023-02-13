import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import rawpy
from rawpy._rawpy import Params

from dataset_format_benchmark.datasets import BaseDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

    def iter_images(self, root: Path) -> Generator[Path, None, None]:
        return (
            file_path
            for file_path in self.iter_files(root)
            if file_path.name.lower().endswith('.arw')
        )

    def _convert_raw(self, raw_image_path: Path):
        # with semaphore:
        for storage in self.storages:
            for color_space in storage.color_spaces:
                for bps in storage.SUPPORTED_BPS:
                    raw_image = rawpy.imread(str(raw_image_path))
                    params = Params(
                        # demosaic_algorithm=DemosaicAlgorithm.DCB, dcb_iterations=1, dcb_enhance=True,
                        median_filter_passes=0, use_camera_wb=True, output_color=color_space, output_bps=bps,
                        no_auto_bright=True
                    )
                    processed_image = np.asarray(raw_image.postprocess(params))
                    storage_dir_name = storage.IMAGE_FILE_EXTENSION
                    color_space_name = str(color_space).split('.')[-1]
                    target_dir_path = Path(raw_image_path.parent, f'.{storage_dir_name}_{bps}_{color_space_name}')

                    target_dir_path.mkdir(exist_ok=True)

                    dst_file_path = Path(target_dir_path, f'{raw_image_path.name}.{storage.IMAGE_FILE_EXTENSION}')

                    logger.info(f'Converting {str(raw_image_path)} to {storage_dir_name}')

                    storage.save_image(dst_file_path, processed_image)

                    logger.info(f'Saved converted image in: {str(dst_file_path)}')

    def _convert_raws(self, root: Path) -> Sequence[str]:
        filenames = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            # semaphore = multiprocessing.Semaphore(multiprocessing.cpu_count())

            for image_path in self.iter_images(root):
                executor.submit(self._convert_raw, image_path)
                filenames.append(image_path.name)

        return filenames

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            self.filenames = self._convert_raws(self.dataset_root_path)
