import argparse
import logging
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch.utils.data
from rawpy._rawpy import ColorSpace

from dataset_format_benchmark.datasets import PyTorchDataset
from dataset_format_benchmark.datasets.own.transport import OwnTransportDataset
from dataset_format_benchmark.models.inception import InceptionNet
from dataset_format_benchmark.runner import benchmark_dataset
from dataset_format_benchmark.storages.containers import NumpyZipImageStorage
from dataset_format_benchmark.storages.fs import (
    JPEGImageStorage, PNGImageStorage, BMPImageStorage, TIFFImageStorage
)

logger = logging.getLogger(__name__)

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('dark_background')


def get_parsed_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root', type=str)
    # parser.add_argument('--id', default=0, type=int)
    # parser.add_argument('--sequence_name', default='sequence', type=str)
    # parser.add_argument('--run_name', default='run', type=str)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--model', default='model_emo_VAE_v3', type=str)
    # parser.add_argument('--datasource', default='datasource_emo_v2', type=str)

    args, args_other = parser.parse_known_args()

    return args


def main():
    args = get_parsed_args()
    data_root_path = Path(args.data_root)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    worker_count = 4  # multiprocessing.cpu_count() if USE_CUDA else 0
    train_test_split = 0.8
    batch_size = 8
    learning_rate = 1e-8
    epochs = 100

    datasets: Sequence = (
        OwnTransportDataset(root=data_root_path),
        # FaceMasksDataset(root=data_root_path),
        # WikiArtDataset(root=data_root_path),
        # NvidiaFFHQDataset(root=data_root_path),
    )
    color_spaces = (
        ColorSpace.sRGB,
        # ColorSpace.Adobe,
        # ColorSpace.ACES,
        # ColorSpace.ProPhoto,
        # ColorSpace.XYZ,
        # ColorSpace.Wide
    )
    storages: Sequence = (
        JPEGImageStorage(quality=100, color_spaces=color_spaces),
        JPEGImageStorage(quality=75, color_spaces=color_spaces),
        JPEGImageStorage(quality=50, color_spaces=color_spaces),
        JPEGImageStorage(quality=25, color_spaces=color_spaces),
        JPEGImageStorage(quality=10, color_spaces=color_spaces),
        PNGImageStorage(color_spaces=color_spaces),
        BMPImageStorage(color_spaces=color_spaces),
        TIFFImageStorage(color_spaces=color_spaces),
        # WebPImageStorage(),
        NumpyZipImageStorage(color_spaces=color_spaces),
        # NumpyMmapImageStorage(),
        # CupyMmapImageStorage(),
    )
    models: Sequence = (
        InceptionNet,
        # DenseNet,
    )

    # Loading original dataset and storing in storages
    for dataset in datasets:
        dataset_class_name = dataset.__class__.__name__

        for storage in storages:
            dataset.add_storage(storage)

        logger.info(f'Loading dataset: {dataset_class_name}')
        dataset.load()

    # Testing models against all dataset storages and models
    for dataset in datasets:
        for storage in dataset.get_storages():
            for model in models:
                dataset_class_name = dataset.__class__.__name__
                model_class_name = model.__name__

                try:
                    start_time = time.perf_counter()
                    logger.info(f'{start_time}: Benchmarking dataset{dataset_class_name} with {model_class_name} model')
                    pytorch_dataset = PyTorchDataset(dataset, storage)
                    benchmark_dataset(pytorch_dataset, epochs, train_test_split, batch_size, learning_rate, use_cuda,
                                      worker_count, device)
                    end_time = time.perf_counter()
                    logger.info(f'{end_time}: Benchmark took {end_time - start_time}')
                except KeyboardInterrupt:
                    logger.info(f'Skipping {dataset_class_name} / {model_class_name}')


main()
