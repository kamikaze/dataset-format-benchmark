import logging
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.utils.data
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_format_benchmark.models.inception import InceptionNet

logger = logging.getLogger(__name__)


class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        # return torch.mean(-y * torch.log(y_prim + 1e-8))
        return -torch.sum(y * torch.log(y_prim + 1e-20))


def run_epoch(data_loaders: Mapping, model, loss_func, optimizer, metric_keys, scaler=None, device=None):
    metrics_epoch = {key: [] for key in metric_keys}

    for stage, data_loader in data_loaders.items():
        for idx, (x, y) in enumerate(tqdm(data_loader)):
            x_on_device = x.to(device=device, dtype=torch.float16, non_blocking=True)
            y_on_device = y.to(device=device, dtype=torch.float16, non_blocking=True)

            with torch.cuda.amp.autocast():
                # Warning: consumes a lot of memory
                y_prim = model.forward(x_on_device)
                loss = loss_func.forward(y_on_device, y_prim)

            metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

            if stage == 'train':
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()

            # loss = loss.cpu()
            y_prim = y_prim.cpu()
            # x = x.cpu()
            # y = y.cpu()

            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)
            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

            if idx % 20 == 0:
                logger.info(f"Loss {np.mean(metrics_epoch[f'{stage}_loss'])} "
                            f"Acc {np.mean(metrics_epoch[f'{stage}_acc'])}")

    return metrics_epoch


def benchmark_dataset(dataset: Dataset, epochs, train_test_split: float, batch_size: int, learning_rate: float,
                      use_cuda: bool, worker_count: int, device=None):
    train_test_split = int(len(dataset) * train_test_split)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset,
        [train_test_split, len(dataset) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=worker_count
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=worker_count
    )

    model = InceptionNet()
    loss_func = LossCrossEntropy()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    if use_cuda:
        model = model.to(device)
        # model = model.cuda()
        loss_func = loss_func.cuda()

    metrics = {}

    data_loaders = {'train': data_loader_train, 'test': data_loader_test}

    for stage in data_loaders.keys():
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, epochs):
        metrics_epoch = run_epoch(data_loaders, model, loss_func, optimizer, metrics.keys(), scaler, device)

        for key, values in metrics_epoch.items():
            mean_value = np.mean(values)
            metrics[key].append(mean_value)

        logger.info(f'epoch: {epoch} {" ".join(f"{k} {v[-1]}" for k, v in metrics.items())}')

        plt.clf()
        plts = []
        c = 0

        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)
            plts += plt.plot(value, f'C{c}', label=key)
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        plt.tight_layout(pad=0.5)
        plt.draw()
        plt.pause(0.1)
