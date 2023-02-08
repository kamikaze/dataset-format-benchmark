import argparse
import time
from collections import deque
from functools import partial
from itertools import repeat

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import torch


def test_matmul(module, array):
    first_matrix = array(
        np.array(
            [[3., 4., 5., ],
             [3., -1., 5., ],
             [4., 5., 2., ]],
            dtype=np.float64
        )
    )
    second_matrix = array(
        np.array(
            [[8., 7., ],
             [5., 3., ],
             [4., 5., ]],
            dtype=np.float64
        )
    )
    expected_matrix = array(
        np.array(
            [[64., 58., ],
             [39., 43., ],
             [65., 53., ]],
            dtype=np.float64
        )
    )
    multiplied_matrices = module.matmul(first_matrix, second_matrix)

    if module == torch:
        assert torch.all(multiplied_matrices.eq(expected_matrix))
    else:
        assert module.array_equal(multiplied_matrices, expected_matrix)


def test_matmul_wise(module, array, diag):
    first_matrix = diag(
        array(np.array([3., 2., 4., ], dtype=np.float64))
    )
    second_matrix = array(
        np.array(
            [[8., 7., 5., ],
             [5., 3., 4., ],
             [4., 5., 6., ]],
            dtype=np.float64
        )
    )

    expected_matrix = array(
        np.array(
            [[24., 21., 15., ],
             [10., 6., 8., ],
             [16., 20., 24., ]],
            dtype=np.float64
        )
    )

    multiplied_matrices = module.matmul(first_matrix, second_matrix)

    if module == torch:
        assert torch.all(multiplied_matrices.eq(expected_matrix))
    else:
        assert module.array_equal(multiplied_matrices, expected_matrix)


def benchmark_matmul(module, array, first_matrix, second_matrix, repetitions: int) -> int:
    start = time.perf_counter_ns()

    if module == np:
        matmul_partial = partial(module.matmul, first_matrix)
        deque(map(matmul_partial, repeat(second_matrix, repetitions)))
    elif module == cp:
        for _ in range(repetitions):
            first_matrix_mod = array(first_matrix)
            second_matrix_mod = array(second_matrix)
            module.dot(first_matrix_mod, second_matrix_mod)
            module.cuda.Device().synchronize()
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        for _ in range(repetitions):
            first_matrix_mod = array(first_matrix, device=device)
            second_matrix_mod = array(second_matrix, device=device)
            module.matmul(first_matrix_mod, second_matrix_mod)

    end = time.perf_counter_ns()

    return end - start


def get_parsed_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--first-matrix-size', '-f', type=str, default='92640x547')
    parser.add_argument('--second-matrix-size', '-s', type=str, default='547x547')
    parser.add_argument('--repetitions', '-r', type=int, default=10)

    args, args_other = parser.parse_known_args()

    return args


def parse_matrix_size(size: str) -> tuple[int]:
    return tuple(map(int, size.split('x', maxsplit=2)))


def main():
    args = get_parsed_args()
    first_matrix_size = parse_matrix_size(args.first_matrix_size)
    second_matrix_size = parse_matrix_size(args.second_matrix_size)
    modules = {
        'numpy': (np, np.array, np.diag),
        'torch': (torch, torch.tensor, torch.diag),
    }

    if cp:
        mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(mempool.malloc)
        modules['cupy'] = (cp, cp.array, cp.diag)

    for (module, array, diag) in modules.values():
        test_matmul(module, array)
        test_matmul_wise(module, array, diag)

    print(f'{"Benchmark":<10}{"First matrix":<15}{"Second matrix":<15}{"Mode":<10}{"Score":<25}{"Units":<10}')

    for module_name, (module, array, diag) in modules.items():
        first_matrix = np.random.rand(*first_matrix_size).astype(dtype=np.float64)
        second_matrix = np.random.rand(*second_matrix_size).astype(dtype=np.float64)

        if len(first_matrix_size) == 1:
            first_matrix = np.diag(first_matrix)

        duration_ns = benchmark_matmul(module, array, first_matrix, second_matrix, args.repetitions)
        duration_ms = duration_ns * 0.000001
        rate = duration_ms / args.repetitions

        print(f'{module_name:<10}{args.first_matrix_size:<15}{args.second_matrix_size:<15}{"avgt":<10}'
              f'{str(rate):<25}{"ms/op":<10}')
        print(f'{module_name:<10}{args.first_matrix_size:<15}{args.second_matrix_size:<15}{"thrpt":<10}'
              f'{str(1.0 / rate):<25}{"op/ms":<10}')

    if cp:
        mempool.free_all_blocks()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


if __name__ == '__main__':
    main()
