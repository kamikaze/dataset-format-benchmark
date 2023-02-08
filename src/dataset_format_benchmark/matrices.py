import argparse
import time

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


def benchmark_matmul(module, array, diag, first_matrix_size: tuple[int], second_matrix_size: tuple[int],
                     repetitions: int) -> int:
    start = time.perf_counter_ns()

    for _ in range(repetitions):
        first_matrix = array(np.random.randn(*first_matrix_size).astype(dtype=np.float64))
        second_matrix = array(np.random.randn(*second_matrix_size).astype(dtype=np.float64))
        result = module.matmul(first_matrix, second_matrix)

    end = time.perf_counter_ns()

    return end - start


def benchmark_matmul_wise(module, array, first_matrix_size: tuple[int], second_matrix_size: tuple[int],
                          repetitions: int = 100) -> int:
    start = time.perf_counter_ns()

    for _ in range(repetitions):
        first_matrix = module.diag(array(np.random.rand(*first_matrix_size)))
        second_matrix = array(np.random.rand(*second_matrix_size))
        result = module.matmul(first_matrix, second_matrix)

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
        modules['cupy'] = (cp, cp.array, cp.diag)

    for (module, array, diag) in modules.values():
        test_matmul(module, array)
        test_matmul_wise(module, array, diag)

    print(f'{"Benchmark":<10}{"First matrix":<15}{"Second matrix":<15}{"Mode":<10}{"Score":<25}{"Units":<10}')

    for module_name, (module, array, diag) in modules.items():
        if len(first_matrix_size) == 1:
            duration_ns = benchmark_matmul_wise(module, array, first_matrix_size, second_matrix_size, args.repetitions)
        else:
            duration_ns = benchmark_matmul(module, array, diag, first_matrix_size, second_matrix_size, args.repetitions)

        rate = duration_ns / 1000.0 / args.repetitions

        print(f'{module_name:<10}{args.first_matrix_size:<15}{args.second_matrix_size:<15}{"avgt":<10}'
              f'{str(rate):<25}{"ms/op":<10}')
        print(f'{module_name:<10}{args.first_matrix_size:<15}{args.second_matrix_size:<15}{"thrpt":<10}'
              f'{str(1.0 / rate):<25}{"op/ms":<10}')


if __name__ == '__main__':
    main()
