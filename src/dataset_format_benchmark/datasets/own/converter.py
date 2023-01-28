import concurrent.futures
from functools import partial
from pathlib import Path
from typing import Sequence


def convert_to_format(raw_image: bytes, data_root_path: Path, fmt):
    pass


def convert_to_formats(raw_image: bytes, data_root_path: Path, formats: Sequence):
    convert_to_format_partial = partial(convert_to_format, raw_image, data_root_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(convert_to_format_partial, formats)
