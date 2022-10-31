# dataset-format-benchmark
This package runs different image format benchmarks for dataset ML tasks

## Installation

```bash
python3.10 -m venv venv --upgrade-deps
source venv/bin/activate
python -m pip install -U -r requirements_dev.txt

# For running on Nvidia GPU:
python -m pip install -U torch torchvision

# For running on CPU:
python -m pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

python setup.py develop
```

## Running dataset format benchmark
```bash
python -m dataset_format_benchmark --data-root /path/to/datasets/
```
