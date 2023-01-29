dataset-format-benchmark
========================

This package runs different image format benchmarks for dataset ML tasks

Installation
------------

Make sure you have some system deps installed:

.. code:: bash

   sudo apt install pkg-config libhdf5-dev

.. code:: bash

   python3.11 -m venv venv --upgrade-deps
   source venv/bin/activate
   python -m pip install -U -r requirements_dev.txt

   # For running on Nvidia GPU:
   python -m pip install -U torch torchvision

   # For running on CPU:
   python -m pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

   # For some reason h5py fails to install Cython while it needs it
   python -m pip install -U Cython

   python setup.py develop

Running dataset format benchmark
--------------------------------

.. code:: bash

   python -m dataset_format_benchmark --data-root /path/to/datasets/
