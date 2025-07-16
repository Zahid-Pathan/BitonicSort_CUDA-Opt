# Bitonic sort GPU accelerated using CUDA C++ toolkit

## Installation of CUDA C++ tool kit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.de
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-4
nvcc --version
