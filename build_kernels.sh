#!/usr/bin/env bash
# Build causal-conv1d (Qwen3.5's short-convolution kernel) into the volume's venv. No wheel exists for this torch, and
# RunPod's base image has no nvcc, so the CUDA 12.6 compiler and headers come from NVIDIA's apt repository. Run once on
# a GPU pod (many cores); every later pod on the volume reuses the build. Arch list: A100 (8.0), H100/H200 (9.0).
set -euxo pipefail
cd "$(dirname "$0")"
source env.sh
if ! python -c "import causal_conv1d" 2>/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  cd /tmp
  curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y -qq cuda-minimal-build-12-6 libcublas-dev-12-6 libcusparse-dev-12-6 libcusolver-dev-12-6 \
    libcurand-dev-12-6
  cd - > /dev/null
  export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0;9.0"
  export MAX_JOBS=$(nproc) CAUSAL_CONV1D_FORCE_BUILD=TRUE
  pip install --no-build-isolation --no-cache-dir causal-conv1d
fi
python -c "
import torch
from causal_conv1d import causal_conv1d_fn
x, w = torch.randn(2, 64, 128, device='cuda'), torch.randn(64, 4, device='cuda')
print('causal_conv1d OK', causal_conv1d_fn(x, w).shape)"
