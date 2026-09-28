#!/usr/bin/env bash
# CPU box, any datacenter: build a causal-conv1d wheel for torch 2.14+cu126 (no prebuilt one exists; releases stop at
# torch 2.10) for sm 8.0/8.6/8.9/9.0, and upload it to a private HF repo (HF_TOKEN, HF_REPO) under wheels/.
# Logs on port 8888; WHEEL_UPLOADED or <STEP>_FAILED. Delete the pod afterwards.
mkdir -p /workspace/out && cd /workspace
(setsid python3 -m http.server 8888 --directory /workspace/out > /dev/null 2>&1 &)
exec > /workspace/out/build.log 2>&1
fail() { echo "$1"; sleep infinity; }
set -x
nproc; python3 --version
export DEBIAN_FRONTEND=noninteractive
cd /tmp && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -qq && apt-get install -y -qq cuda-minimal-build-12-6 \
  libcublas-dev-12-6 libcusparse-dev-12-6 libcusolver-dev-12-6 libcurand-dev-12-6 > /dev/null || fail CUDA_FAILED
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q --upgrade pip wheel ninja packaging \
  && pip install -q torch==2.14.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126 || fail TORCH_FAILED
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=$(nproc) CAUSAL_CONV1D_FORCE_BUILD=TRUE
mkdir -p /workspace/wheels && pip wheel --no-build-isolation --no-deps -w /workspace/wheels causal-conv1d || fail BUILD_FAILED
ls -la /workspace/wheels
set +x  # HF_TOKEN stays out of the log
pip install -q "huggingface_hub>=0.34" && python3 - <<'PY' || fail UPLOAD_FAILED
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for path in glob.glob("/workspace/wheels/causal_conv1d-*.whl"):
    api.upload_file(path_or_fileobj=path, path_in_repo="wheels/" + os.path.basename(path), repo_id=os.environ["HF_REPO"])
    print("uploaded", os.path.basename(path))
PY
echo WHEEL_UPLOADED
sleep infinity
