# Source this in every shell on the box: source env.sh
# Everything durable lives under $S1_HOME, meant to be a persistent volume (RunPod network volume: /workspace).
export S1_HOME=${S1_HOME:-/workspace/s1}
export HF_HOME=$S1_HOME/hf              # model weights and hub downloads survive new pods
export PIP_CACHE_DIR=$S1_HOME/pip-cache
mkdir -p "$S1_HOME"/{raw,data,runs}
if [ -f "$S1_HOME/venv/bin/activate" ]; then source "$S1_HOME/venv/bin/activate"; fi
