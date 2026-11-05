#!/usr/bin/env bash
# Source this file from bash or zsh; paths are relative to this script.
if [ -n "${ZSH_VERSION:-}" ]; then
    _rover_script="${(%):-%x}"
else
    _rover_script="${BASH_SOURCE[0]}"
fi
_rover_root="$(cd "$(dirname "$_rover_script")/.." && pwd)"
export JEPAWM_HOME="$_rover_root/external/jepa-wms"
export JEPAWM_DSET="$_rover_root/data"
export JEPAWM_LOGS="$_rover_root/runs"
export JEPAWM_CKPT="$_rover_root/checkpoints"
export TORCH_HOME="$_rover_root/.cache/torch"
export HF_HOME="$_rover_root/.cache/huggingface"
export UV_CACHE_DIR="$_rover_root/.cache/uv"
source "$_rover_root/.venv-jepa/bin/activate"
unset _rover_root _rover_script
