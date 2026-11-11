#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
    jepa|data) python_version=3.10 ;;
    rover)
        [[ "${ACCEPT_EULA:-}" == Y ]] || { echo 'Isaac requires ACCEPT_EULA=Y' >&2; exit 2; }
        python_version=3.12 ;;
    *) echo 'Usage: runtime-setup.sh jepa|rover|data' >&2; exit 2 ;;
esac
run_dir="/data/runs/setup/${HOSTNAME}"
mkdir -p "$run_dir" /runtime
exec > >(tee "$run_dir/install.log") 2>&1
uv venv --python "$python_version" --seed "$VIRTUAL_ENV"

fetch_repo() {
    local url="$1" revision="$2" directory="$3"
    git init "$directory"
    git -C "$directory" remote add origin "$url"
    git -C "$directory" fetch --depth 1 origin "$revision"
    git -C "$directory" checkout --detach FETCH_HEAD
    test "$(git -C "$directory" rev-parse HEAD)" = "$revision"
}
pin() {
    python -c 'import json,sys; print(json.load(open("/opt/upstream.json"))[sys.argv[1]]["commit"])' "$1"
}

if [[ "$mode" == data ]]; then
    fetch_repo https://github.com/facebookresearch/jepa-wms.git "$(pin jepa-wms)" /runtime/jepa-wms
    uv pip install huggingface-hub
elif [[ "$mode" == jepa ]]; then
    fetch_repo https://github.com/facebookresearch/jepa-wms.git "$(pin jepa-wms)" /runtime/jepa-wms
    # Conda-forge supplies FFmpeg 7's shared libraries; Python packages use uv.
    mkdir -p /runtime/micromamba
    curl --retry 3 -fL https://micro.mamba.pm/api/micromamba/linux-64/2.3.2 -o /runtime/micromamba.tar.bz2
    tar -xjf /runtime/micromamba.tar.bz2 -C /runtime/micromamba bin/micromamba
    MAMBA_ROOT_PREFIX=/runtime/mamba /runtime/micromamba/bin/micromamba create -y -p /runtime/ffmpeg -c conda-forge ffmpeg=7
    MAMBA_ROOT_PREFIX=/runtime/mamba /runtime/micromamba/bin/micromamba clean --all --yes
    rm /runtime/micromamba.tar.bz2
    cd /runtime/jepa-wms
    # jepa-wms leaves metaworld and datasets unpinned. Two upstream drifts break
    # a fresh resolve, so override both:
    #  - metaworld's source branch `master` was renamed to `main` by Farama; pin
    #    a commit (see notes/upstream.json) that still supports Python 3.10.
    #  - unpinned `datasets` resolves to 2.14.4 (dragged down by a very new
    #    fsspec); that release calls the removed `pyarrow.PyExtensionType` and
    #    crashes against the modern pyarrow/numpy the rest of the stack locks in.
    #    Force a datasets line built for pyarrow>=15 and numpy 2.
    { printf 'metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@%s\n' "$(pin metaworld)";
      printf 'datasets>=3.0.0,<4.0.0\n'; } > /runtime/overrides.txt
    uv pip install -e . 'torchcodec==0.3.0' --override /runtime/overrides.txt
    mkdir -p /runtime/mujoco
    curl --retry 3 -fL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -o /runtime/mujoco.tgz
    tar -xzf /runtime/mujoco.tgz -C /runtime/mujoco
    rm /runtime/mujoco.tgz
    python setup_macros.py
    python -c 'import torch; import torchcodec'
    # The pointmaze model's DinoEncoder pulls the dinov2_vits14 backbone from
    # GitHub via torch.hub at run time, but only this setup container can reach
    # github.com (the GPU/main container cannot resolve it). Pre-warm the
    # persistent torch.hub cache (TORCH_HOME on /data) so planning loads offline.
    python - <<'PY'
import torch
torch.hub._validate_not_a_forked_repo = lambda *a, **k: True
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
print("prefetched dinov2_vits14 into", torch.hub.get_dir())
PY
else
    # Isaac Lab needs LFS; install on the AMD64 node, avoiding its apt postinst
    # during cross-platform image builds on Apple Silicon.
    mkdir -p /runtime/tools
    curl --retry 3 -fL https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz -o /runtime/git-lfs.tgz
    tar -xzf /runtime/git-lfs.tgz -C /runtime/tools --strip-components=1 git-lfs-3.7.1/git-lfs
    rm /runtime/git-lfs.tgz
    git lfs install --skip-repo
    # Supported upstream native installation, instead of the large Isaac image.
    uv pip install 'isaacsim[all,extscache]==6.0.1.0' \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match --prerelease=allow
    fetch_repo https://github.com/isaac-sim/IsaacLab.git ffff603eafc6b74264a5261cc0183d6a65390d78 "$ISAAC_LAB_PATH"
    git -C "$ISAAC_LAB_PATH" fetch --depth 1 origin tag v3.0.0-beta2.patch1
    test "$(git -C "$ISAAC_LAB_PATH" rev-parse 'v3.0.0-beta2.patch1^{commit}')" = ffff603eafc6b74264a5261cc0183d6a65390d78
    cd "$ISAAC_LAB_PATH"
    ./isaaclab.sh --install 'rl[skrl],visualizer[kit]'
    fetch_repo https://github.com/abmoRobotics/RLRoverLab.git "$(pin RLRoverLab)" "$ROVER_LAB_PATH"
    cd "$ROVER_LAB_PATH"
    uv pip install -e .
    uv pip install 'gdown==6.5.1'
    python -m gdown '1VXFTD2OgHcsQL_ifO81AzD2HDkA98h93' --timeout 60 --retries 3 -O /runtime/terrain_assets.zip
    python -c 'import zipfile; z=zipfile.ZipFile("/runtime/terrain_assets.zip"); bad=z.testzip(); assert bad is None, f"Corrupt terrain archive member: {bad}"'
    python -m zipfile -e /runtime/terrain_assets.zip .
    rm /runtime/terrain_assets.zip
    test -s rover_envs/assets/terrains/mars/terrain1/terrain_only.usd
    ln -s /data/runs/rover/training logs
fi
uv pip freeze > "$run_dir/packages.txt"
cp /opt/upstream.json "$run_dir/"
# No package caches need to remain after installation; keep the runtime small.
uv cache clean
touch "/runtime/ready-$mode"
