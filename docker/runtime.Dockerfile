FROM ghcr.io/astral-sh/uv:0.10.9 AS uv
FROM ubuntu:24.04
COPY --from=uv /uv /usr/local/bin/uv
# Only shared OS libraries/tools belong in the image. ML/simulator packages are
# installed by the init container onto /runtime, never onto the shared PVC.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bzip2 unzip build-essential cmake patchelf \
    libglib2.0-0 libgl1 libegl1 libglew-dev libosmesa6-dev libvulkan1 \
    libx11-6 libxext6 libxi6 libxrender1 libxrandr2 libxinerama1 libxcursor1 \
    libxkbcommon-x11-0 libsm6 libice6 libnss3 libasound2t64 libxt6 && \
    rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/runtime/env \
    PATH=/runtime/env/bin:/runtime/ffmpeg/bin:/runtime/tools:$PATH \
    UV_CACHE_DIR=/runtime/uv-cache \
    UV_PYTHON_INSTALL_DIR=/runtime/python \
    XDG_CACHE_HOME=/runtime/cache \
    PYTHONUNBUFFERED=1 \
    JEPAWM_HOME=/runtime \
    JEPAWM_DSET=/data/datasets \
    JEPAWM_LOGS=/data/runs/jepa \
    JEPAWM_CKPT=/data/checkpoints \
    JEPAWM_OSSCKPT=/data/checkpoints/encoders \
    TORCH_HOME=/data/cache/torch \
    HF_HOME=/data/cache/huggingface \
    ISAAC_LAB_PATH=/runtime/isaac_lab \
    ROVER_LAB_PATH=/runtime/rlroverlab \
    RLROVERLAB_PATH=/runtime/rlroverlab \
    MUJOCO_PY_MUJOCO_PATH=/runtime/mujoco/mujoco210 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LD_LIBRARY_PATH=/runtime/ffmpeg/lib:/runtime/mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY deploy/nautilus/*-run.sh deploy/nautilus/runtime-setup.sh src/step1/jepa_*.py /opt/cluster/
COPY deploy/nautilus/rover_record*.py /opt/cluster/
COPY notes/upstream.json /opt/upstream.json
WORKDIR /runtime
ENTRYPOINT ["/bin/bash"]
CMD ["/opt/cluster/jepa-run.sh", "python", "/opt/cluster/jepa_smoke.py"]
