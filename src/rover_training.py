"""Disk feature caches, episode-local windows, and resumable training state."""

from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def cached_episodes(paths, encoder, device, directory, signature, encode, digest):
    directory = Path(directory)
    result = []
    for path in paths:
        identity = {"schema": 1, "recording": digest(path), "encoder": signature}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = directory / f"{key}.pt"
        if not target.exists():
            print(f"Encoding {path}", flush=True)
            episodes = encode([path], encoder, device)
            atomic_save({"identity": identity, "episodes": episodes}, target)
        else:
            print(f"Feature cache hit: {path}", flush=True)
        cached = torch.load(target, weights_only=True, mmap=True, map_location="cpu")
        if cached["identity"] != identity or not cached["episodes"]:
            raise ValueError(f"Feature cache identity mismatch: {target}")
        for name, features, states, actions in cached["episodes"]:
            if (features.shape[1:] != (64, 384) or states.shape != (len(features), 6)
                    or actions.shape != (len(features) - 1, 10) or len(features) < 7
                    or not all(torch.isfinite(value).all() for value in (features, states, actions))):
                raise ValueError(f"Invalid cached episode: {name}")
        result.extend(cached["episodes"])
    return result


def feature_statistics(episodes, project=True):
    total = torch.zeros(384, dtype=torch.float64)
    cross = torch.zeros(384, 384, dtype=torch.float64) if project else None
    patch_total = torch.zeros(64, 384, dtype=torch.float64)
    patch_square = torch.zeros_like(patch_total)
    frames = 0
    for _, features, _, _ in episodes:
        for chunk in features.split(32):
            chunk = chunk.double()
            rows = chunk.flatten(0, 1)
            total += rows.sum(dim=0)
            if project:
                cross += rows.T @ rows
            patch_total += chunk.sum(dim=0)
            patch_square += chunk.square().sum(dim=0)
            frames += len(chunk)
    if frames < 2:
        raise ValueError("Need at least two encoded training frames")
    count = frames * 64
    mean = total / count
    variance = patch_square.sum(dim=0) / count - mean.square()
    scale = variance.clamp_min(0).sqrt().clamp_min(0.1)
    patch_variance = (patch_square / frames - (patch_total / frames).square()).clamp_min(0)
    feature_std = patch_variance.sqrt().mean().item()
    if feature_std < 0.01:
        raise ValueError("Encoder features have insufficient variation")
    if not project:
        return mean.float(), scale.float(), None, None, feature_std, 1.
    covariance = cross / count - mean[:, None] * mean[None, :]
    covariance = covariance / scale[:, None] / scale[None, :]
    eigenvalues, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    projection = vectors[:, -64:].flip(1)
    latent_scale = eigenvalues[-64:].flip(0).clamp_min(0).sqrt().clamp_min(0.1)
    retained = eigenvalues[-64:].sum().item() / eigenvalues.sum().clamp_min(1e-12).item()
    return mean.float(), scale.float(), projection.float(), latent_scale.float(), feature_std, retained


class WindowDataset:
    def __init__(self, episodes, mean, scale, projection, latent_scale):
        self.episodes = episodes
        self.mean, self.scale = mean, scale
        self.projection, self.latent_scale = projection, latent_scale
        self.ends = []
        for _, features, states, actions in episodes:
            if states.shape != (len(features), 6) or actions.shape != (len(features) - 1, 10):
                raise ValueError("Invalid window episode alignment")
            self.ends.append((self.ends[-1] if self.ends else 0) + max(0, len(features) - 6))
        if not self.ends or self.ends[-1] == 0:
            raise ValueError("No complete two-context, five-future windows")

    def __len__(self):
        return self.ends[-1]

    def location(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        return self.episodes[episode], index - (self.ends[episode - 1] if episode else 0)

    def batch(self, indices):
        samples = [self.location(index) for index in indices]
        full = (torch.stack([episode[1][start:start + 7] for episode, start in samples]) - self.mean) / self.scale
        latent = full if self.projection is None else full @ self.projection / self.latent_scale
        states = torch.stack([episode[2][start:start + 7] for episode, start in samples])
        actions = torch.stack([episode[3][start:start + 6] for episode, start in samples])
        return latent, states, actions, full

    def columns(self):
        return tuple(_WindowColumn(self, field) for field in range(4))


class _WindowColumn:
    def __init__(self, dataset, field):
        self.dataset, self.field = dataset, field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            rows, columns = indices
            return self[rows][:, columns]
        if isinstance(indices, slice):
            indices = range(*indices.indices(len(self)))
        if self.field == 2:
            samples = [self.dataset.location(index) for index in indices]
            return torch.stack([episode[3][start:start + 6] for episode, start in samples])
        return self.dataset.batch(indices)[self.field]

    def clone(self):
        return self[:].clone()


def save_progress(path, model, optimizer, manifest, epoch, history, best, plateau, stale, stopped):
    numpy_state = np.random.get_state()
    atomic_save({"schema": "rover-training-state-v1", "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "manifest": manifest, "epoch": epoch,
                 "history": history, "best": best, "plateau": plateau, "stale": stale, "stopped": stopped,
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def restore_progress(path, model, optimizer, manifest):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != "rover-training-state-v1" or saved["manifest"] != manifest:
        raise ValueError("Resume requires the same data, source, hardware, and training configuration")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved
