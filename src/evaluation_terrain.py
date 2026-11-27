"""Deterministic ground and rock meshes for held-out obstacle layouts."""

import numpy as np


def heldout_meshes(seed):
    if seed not in (501, 502) and not 601 <= seed <= 620:
        raise ValueError("Only preregistered layout seeds are allowed")
    random = np.random.default_rng(seed)
    # Pad the fixed rock field for the upstream 25 m spawn border.
    axis = np.arange(-25, 55.01, 0.2)
    x, y = np.meshgrid(axis, axis)
    z = 0.015 * x + 0.04 * np.sin(x / 3) * np.cos(y / 4)
    ground = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    cells = np.arange(len(axis) ** 2).reshape(len(axis), len(axis))[:-1, :-1].flatten()
    ground_faces = np.concatenate((np.stack((cells, cells + 1, cells + len(axis)), axis=1),
                                   np.stack((cells + 1, cells + len(axis) + 1, cells + len(axis)), axis=1)))
    cube = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) / 2
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    centers = random.uniform(2, 28, (80, 2))
    sizes = random.uniform(0.3, 0.65, (80, 3))
    rocks = cube[None] * sizes[:, None]
    angles = random.uniform(-np.pi, np.pi, 80)
    rx, ry = rocks[:, :, 0].copy(), rocks[:, :, 1].copy()
    rocks[:, :, 0] = rx * np.cos(angles)[:, None] - ry * np.sin(angles)[:, None]
    rocks[:, :, 1] = rx * np.sin(angles)[:, None] + ry * np.cos(angles)[:, None]
    rocks[:, :, :2] += centers[:, None]
    height = 0.015 * centers[:, 0] + 0.04 * np.sin(centers[:, 0] / 3) * np.cos(centers[:, 1] / 4)
    rocks[:, :, 2] += (height + sizes[:, 2] / 4)[:, None]
    rock_faces = faces[None] + 8 * np.arange(80)[:, None, None]
    return ground.astype(np.float32), ground_faces, rocks.reshape(-1, 3).astype(np.float32), rock_faces.reshape(-1, 3)
