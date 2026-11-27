"""Read terrain costs from the frozen rover model's spatial features."""

import torch
from torch import nn
from torch.nn import functional as F

from scripts.rover_plan import scan_costmap


class TerrainCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(64, 1)

    def forward(self, features):
        if features.shape[-2:] != (64, 64) or not torch.isfinite(features).all():
            raise ValueError("Terrain costs require finite 64-token, 64-channel features")
        return self.readout(features).squeeze(-1).sigmoid()


# shortcut: height rules label geometric clearance, use measured traversability before safety claims.
def terrain_targets(scans):
    """Return the blocked fraction of each encoder patch using measured scans."""
    if scans.ndim != 2 or scans.shape[1] != 10201 or len(scans) == 0:
        raise ValueError("Terrain targets require nonempty 101 by 101 scans")
    blocked = torch.stack([(scan_costmap(scan) >= 100).float() for scan in scans])[:, None]
    resized = F.interpolate(blocked, (112, 112), mode="bilinear", align_corners=False)
    return F.avg_pool2d(resized, 14).flatten(1)
