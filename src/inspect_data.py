"""Load one PointMaze sample through the upstream dataset reader."""
import json
import os
from pathlib import Path

import torch

from app.plan_common.datasets.point_maze_dset import PointMazeDataset


def check_sample(observations, actions, states):
    visual = observations["visual"]
    if len(actions) < 2 or not (len(visual) == len(actions) == len(states)):
        raise ValueError("PointMaze sample has missing or misaligned frames")
    for tensor in (visual, observations["proprio"], actions, states):
        if not torch.isfinite(tensor).all():
            raise ValueError("PointMaze sample contains non-finite values")
    return {"frames": len(actions), "visual_shape": list(visual.shape),
            "action_shape": list(actions.shape), "state_shape": list(states.shape)}


def main():
    data_path = Path(os.environ["JEPAWM_DSET"]) / "point_maze"
    dataset = PointMazeDataset(data_path=data_path, n_rollout=1)
    observations, actions, states, _, _ = dataset[0]
    report = check_sample(observations, actions, states)
    report.update(dataset="facebook/jepa-wms:point_maze", path=str(data_path), episode=0)
    output = Path(os.environ["JEPAWM_LOGS"]) / os.environ["HOSTNAME"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
