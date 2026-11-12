"""Prepare and run a small upstream PointMaze planning evaluation."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import yaml

SOURCE = Path("/runtime/jepa-wms/configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml")


def prepare_config(original, output, episodes):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = deepcopy(original)
    config.update(nodes=1, tasks_per_node=1, cpus_per_task=4,
                  folder=str(output), tag="pointmaze-smoke")
    config["meta"].update(eval_episodes=episodes, seed=0)
    config["distributed"]["distribute_multitask_eval"] = False
    config["logging"]["optional_plots"] = False
    config["model_kwargs"]["checkpoint"] = "https://dl.fbaipublicfiles.com/jepa-wms/mz_jepa-wm.pth.tar"
    config["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]["pretrain_dec_path"] = {}
    config["model_kwargs"]["data"]["loader"]["num_workers"] = 0
    config["planner"].update(decode_each_iteration=False, iterations=3, num_samples=32, num_elites=4)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    output = Path(os.environ["JEPAWM_LOGS"]) / os.environ["HOSTNAME"]
    output.mkdir(parents=True, exist_ok=True)
    config = prepare_config(yaml.safe_load(SOURCE.read_text()), output, args.episodes)
    path = output / "pointmaze.yaml"
    path.write_text(yaml.safe_dump(config))
    # Compile/load legacy MuJoCo before starting the upstream evaluator.
    subprocess.run([sys.executable, "-c", "import mujoco_py"], check=True)
    subprocess.run([sys.executable, "-m", "evals.main", "--fname", str(path), "--debug"], check=True)


if __name__ == "__main__":
    main()
