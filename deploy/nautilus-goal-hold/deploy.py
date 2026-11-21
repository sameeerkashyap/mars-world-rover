"""Freeze and launch a six-trial goal-hold replay on an inspected layout."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
NAME = "mars-rover-goal-hold-20261121-v1"
KUBE = ["rtk", "kubectl", "--context", "nautilus", "-n", "razvanlab"]
SOURCES = {
    "rover_navigate.py": "deploy/nautilus-navigation/rover_navigate.py",
    "rover_plan.py": "scripts/rover_plan.py",
    "train_rover_baseline.py": "scripts/train_rover_baseline.py",
    "train_rover_jepa.py": "scripts/train_rover_jepa.py",
    "rover_forecast.py": "src/rover_forecast.py",
    "terrain_cost.py": "src/terrain_cost.py",
    "evaluation_terrain.py": "src/evaluation_terrain.py",
    "runtime-setup.sh": "deploy/nautilus/runtime-setup.sh",
    "setup.sh": "deploy/nautilus-goal-hold/setup.sh",
    "run.sh": "deploy/nautilus-goal-hold/run.sh",
    "verify_navigation.py": "deploy/nautilus-jepa-navigation/verify.py",
    "verify_goal_hold.py": "deploy/nautilus-goal-hold/verify.py",
    "upstream.json": "notes/upstream.json",
}


def make_plan(checkpoint):
    files = {name: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for name, relative in SOURCES.items()}
    return {"development": [{"terrain": "test-rocks-501", "seeds": [30, 31, 32]}],
            "controllers": ["model", "model-hold"], "steps": 750,
            "scope": "Goal-hold development replay on an already inspected layout",
            "sha256": {"motion_checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "runner_source": files["rover_navigate.py"], "planner_source": files["rover_plan.py"],
                       "motion_source": files["train_rover_baseline.py"], "forecast_source": files["rover_forecast.py"],
                       "terrain_source": files["evaluation_terrain.py"]}, "all_source_sha256": files}


def main():
    checkpoint = Path(sys.argv[1])
    plan = make_plan(checkpoint)
    if plan["sha256"]["motion_checkpoint"] != "76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283":
        raise ValueError("Replay must use the unchanged original navigation checkpoint")
    existing = subprocess.run([*KUBE, "get", "job", NAME, "--ignore-not-found", "-o", "name"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if existing:
        raise RuntimeError(f"{NAME} already exists; inspect its evidence instead of replacing it")
    manifest = str(Path(__file__).with_name("job.yaml"))
    subprocess.run([*KUBE, "apply", "--dry-run=server", "-f", manifest], check=True)
    sources = [f"--from-file={name}={ROOT / path}" for name, path in SOURCES.items()]
    subprocess.run([*KUBE, "create", "configmap", NAME + "-source", *sources,
                    f"--from-file=model.pt={checkpoint}", f"--from-literal=replay-plan.json={json.dumps(plan)}"], check=True)
    subprocess.run([*KUBE, "patch", "configmap", NAME + "-source", "--type=merge", "-p", '{"immutable":true}'], check=True)
    subprocess.run([*KUBE, "create", "-f", manifest], check=True)


if __name__ == "__main__":
    main()
