"""Freeze one paired navigation experiment on twenty new obstacle layouts."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
NAME = "mars-rover-jepa-confirmation-20261122-v1"
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
    "setup.sh": "deploy/nautilus-jepa-navigation/setup.sh",
    "run.sh": "deploy/nautilus-jepa-confirmation/run.sh",
    "verify_navigation.py": "deploy/nautilus-jepa-navigation/verify.py",
    "verify_confirmation.py": "deploy/nautilus-jepa-confirmation/verify.py",
    "meta_vit.py": "external/jepa-wms/app/plan_common/models/vit.py",
    "upstream.json": "notes/upstream.json",
}


def make_plan(checkpoint):
    plan = json.loads((ROOT / "deploy/nautilus-jepa-navigation/test-plan.json").read_text())
    files = {name: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for name, relative in SOURCES.items()}
    plan["sha256"].update({
        "motion_checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "runner_source": files["rover_navigate.py"], "planner_source": files["rover_plan.py"],
        "motion_source": files["train_rover_baseline.py"], "forecast_source": files["rover_forecast.py"],
        "terrain_source": files["evaluation_terrain.py"], "predictor_source": files["meta_vit.py"],
        "jepa_source": files["train_rover_jepa.py"], "readout_source": files["terrain_cost.py"],
    })
    if plan["sha256"]["motion_checkpoint"] != "76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283":
        raise ValueError("Use the original frozen navigation motion checkpoint")
    plan.update({"schema": "rover-jepa-confirmation-v1", "development": [],
                 "fresh": [{"terrain": f"test-rocks-{601 + index}", "seeds": [100 + index]} for index in range(20)],
                 "controllers": ["transport", "jepa"], "counterbalance_order": True, "audit_forecasts": True,
                 "all_source_sha256": files,
                 "primary_test": {"pairs": 20, "minimum_success_gain": 0.1, "two_sided_alpha": 0.05,
                                  "maximum_mean_planning_s": 0.2, "no_added_collision_per_pair": True},
                 "scope": "Fixed twenty-layout paired success test; no retraining or post-outcome tuning"})
    plan["fresh_layout"]["seeds"] = list(range(601, 621))
    plan["fresh_layout"].pop("rock_vertices_sha256", None)
    plan.pop("development_plan_sha256", None)
    plan.pop("retry_reason", None)
    return plan


def main():
    checkpoint = Path(sys.argv[1])
    plan = make_plan(checkpoint)
    frozen = Path(__file__).with_name("test-plan.json")
    if json.loads(frozen.read_text()) != plan:
        raise ValueError("Source differs from the committed preregistration; do not change it after launch")
    existing = subprocess.run([*KUBE, "get", "job", NAME, "--ignore-not-found", "-o", "name"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if existing:
        raise RuntimeError(f"{NAME} already exists; inspect its evidence instead of replacing it")
    job = yaml.safe_load((ROOT / "deploy/nautilus-goal-hold/job.yaml").read_text())
    job["metadata"]["name"] = NAME
    config = job["spec"]["template"]["spec"]["volumes"][-1]["configMap"]
    config["name"] = NAME + "-source"
    paths = {row["key"]: row["path"] for row in config["items"]}
    paths.update({"meta_vit.py": "meta_vit.py", "verify_confirmation.py": "verify_confirmation.py",
                  "test-plan.json": "test-plan.json"})
    config["items"] = [{"key": name, "path": paths[name]} for name in [*SOURCES, "model.pt", "test-plan.json"]]
    manifest = json.dumps(job)
    subprocess.run([*KUBE, "apply", "--dry-run=server", "-f", "-"], input=manifest, text=True, check=True)
    sources = [f"--from-file={name}={ROOT / relative}" for name, relative in SOURCES.items()]
    subprocess.run([*KUBE, "create", "configmap", NAME + "-source", *sources,
                    f"--from-file=model.pt={checkpoint}", f"--from-file=test-plan.json={frozen}"], check=True)
    subprocess.run([*KUBE, "patch", "configmap", NAME + "-source", "--type=merge", "-p", '{"immutable":true}'], check=True)
    subprocess.run([*KUBE, "create", "-f", "-"], input=manifest, text=True, check=True)


if __name__ == "__main__":
    main()
