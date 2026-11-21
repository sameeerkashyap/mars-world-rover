import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GoalHoldTests(unittest.TestCase):
    def test_replay_verifies_paired_commands_and_keeps_strict_goal_thresholds(self):
        common = module_at("verify_goal_hold_traces", ROOT / "deploy/nautilus-jepa-navigation/verify.py")
        replay = module_at("verify_goal_hold_replay", ROOT / "deploy/nautilus-goal-hold/verify.py")
        hashes = {key: "fixed" for key in ("motion_checkpoint", "runner_source", "planner_source", "motion_source", "forecast_source", "terrain_source")}
        sources = {name: "fixed" for name in ("rover_navigate.py", "rover_plan.py", "train_rover_baseline.py", "rover_forecast.py", "evaluation_terrain.py")}
        config = {"development_replay": True, "phase": "development", "terrain": "test-rocks-501",
                  "seeds": [30], "controllers": ["model", "model-hold"], "checkpoint_sha256": "fixed", "source_sha256": sources,
                  "frozen_test_plan": {"development": [{"terrain": "test-rocks-501", "seeds": [30]}], "sha256": hashes}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            trials = []
            for controller in config["controllers"]:
                rows = []
                for step in range(1, 8):
                    distance, heading = (0.3, 0.2) if step < 6 else (0.179, 0.08)
                    holding = controller == "model-hold" and step == 7
                    if step == 7:
                        distance, heading = (0.1795 if holding else 0.181), 0.05
                    rows.append({"step": step, "time_s": step * 0.2, "x_m": 1 - distance, "y_m": 0., "yaw_rad": -heading,
                                 "goal_x_m": 1., "goal_y_m": 0., "distance_m": distance, "heading_error_rad": heading,
                                 "planning_s": 0.01, "speed_command": 0., "turn_command": 0. if holding else 0.04,
                                 "native_success": step == 7, "collision": False, "forecast_ready": False, "goal_hold": holding})
                with (folder / f"seed-30-{controller}.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                trials.append({"seed": 30, "controller": controller, "initial": {"position": [0., 0., 0.],
                               "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.}, "steps": 7,
                               "verified_success": controller == "model-hold", "collision": False,
                               "path_length_m": 0.8215 if controller == "model-hold" else 0.823,
                               "final_distance_m": rows[-1]["distance_m"], "max_consecutive_stop_steps": int(controller == "model-hold"),
                               "forecast_ready_steps": 0, "forecast_scoring_calls": 0, "goal_hold_steps": int(controller == "model-hold")})
            summary = {"config": config, "trials": trials}
            (folder / "summary.json").write_text(json.dumps(summary))
            report = replay.verify(folder, common.verify_trial)
            self.assertEqual(report["controllers"]["model"]["successes"], 0)
            self.assertEqual(report["controllers"]["model-hold"]["successes"], 1)
            self.assertEqual(report["controllers"]["model-hold"]["goal_hold_steps"], 1)
            normal = {**trials[1], "seed": 31, "controller": "model", "hold_reached_goal": True}
            (folder / "seed-31-model.csv").write_bytes((folder / "seed-30-model-hold.csv").read_bytes())
            self.assertEqual(common.verify_trial(folder, normal)["goal_hold_steps"], 1)
            normal["hold_reached_goal"] = False
            with self.assertRaises(ValueError):
                common.verify_trial(folder, normal)
            summary["trials"][1]["goal_hold_steps"] = 0
            (folder / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                replay.verify(folder, common.verify_trial)

    def test_replay_deployment_freezes_all_sources_without_elapsed_cutoffs(self):
        folder = ROOT / "deploy/nautilus-goal-hold"
        deploy = module_at("deploy_goal_hold_replay", folder / "deploy.py")
        job = yaml.safe_load((folder / "job.yaml").read_text())
        pod = job["spec"]["template"]["spec"]
        for spec in (job["spec"], pod):
            self.assertNotIn("activeDeadlineSeconds", spec)
            self.assertNotIn("ttlSecondsAfterFinished", spec)
        items = next(volume["configMap"]["items"] for volume in pod["volumes"] if volume["name"] == "source")
        self.assertEqual({item["key"] for item in items}, set(deploy.SOURCES) | {"model.pt", "replay-plan.json"})
        self.assertTrue(all((ROOT / source).is_file() for source in deploy.SOURCES.values()))
        self.assertNotIn("timeout ", (folder / "run.sh").read_text())
        self.assertIn("--replay-plan", (folder / "run.sh").read_text())


if __name__ == "__main__":
    unittest.main()
