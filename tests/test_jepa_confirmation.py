import ast
import csv
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.evaluation_terrain import heldout_meshes


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "deploy/nautilus-jepa-confirmation"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfirmationTests(unittest.TestCase):
    def test_exact_paired_test_requires_success_gain_and_no_added_collision(self):
        verifier = module_at("confirmation_verify", FOLDER / "verify.py")
        success = {"verified_success": True, "collision": False, "planning_mean_s": 0.07}
        failure = {**success, "verified_success": False}
        pairs = [(failure.copy(), success.copy()) for _ in range(6)] + [(success.copy(), success.copy()) for _ in range(14)]
        result = verifier.paired_result(pairs)
        self.assertEqual(result["exact_two_sided_p"], 0.03125)
        self.assertTrue(result["promotion_passed"])
        pairs[0][1]["collision"] = True
        self.assertFalse(verifier.paired_result(pairs)["promotion_passed"])
        self.assertFalse(verifier.paired_result([(success, success)] * 20)["promotion_passed"])
        self.assertFalse(verifier.paired_result([(failure, success)] * 2 + [(success, success)] * 18)["promotion_passed"])
        with self.assertRaises(ValueError):
            verifier.paired_result(pairs[:19])

    def test_counterfactual_audit_restores_actual_forecast_mode_and_counts(self):
        path = ROOT / "deploy/nautilus-navigation/rover_navigate.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "audit_forecast_action")
        forecast = SimpleNamespace(mode="jepa", scoring_calls=10, scored_candidates=40, costs=None)
        hint = torch.tensor([1., .5])
        def choose(*args, previous_waypoint=None, **kwargs):
            self.assertEqual(forecast.mode, "transport")
            self.assertIs(previous_waypoint, hint)
            forecast.scoring_calls += 1
            forecast.scored_candidates += 4
            return torch.tensor([[0.35, 0.]])
        namespace = {"plan_action": choose, "inspect": inspect}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        action = namespace["audit_forecast_action"](forecast, None, None, None, False, True, hint)
        torch.testing.assert_close(action, torch.tensor([[0.35, 0.]]))
        torch.testing.assert_close(hint, torch.tensor([1., .5]))
        self.assertEqual((forecast.mode, forecast.scoring_calls, forecast.scored_candidates), ("jepa", 10, 40))
        def fail(*args, **kwargs):
            raise ValueError("Invalid counterfactual")
        namespace["plan_action"] = fail
        with self.assertRaises(ValueError):
            namespace["audit_forecast_action"](forecast, None, None, None, False, True)
        self.assertEqual((forecast.mode, forecast.scoring_calls, forecast.scored_candidates), ("jepa", 10, 40))

    def test_new_layouts_keep_the_old_generator_and_are_distinct(self):
        first = heldout_meshes(601)
        repeated = heldout_meshes(601)
        other = heldout_meshes(620)
        for a, b in zip(first, repeated):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[2], other[2]))
        self.assertEqual(first[2].shape, (640, 3))
        for seed in (503, 600, 621):
            with self.assertRaises(ValueError):
                heldout_meshes(seed)

    def test_protocol_has_fixed_independent_cases_and_reuses_no_deadline_job(self):
        plan = json.loads((FOLDER / "test-plan.json").read_text())
        self.assertEqual(len(plan["fresh"]), 20)
        self.assertEqual(len({case["terrain"] for case in plan["fresh"]}), 20)
        self.assertEqual(plan["controllers"], ["transport", "jepa"])
        self.assertTrue(plan["audit_forecasts"])
        self.assertTrue(plan["counterbalance_order"])
        self.assertEqual(plan["primary_test"]["two_sided_alpha"], 0.05)
        job = yaml.safe_load((ROOT / "deploy/nautilus-goal-hold/job.yaml").read_text())
        self.assertNotIn("activeDeadlineSeconds", job["spec"])
        self.assertNotIn("ttlSecondsAfterFinished", job["spec"])

    def test_all_paired_csvs_are_verified_and_a_changed_initial_scan_is_rejected(self):
        verifier = module_at("confirmation_check", FOLDER / "verify.py")
        common = module_at("confirmation_common", ROOT / "deploy/nautilus-jepa-navigation/verify.py")
        plan = json.loads((FOLDER / "test-plan.json").read_text())
        deploy = module_at("confirmation_sources", FOLDER / "deploy.py")
        source_names = {"rover_navigate.py": "runner_source", "rover_plan.py": "planner_source",
                        "train_rover_baseline.py": "motion_source", "rover_forecast.py": "forecast_source",
                        "evaluation_terrain.py": "terrain_source"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            for name, relative in deploy.SOURCES.items():
                shutil.copyfile(ROOT / relative, root / "source" / name)
                plan["all_source_sha256"][name] = hashlib.sha256((root / "source" / name).read_bytes()).hexdigest()
            for name, key in source_names.items():
                plan["sha256"][key] = plan["all_source_sha256"][name]
            (root / "source/test-plan.json").write_text(json.dumps(plan))
            hidden = root / "source/..data"
            hidden.mkdir()
            shutil.copyfile(root / "source/rover_navigate.py", hidden / "rover_navigate.py")
            last_folder = None
            for case in plan["fresh"]:
                folder = root / ("fresh-" + case["terrain"]) / "pod"
                (folder / "source").mkdir(parents=True)
                for name in source_names:
                    shutil.copyfile(root / "source" / name, folder / "source" / name)
                seed = case["seeds"][0]
                order = ["jepa", "transport"] if seed % 2 else plan["controllers"]
                trials = []
                for controller in order:
                    scored = seed != 100
                    row = {"step": 1, "time_s": 0.2, "x_m": 0.9, "y_m": 0., "yaw_rad": 0.,
                           "goal_x_m": 1., "goal_y_m": 0., "distance_m": 0.1, "heading_error_rad": 0.,
                           "planning_s": 0.07, "speed_command": 0.5, "turn_command": 0., "collision": False,
                           "native_success": True, "forecast_ready": True, "goal_hold": False,
                           "alternate_speed_command": 0.35 if scored else "", "alternate_turn_command": 0. if scored else ""}
                    with (folder / f"seed-{seed}-{controller}.csv").open("w") as handle:
                        writer = csv.DictWriter(handle, fieldnames=row)
                        writer.writeheader()
                        writer.writerow(row)
                    np.savez_compressed(folder / f"seed-{seed}-{controller}-scan.npz", initial=np.zeros(10201))
                    trials.append({"seed": seed, "controller": controller, "initial": {"position": [0., 0., 0.],
                                   "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.,
                                   "root_velocity": [0.] * 6, "joint_position": [0.] * 6,
                                   "joint_velocity": [0.] * 6}, "steps": 1,
                                   "verified_success": True, "collision": False, "path_length_m": 0.9,
                                   "final_distance_m": 0.1, "max_consecutive_stop_steps": 0, "forecast_ready_steps": 1,
                                   "hold_reached_goal": True, "goal_hold_steps": 0, "forecast_scoring_calls": int(scored),
                                   "forecast_comparison_steps": int(scored), "forecast_command_disagreements": int(scored)})
                config = {**case, "frozen_test_plan": plan, "phase": "fresh", "max_steps": 750,
                          "controllers": plan["controllers"], "checkpoint_sha256": plan["sha256"]["motion_checkpoint"],
                          "success_distance_m": 0.18, "success_heading_rad": 0.1,
                          "source_sha256": {name: hashlib.sha256((root / "source" / name).read_bytes()).hexdigest()
                                            for name in source_names}}
                (folder / "summary.json").write_text(json.dumps({"config": config, "trials": trials}))
                last_folder = folder
            with patch("builtins.print"):
                report = verifier.verify(root, common.verify_trial)
            self.assertEqual(len(report["trials"]), 40)
            self.assertFalse(report["primary"]["promotion_passed"])
            np.savez_compressed(last_folder / "seed-119-jepa-scan.npz", initial=np.ones(10201))
            with self.assertRaisesRegex(ValueError, "initial height scans differ"):
                verifier.verify(root, common.verify_trial)
            np.savez_compressed(last_folder / "seed-119-jepa-scan.npz", initial=np.zeros(10201))
            with (root / "source/rover_navigate.py").open("a") as handle:
                handle.write("\nmodified = True\n")
            with self.assertRaisesRegex(ValueError, "Frozen experiment source changed"):
                verifier.verify(root, common.verify_trial)


if __name__ == "__main__":
    unittest.main()
