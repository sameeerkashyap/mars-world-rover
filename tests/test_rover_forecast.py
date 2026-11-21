import csv
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.rover_plan import plan_action
from scripts.train_rover_baseline import GoalDynamics
from src.evaluation_terrain import heldout_meshes
from src.rover_forecast import TerrainForecast
from src.terrain_cost import TerrainCost


class Encoder(torch.nn.Module):
    def forward_features(self, images):
        return {"x_norm_patchtokens": F.avg_pool2d(images[:, :1], 14).flatten(2).transpose(1, 2).expand(-1, -1, 384)}


class NoMotion(torch.nn.Module):
    def motion(self, commands):
        return torch.zeros(len(commands), 3)


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_prior = NoMotion()

    def forward(self, features, states, actions):
        correction = actions[:, -1].mean(dim=-1)[:, None, None]
        return features[:, -1] + correction, states[:, -1, 2:]


class IdealMotion(GoalDynamics):
    def motion(self, commands):
        return torch.stack((commands[:, 2] * 0.11 * 0.2, torch.zeros_like(commands[:, 2]), commands[:, 3] * 0.2), dim=1)


class RoverForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def forecast(self):
        head = TerrainCost()
        with torch.no_grad():
            head.readout.weight.fill_(1 / 64)
            head.readout.bias.zero_()
        return TerrainForecast(Encoder(), Predictor(), head,
                               {"mean": torch.zeros(384), "scale": torch.ones(384),
                                "projection": torch.eye(384)[:, :64], "latent_scale": torch.ones(64)}, "cpu")

    def test_history_timing_reset_and_action_conditioned_costs(self):
        forecast = self.forecast()
        observation = torch.zeros(10206)
        forecast.observe(observation)
        self.assertFalse(forecast.ready)
        command = torch.tensor([0.2, 0.0])
        observation[:2] = command
        for _ in range(5):
            forecast.observe(observation, command)
        self.assertTrue(forecast.ready)
        commands = torch.zeros(2, 25, 2)
        commands[1, :, 0] = 0.4
        lengths = torch.tensor([20, 80])
        learned = forecast.costs(commands, lengths)
        self.assertGreater(float(learned[1]), float(learned[0]))
        forecast.mode = "transport"
        transported = forecast.costs(commands, lengths)
        torch.testing.assert_close(transported[0], transported[1])
        self.assertTrue(all(not parameter.requires_grad for parameter in forecast.readout.parameters()))
        with self.assertRaises(ValueError):
            forecast.observe(observation, torch.zeros(2))
        observation[5] = float("nan")
        forecast.observe(observation, command)
        self.assertFalse(forecast.ready)
        forecast.reset()
        with self.assertRaises(ValueError):
            forecast.costs(commands, lengths)

    def test_forecast_reranks_safe_paths_but_cannot_override_scan_rejection(self):
        model = IdealMotion()
        state = torch.tensor([[0., 0., 0.15, 0., 0.]])
        scan = torch.zeros(10201)
        original = plan_action(model, state, height_scan=scan)
        def cost(commands, lengths):
            self.assertEqual(commands.shape[1:], (25, 2))
            return (commands[:, 0] == original[0]).all(dim=1).float()
        changed = plan_action(model, state, height_scan=scan, terrain_cost=cost)
        self.assertFalse(torch.equal(changed, original))
        def forbidden(commands, lengths):
            self.fail("Forecast called on an entirely blocked scan")
        stopped = plan_action(model, state, height_scan=torch.full((10201,), float("nan")), terrain_cost=forbidden)
        torch.testing.assert_close(stopped, torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            plan_action(model, state, terrain_cost=cost)
        with self.assertRaises(RuntimeError):
            plan_action(model, state, height_scan=scan, terrain_cost=lambda commands, lengths: torch.full_like(lengths, float("nan"), dtype=torch.float32))

    def test_fresh_layouts_are_deterministic_distinct_and_have_valid_meshes(self):
        first, repeated, second = heldout_meshes(501), heldout_meshes(501), heldout_meshes(502)
        for original, copy in zip(first, repeated):
            np.testing.assert_array_equal(original, copy)
        self.assertFalse(np.array_equal(first[2], second[2]))
        for vertices, faces in ((first[0], first[1]), (first[2], first[3])):
            self.assertTrue(np.isfinite(vertices).all())
            self.assertGreaterEqual(faces.min(), 0)
            self.assertLess(faces.max(), len(vertices))
        normal = np.cross(first[0][first[1][0, 1]] - first[0][first[1][0, 0]],
                          first[0][first[1][0, 2]] - first[0][first[1][0, 0]])
        self.assertGreater(normal[2], 0)
        with self.assertRaises(ValueError):
            heldout_meshes(503)

    def test_fresh_ground_supports_the_upstream_25_metre_spawn_border(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "external/RLRoverLab/rover_envs/envs/navigation/utils/terrains/terrain_utils.py"
        module = ast.parse(source.read_text())
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TerrainManager")
        sampler = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "random_rover_spawns")
        namespace = {"np": np, "Optional": Optional}
        exec(compile(ast.Module(body=[sampler], type_ignores=[]), str(source), "exec"), namespace)
        ground = heldout_meshes(501)[0]
        width = float(np.ptp(ground[:, 0])) - 2
        cells = int(round(width / 0.05)) + 1
        heightmap = np.zeros((cells, cells), dtype=np.float32)
        geometry = SimpleNamespace(min_x=float(ground[:, 0].min()) + 1, min_y=float(ground[:, 1].min()) + 1)
        context = SimpleNamespace(resolution_in_m=0.05, _heightmap_manager=geometry)
        locations = namespace["random_rover_spawns"](context, np.zeros_like(heightmap), heightmap,
                                                    n_spawns=100, border_offset=25., seed=12345)
        self.assertEqual(locations.shape, (100, 3))
        self.assertTrue(((locations[:, :2] >= 0) & (locations[:, :2] <= 30)).all())

    def test_trace_verification_catches_false_success_and_manifest_preserves_source(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "deploy/nautilus-jepa-navigation"
        spec = importlib.util.spec_from_file_location("verify_navigation", folder / "verify.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed-4-model.csv"
            row = {"step": 1, "time_s": 0.2, "x_m": 0.9, "y_m": 0., "yaw_rad": 0.,
                   "goal_x_m": 1., "goal_y_m": 0., "distance_m": 0.1, "heading_error_rad": 0.,
                   "planning_s": 0.01, "speed_command": 0.5, "turn_command": 0.,
                   "collision": False, "native_success": True, "forecast_ready": False}
            with path.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            trial = {"seed": 4, "controller": "model", "initial": {"position": [0., 0., 0.],
                     "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.}, "steps": 1,
                     "verified_success": True, "collision": False, "path_length_m": 0.9,
                     "final_distance_m": 0.1, "max_consecutive_stop_steps": 0, "forecast_ready_steps": 0,
                     "forecast_scoring_calls": 0}
            self.assertTrue(verifier.verify_trial(path.parent, trial)["verified_success"])
            trial["verified_success"] = False
            with self.assertRaises(ValueError):
                verifier.verify_trial(path.parent, trial)
        job = yaml.safe_load((folder / "job.yaml").read_text())
        self.assertNotIn("activeDeadlineSeconds", job["spec"])
        self.assertNotIn("ttlSecondsAfterFinished", job["spec"])
        self.assertNotIn("timeout ", (folder / "run.sh").read_text())
        plan = json.loads((folder / "test-plan.json").read_text())
        self.assertEqual(plan["controllers"], ["model", "transport", "jepa"])

    def test_retry_reuses_development_only_with_identical_controllers(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "deploy/nautilus-jepa-navigation"
        spec = importlib.util.spec_from_file_location("verify_retry", folder / "verify.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        original = json.loads((folder / "test-plan-v1.json").read_text())
        corrected = json.loads((folder / "test-plan.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            fresh_paths = []
            for phase, plan in (("development", original), ("fresh", corrected)):
                for case in plan[phase]:
                    output = destination / f"{phase}-{case['terrain']}" / "pod"
                    output.mkdir(parents=True)
                    trials = []
                    for seed in case["seeds"]:
                        for controller in plan["controllers"]:
                            row = {"step": 1, "time_s": 0.2, "x_m": 0.9, "y_m": 0., "yaw_rad": 0.,
                                   "goal_x_m": 1., "goal_y_m": 0., "distance_m": 0.1, "heading_error_rad": 0.,
                                   "planning_s": 0.01, "speed_command": 0.5, "turn_command": 0.,
                                   "collision": False, "native_success": True, "forecast_ready": False}
                            with (output / f"seed-{seed}-{controller}.csv").open("w") as handle:
                                writer = csv.DictWriter(handle, fieldnames=row)
                                writer.writeheader()
                                writer.writerow(row)
                            trials.append({"seed": seed, "controller": controller, "initial": {
                                "position": [0., 0., 0.], "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.},
                                "steps": 1, "verified_success": True, "collision": False, "path_length_m": 0.9,
                                "final_distance_m": 0.1, "max_consecutive_stop_steps": 0,
                                "forecast_ready_steps": 0, "forecast_scoring_calls": 0})
                    summary = {"config": {**case, "phase": phase, "frozen_test_plan": plan,
                                          "controllers": plan["controllers"]}, "trials": trials}
                    path = output / "summary.json"
                    path.write_text(json.dumps(summary))
                    if phase == "fresh":
                        fresh_paths.append(path)
            with patch("builtins.print"):
                report = verifier.verify(destination)
            self.assertEqual(len(report["trials"]), 36)
            for path in fresh_paths:
                summary = json.loads(path.read_text())
                summary["config"]["frozen_test_plan"]["sha256"]["planner_source"] = "changed"
                path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verifier.verify(destination)

    def test_native_goal_timing_diagnostic_preserves_strict_failure(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("verify_goal_timing", root / "deploy/nautilus-jepa-navigation/verify.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed-30-model.csv"
            rows = []
            for step, distance, heading, success in ((1, 0.179, 0.08, False), (2, 0.181, 0.05, True)):
                rows.append({"step": step, "time_s": step * 0.2, "x_m": 1 - distance, "y_m": 0.,
                             "yaw_rad": -heading, "goal_x_m": 1., "goal_y_m": 0., "distance_m": distance,
                             "heading_error_rad": heading, "planning_s": 0.01, "speed_command": 0.,
                             "turn_command": 0.04, "collision": False, "native_success": success, "forecast_ready": False})
            with path.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            trial = {"seed": 30, "controller": "model", "initial": {"position": [0., 0., 0.],
                     "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.}, "steps": 2,
                     "verified_success": False, "collision": False, "path_length_m": 0.823,
                     "final_distance_m": 0.181, "max_consecutive_stop_steps": 0, "forecast_ready_steps": 0,
                     "forecast_scoring_calls": 0}
            report = verifier.verify_trial(path.parent, trial)
            self.assertFalse(report["verified_success"])
            self.assertEqual(report["geometric_goal_steps"], 1)
            self.assertEqual(report["native_geometry_disagreement_steps"], 2)


if __name__ == "__main__":
    unittest.main()
