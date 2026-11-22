import ast
from contextlib import ExitStack, redirect_stdout
import csv
import io
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

from scripts.train_rover_baseline import model_for
from scripts.rover_plan import plan_action


class TimeoutDiagnosisTests(unittest.TestCase):
    def test_native_runner_records_the_exact_pre_command_observation(self):
        class Environment:
            def __init__(self):
                self.unwrapped = self
                self.device = "cpu"
                self.step_dt = .2
                values = SimpleNamespace(torch=torch.zeros((1, 6)))
                self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(root_link_vel_w=values,
                              joint_pos=values, joint_vel=values))}

            def reset(self, seed):
                self.tick = 0

            def step(self, action):
                self.tick += 1
                self.navigation_sample = {"position": [.01 * self.tick, 0., 0.], "yaw": np.pi / 2,
                                          "goal": [1., 0., 0.], "distance_m": 1 - .01 * self.tick,
                                          "heading_error_rad": 0., "termination": {"is_success": False,
                                          "collision": False, "time_limit": self.tick == 7}}
                state = torch.tensor([[*action[0].tolist(), (1 - .01 * self.tick) * .11, -.5, 0.]])
                return {"policy": torch.cat((state, torch.full((1, 10201), float(self.tick))), dim=1)}, torch.zeros(1), torch.tensor([False]), torch.tensor([self.tick == 7]), {}

        source = Path(__file__).resolve().parents[1] / "deploy/nautilus-navigation/rover_navigate.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_trial")
        args = SimpleNamespace(replay_plan=None, steps=2, record_decisions=True, planner_role="corrected")
        namespace = {"args": args, "torch": torch, "np": np, "csv": csv, "json": json, "time": time,
                     "plan_action": plan_action, "h5py": h5py, "ExitStack": ExitStack, "inspect": inspect}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        model = model_for(5, 2)
        for parameter in model.parameters():
            parameter.data.zero_()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            result = namespace["run_trial"](Environment(), model, 1, "model", output)
            self.assertEqual(result["initial"]["root_velocity"], [0.] * 6)
            self.assertEqual(result["initial"]["joint_position"], [0.] * 6)
            self.assertEqual(result["initial"]["joint_velocity"], [0.] * 6)
            with h5py.File(output / "seed-1-model-decisions.h5") as recorded:
                self.assertTrue(recorded.attrs["complete"])
                self.assertEqual(recorded.attrs["steps_written"], 2)
                np.testing.assert_array_equal(recorded["policy"][:2, 5], [5., 6.])
                np.testing.assert_allclose(recorded["position_w"][:2, 0], [.05, .06])
                detail = json.loads(recorded["diagnostics"][0])
                waypoint = detail["waypoint_m"]
                np.testing.assert_allclose(detail["world_waypoint_m"], [.05 - waypoint[1], waypoint[0]], atol=1e-6)
                recorded_commands = recorded["command"][:2]
            with (output / "seed-1-model.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["x_m"]), .06)
            expected = [[float(row["speed_command"]), float(row["turn_command"])] for row in rows]
            np.testing.assert_array_equal(recorded_commands, expected)
            args.record_decisions = False
            namespace["run_trial"](Environment(), model, 2, "model", output)
            self.assertFalse((output / "seed-2-model-decisions.h5").exists())
            with (output / "seed-2-model.csv").open() as handle:
                unrecorded = list(csv.DictReader(handle))
            self.assertEqual([[row["speed_command"], row["turn_command"]] for row in unrecorded],
                             [[row["speed_command"], row["turn_command"]] for row in rows])
            def legacy(model, state, align_heading=False, height_scan=None, terrain_cost=None, hold_reached_goal=True):
                return plan_action(model, state, align_heading=align_heading, height_scan=height_scan,
                                   terrain_cost=terrain_cost, hold_reached_goal=hold_reached_goal)
            namespace["plan_action"] = legacy
            args.record_decisions = True
            namespace["run_trial"](Environment(), model, 3, "model", output)
            with h5py.File(output / "seed-3-model-decisions.h5") as recorded:
                self.assertEqual(json.loads(recorded["diagnostics"][0])["reason"], "legacy_planner")
                np.testing.assert_array_equal(recorded["command"][:2], expected)
            class ReadyForecast:
                device = torch.device("cpu")
                ready = True
                scoring_calls = scored_candidates = 0

                def reset(self, controller):
                    pass

                def observe(self, observation, previous_action):
                    pass

                def costs(self, commands, lengths):
                    self.scoring_calls += 1

            def legacy_scoring(model, state, align_heading=False, height_scan=None, terrain_cost=None, hold_reached_goal=True):
                if float(height_scan[0]) == 6.:
                    terrain_cost(None, None)
                return torch.zeros((1, 2))

            namespace["plan_action"] = legacy_scoring
            forecast = ReadyForecast()
            namespace["run_trial"](Environment(), model, 4, "transport", output, forecast)
            with h5py.File(output / "seed-4-transport-decisions.h5") as recorded:
                applied = [json.loads(row)["forecast_applied"] for row in recorded["diagnostics"][:2]]
                self.assertEqual(applied, [False, True])
            self.assertEqual(forecast.scoring_calls, 1)


    def test_cli_matches_last_scan_to_pre_command_pose_and_previous_command(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            trace = folder / "seed-1-transport.csv"
            fields = ["step", "time_s", "x_m", "y_m", "yaw_rad", "goal_x_m", "goal_y_m",
                      "distance_m", "heading_error_rad", "speed_command", "turn_command",
                      "native_success", "collision"]
            rows = [[1, .2, 0, 0, 0, 1, 0, 1, .2, .15, .12, False, False],
                    [2, .4, .1, 0, .1, 1, 0, .9, .1, 0, -.35, False, False]]
            with trace.open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            trace.with_suffix(".json").write_text(json.dumps({"steps": 2, "final": {
                "termination": {"time_limit": True, "is_success": False, "collision": False}}}))
            np.savez_compressed(folder / "seed-1-transport-scan.npz", last=np.zeros(10201, dtype=np.float32))
            model = model_for(5, 2)
            for parameter in model.parameters():
                parameter.data.zero_()
            checkpoint = folder / "model.pt"
            torch.save({"schema": "goal-motion-v1", "use_actions": True, "model": model.state_dict()}, checkpoint)
            command = [sys.executable, "-m", "scripts.diagnose_rover_timeouts", str(trace),
                       "--checkpoint", str(checkpoint)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)["trials"][0]
            self.assertEqual(report["scan_pre_command_step"], 1)
            np.testing.assert_allclose(report["state"], [.15, .12, .11, 0., .2 / np.pi])
            self.assertEqual(report["recorded_final_action"], [0., -.35])
            self.assertEqual(report["scan_only_action"], [0., 0.])
            self.assertTrue(report["route"]["goal_reachable"])
            self.assertEqual(report["candidates"]["standard"]["count"], 79)
            failed = subprocess.run(command + ["--require-success"], text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Native timeout", failed.stderr)
            rows = [[1, .2, .9, 0, 0, 1, 0, .1, 0, 0, 0, False, False],
                    [2, .4, .9, 0, 0, 1, 0, .1, 0, 0, 0, True, False]]
            with trace.open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            trace.with_suffix(".json").write_text(json.dumps({"steps": 2, "final": {
                "termination": {"time_limit": False, "is_success": True, "collision": False}}}))
            passed = subprocess.run(command + ["--require-success"], text=True, capture_output=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            reached = json.loads(passed.stdout)["trials"][0]
            self.assertTrue(reached["verified_success"])
            self.assertEqual(reached["planning_mode"], "hold")
            self.assertEqual(reached["candidates"], {})


if __name__ == "__main__":
    unittest.main()
