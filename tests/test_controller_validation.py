import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.evaluation_terrain import heldout_meshes

from scripts.rover_plan import plan_action


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "deploy/nautilus-controller-validation"


def module_at(path):
    spec = importlib.util.spec_from_file_location("controller_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ControllerValidationTests(unittest.TestCase):
    def test_verifier_checks_native_truth_full_starts_meshes_and_source(self):
        tooling = module_at(FOLDER / "validation.py")
        common = module_at(ROOT / "deploy/nautilus-jepa-navigation/verify.py")
        plan = json.loads((ROOT / "deploy/nautilus-jepa-confirmation/frozen-v1-test-plan.json").read_text())
        plan.update(tooling.design("development", "jepa"))
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output = Path(directory) / "source", Path(directory) / "run"
            snapshot.mkdir()
            output.mkdir()
            files = {}
            for target, relative in tooling.FILES.items():
                path = snapshot / target
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
                files[target] = tooling.digest(path)
            original = subprocess.run(["rtk", "proxy", "git", "show", f"{tooling.ORIGINAL}:scripts/rover_plan.py"], cwd=ROOT,
                                      capture_output=True, check=True).stdout
            plan["planners"] = {}
            for role in ("original", "corrected"):
                relative = f"planners/{role}/rover_plan.py"
                path = snapshot / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(original if role == "original" else (ROOT / "scripts/rover_plan.py").read_bytes())
                files[relative] = tooling.digest(path)
                plan["planners"][role] = {"commit": tooling.ORIGINAL if role == "original" else "corrected", "path": relative, "sha256": files[relative]}
            plan["all_source_sha256"] = files
            order = []
            (snapshot / "test-plan.json").write_text(json.dumps(plan))
            (output / "test-plan.json").write_text(json.dumps(plan))
            (output / "runtime.json").write_text('{"gpu": "same-device", "stack": "pinned"}')
            for index, case in enumerate(plan["development"]):
                seed = case["seeds"][0]
                for role in plan["planner_order"][::(-1 if index % 2 else 1)]:
                    name = f"{case['terrain']}-{seed}-{role}"
                    order.append(name)
                    folder = output / name / "pod"
                    (folder / "source").mkdir(parents=True)
                    (folder / "terrain").mkdir()
                    sources = {}
                    for relative in tooling.SOURCE_KEYS:
                        if relative in ("scripts/train_rover_jepa.py", "src/terrain_cost.py"):
                            continue
                        source = snapshot / relative
                        shutil.copyfile(source, folder / "source" / source.name)
                        sources[source.name] = tooling.digest(source)
                    source = snapshot / plan["planners"][role]["path"]
                    shutil.copyfile(source, folder / "source/rover_plan.py")
                    sources["rover_plan.py"] = tooling.digest(source)
                    shutil.copyfile(output / "runtime.json", folder / "runtime.json")
                    meshes = heldout_meshes(int(case["terrain"].removeprefix("test-rocks-")))
                    np.savez(folder / "terrain/meshes.npz", **dict(zip(("ground", "faces", "rocks", "rock_faces"), meshes)))
                    np.savez(folder / f"seed-{seed}-jepa-scan.npz", initial=np.zeros(10201))
                    row = {"step": 1, "time_s": .2, "x_m": .9, "y_m": 0., "yaw_rad": 0.,
                           "goal_x_m": 1., "goal_y_m": 0., "distance_m": .1, "heading_error_rad": 0.,
                           "speed_command": .5, "turn_command": 0., "planning_s": .07,
                           "native_success": False, "collision": False, "forecast_ready": False, "goal_hold": False}
                    with (folder / f"seed-{seed}-jepa.csv").open("w") as handle:
                        writer = csv.DictWriter(handle, fieldnames=row)
                        writer.writeheader()
                        writer.writerow(row)
                    trial = {"seed": seed, "controller": "jepa", "verified_success": False, "collision": False,
                             "steps": 1, "path_length_m": .9, "final_distance_m": .1, "max_consecutive_stop_steps": 0,
                             "forecast_ready_steps": 0, "forecast_scoring_calls": 0, "hold_reached_goal": True,
                             "initial": {"position": [0., 0., 0.], "goal": [1., 0., 0.], "yaw": 0., "heading_error_rad": 0.,
                                         "root_velocity": [0.] * 6, "joint_position": [0.] * 6, "joint_velocity": [0.] * 6}}
                    config = {**case, "frozen_test_plan": plan, "planner_role": role, "controllers": ["jepa"],
                              "phase": "development", "max_steps": 750, "success_distance_m": .18, "success_heading_rad": .1,
                              "task": "AAURoverEnvSimple-v0", "torch": "pinned", "numpy": "pinned", "cameras": False,
                              "reset_joint_state": "default position and zero velocity", "settling_steps": 5, "checkpoint_sha256": plan["sha256"]["motion_checkpoint"],
                              "runtime_sha256": tooling.digest(output / "runtime.json"),
                              "planner_source_sha256": sources["rover_plan.py"], "source_sha256": sources}
                    (folder / "summary.json").write_text(json.dumps({"config": config, "trials": [trial]}))
            (output / "execution-order.json").write_text(json.dumps(order))
            report = tooling.verify(snapshot, output, common.verify_trial)
            self.assertEqual(len(report["trials"]), 4)
            self.assertTrue(all(not trial["verified_success"] for trial in report["trials"]))
            summary_path = folder / "summary.json"
            saved = summary_path.read_text()
            summary = json.loads(saved)
            summary["trials"][0]["initial"]["root_velocity"][0] = .1
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "initial root_velocity"):
                tooling.verify(snapshot, output, common.verify_trial)
            summary_path.write_text(saved)
            (folder / "source/rover_plan.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Executed source changed"):
                tooling.verify(snapshot, output, common.verify_trial)

    def test_plan_keeps_fixed_fresh_pairs_and_exact_declared_endpoint(self):
        tooling = module_at(FOLDER / "validation.py")
        plan = tooling.design("fresh", "transport")
        self.assertEqual(plan["fresh"], [{"terrain": f"test-rocks-{701 + i}", "seeds": [200 + i]} for i in range(20)])
        self.assertEqual(heldout_meshes(701)[2].shape, (640, 3))
        self.assertFalse(np.array_equal(heldout_meshes(701)[2], heldout_meshes(720)[2]))
        success = {"verified_success": True, "collision": False, "planning_mean_s": .07}
        failure = {**success, "verified_success": False}
        result = tooling.paired_result([(failure, success)] * 6 + [(success, success)] * 14)
        self.assertEqual(result["exact_two_sided_p"], .03125)
        self.assertTrue(result["primary_passed"])
        self.assertFalse(tooling.paired_result([(failure, failure)] * 20)["primary_passed"])
        with self.assertRaises(ValueError):
            tooling.paired_result([(failure, success)] * 19)

    def test_selected_planner_is_checked_before_import_and_accepts_legacy_actions(self):
        runner = ROOT / "deploy/nautilus-navigation/rover_navigate.py"
        function = next(node for node in ast.parse(runner.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "select_planner")
        namespace = {"hashlib": hashlib, "importlib": __import__("importlib"), "plan_action": plan_action}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.py"
            source.write_text("raise RuntimeError('imported unverified source')\n")
            plan = {"planners": {"original": {"sha256": "0" * 64}}}
            with self.assertRaisesRegex(ValueError, "planner source"):
                namespace["select_planner"](source, plan, "original")
            original = subprocess.run(["rtk", "proxy", "git", "show", "815f19e:scripts/rover_plan.py"], cwd=ROOT,
                                      capture_output=True, check=True).stdout
            source.write_bytes(original)
            plan["planners"]["original"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            selected = namespace["select_planner"](source, plan, "original")
            state = torch.tensor([[0., 0., 0.01, 0., 0.]])
            torch.testing.assert_close(selected(None, state, height_scan=torch.zeros(10201)), torch.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
