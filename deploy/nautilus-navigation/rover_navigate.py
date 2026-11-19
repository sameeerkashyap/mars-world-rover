"""Evaluate learned-model planning and scripted forward motion in Isaac Sim."""

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--terrain", choices=("debug", "mars"), default="debug")
parser.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--checkpoint", type=Path, default=Path("/opt/navigation/model.pt"))
parser.add_argument("--output", type=Path, default=Path("/data/runs/rover-navigation"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
    parser.error("Steps must be positive and seeds must be distinct")
args.enable_cameras = False
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import parse_env_cfg
import rover_envs.envs.navigation.robots

from scripts.rover_plan import plan_action
from scripts.train_rover_baseline import model_for


class NavigationSnapshot(RecorderTerm):
    def record_post_step(self):
        # This hook preserves terminal poses before Isaac automatically resets the rover.
        env = self._env
        robot = env.scene["robot"]
        position = robot.data.root_link_pos_w.torch[0]
        yaw = robot.data.heading_w.torch[0]
        command = env.command_manager.get_term("target_pose")
        goal = command.pos_command_w[0]
        heading_error = (command.heading_command_w[0] - yaw + torch.pi).remainder(2 * torch.pi) - torch.pi
        env.navigation_sample = {
            "position": position.detach().cpu().tolist(), "yaw": float(yaw),
            "goal": goal.detach().cpu().tolist(),
            "distance_m": float(torch.linalg.vector_norm(goal[:2] - position[:2])),
            "heading_error_rad": float(heading_error),
            "termination": {name: bool(env.termination_manager.get_term(name)[0])
                            for name in env.termination_manager.active_terms},
        }
        return None, None


@configclass
class NavigationRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    telemetry = RecorderTermCfg(class_type=NavigationSnapshot)


def run_trial(env, model, seed, controller, output):
    env.reset(seed=seed)
    device = env.unwrapped.device
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(torch.zeros((1, 2), device=device))
        if bool((terminated | truncated)[0]):
            raise RuntimeError("A settling step terminated the episode before control")
    initial = env.unwrapped.navigation_sample
    samples = [initial]
    path_length = 0.0
    reward_sum = 0.0
    align_heading = False
    initial_scan = obs["policy"][0, 5:].detach().cpu().numpy().copy()
    stopped_steps = 0
    result_path = output / f"seed-{seed}-{controller}"
    with result_path.with_suffix(".csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_s", "x_m", "y_m", "z_m", "yaw_rad", "goal_x_m", "goal_y_m",
                         "distance_m", "heading_error_rad", "speed_command", "turn_command", "native_success", "collision"])
        for step in range(args.steps):
            state = obs["policy"][:, :5].detach().cpu()
            distance = float(state[0, 2]) / 0.11
            heading = abs(float(state[0, 4])) * torch.pi
            if distance < 0.16:
                align_heading = True
            if distance > 0.25 or (distance > 0.18 and heading < 0.08):
                align_heading = False
            scan = obs["policy"][0, 5:].detach().cpu()
            if controller == "forward":
                action = torch.tensor([[0.65, 0.0]])
            else:
                action = plan_action(model, state, align_heading=align_heading,
                                     height_scan=scan if controller == "model" else None)
            stopped_steps += int(bool((action == 0).all()))
            if not torch.isfinite(action).all() or action[:, 0].abs().max() > 0.7 or action[:, 1].abs().max() > 0.35:
                raise RuntimeError("Controller produced an invalid action")
            obs, reward, terminated, truncated, _ = env.step(action.to(device))
            sample = env.unwrapped.navigation_sample
            if not np.allclose(sample["goal"], initial["goal"], atol=1e-5, rtol=0):
                raise RuntimeError("The goal changed inside an evaluated episode")
            path_length += float(np.linalg.norm(np.asarray(sample["position"][:2]) - samples[-1]["position"][:2]))
            reward_sum += float(reward[0])
            samples.append(sample)
            term = sample["termination"]
            writer.writerow([step + 1, (step + 1) * env.unwrapped.step_dt, *sample["position"], sample["yaw"],
                             *sample["goal"][:2], sample["distance_m"], sample["heading_error_rad"],
                             *action[0].tolist(), term["is_success"], term["collision"]])
            if step % 50 == 0:
                handle.flush()
                print(f"seed={seed} controller={controller} step={step + 1} distance={sample['distance_m']:.3f}m heading={sample['heading_error_rad']:.3f}rad", flush=True)
            if bool((terminated | truncated)[0]):
                break
    final = samples[-1]
    np.savez_compressed(str(result_path) + "-scan.npz", initial=initial_scan, last=scan.numpy())
    success = final["termination"]["is_success"]
    verified = success and not final["termination"]["collision"] and final["distance_m"] < 0.18 and abs(final["heading_error_rad"]) < 0.1
    result = {
        "seed": seed, "controller": controller, "steps": len(samples) - 1,
        "simulation_seconds": (len(samples) - 1) * env.unwrapped.step_dt,
        "initial": initial, "final": final, "native_success": success, "verified_success": verified,
        "initial_distance_m": initial["distance_m"], "final_distance_m": final["distance_m"],
        "progress_m": initial["distance_m"] - final["distance_m"],
        "min_distance_m": min(sample["distance_m"] for sample in samples),
        "path_length_m": path_length, "reward_sum": reward_sum,
        "collision": any(sample["termination"]["collision"] for sample in samples),
        "stopped_steps": stopped_steps,
    }
    if not np.isfinite([result["progress_m"], path_length, reward_sum]).all():
        raise RuntimeError("Non-finite trial result")
    result_path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "goal-motion-v1" or not checkpoint.get("use_actions"):
        raise ValueError("Navigation requires the trained goal-motion-v1 command model")
    model = model_for(checkpoint["obs_dim"], checkpoint["action_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = args.output / os.environ.get("HOSTNAME", "local")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    source_hashes = {}
    for source in (__file__, inspect.unwrap(plan_action).__code__.co_filename, model_for.__code__.co_filename):
        source = Path(source)
        source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, source_dir / source.name)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seeds[0]
    cfg.scene.set_terrain(args.terrain)
    pattern = cfg.scene.height_scanner.pattern_cfg
    if tuple(pattern.size) != (5.0, 5.0) or pattern.resolution != 0.05 or pattern.ordering != "xy":
        raise ValueError("The planner requires the calibrated 5 m, 5 cm, xy-ordered height scan")
    if cfg.scene.height_scanner.ray_alignment != "yaw":
        raise ValueError("The planner requires a yaw-aligned height scan")
    cfg.recorders = NavigationRecorderCfg()
    cfg.events.reset_joints = EventTermCfg(func=mdp.reset_joints_by_scale, mode="reset",
                                         params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})
    config = {"task": args.task, "terrain": args.terrain, "seeds": args.seeds, "max_steps": args.steps,
              "controllers": ["model", "goal-only", "forward"],
              "source_sha256": source_hashes,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "planner": {"horizon_steps": 20, "alignment_horizon_steps": 5,
                          "candidates": 79, "alignment_candidates": 9,
                          "scan_avoidance": {"size_m": 5.0, "resolution_m": 0.05,
                                             "height_threshold_m": 0.18, "edge_threshold_m": 0.08,
                                             "footprint_half_width_m": 0.7, "soft_clearance_m": 1.0,
                                             "turn_steps": [10, 20, 40, 60]},
                          "replan_each_step": True, "alignment_enter_m": 0.16, "alignment_exit_m": 0.25},
              "success_distance_m": 0.18, "success_heading_rad": 0.1, "cameras": False,
              "settling_steps": 5, "reset_joint_state": "default position and zero velocity",
              "torch": str(torch.__version__), "numpy": np.__version__}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = gym.make(args.task, cfg=cfg)
    results = []
    try:
        with torch.inference_mode():
            for seed in args.seeds:
                pair = []
                for controller in config["controllers"]:
                    result = run_trial(env, model, seed, controller, output)
                    results.append(result)
                    pair.append(result)
                for comparison in pair[1:]:
                    for field, tolerance in (("position", 0.01), ("goal", 1e-5), ("yaw", 0.01)):
                        if not np.allclose(pair[0]["initial"][field], comparison["initial"][field], atol=tolerance, rtol=0):
                            raise RuntimeError(f"Seed {seed} did not reproduce the same initial {field}")
    finally:
        env.close()
    summary = {"config": config, "trials": results, "controllers": {}}
    for controller in config["controllers"]:
        trials = [result for result in results if result["controller"] == controller]
        summary["controllers"][controller] = {
            "trials": len(trials), "native_successes": sum(result["native_success"] for result in trials),
            "verified_successes": sum(result["verified_success"] for result in trials),
            "collisions": sum(result["collision"] for result in trials),
            "mean_progress_m": float(np.mean([result["progress_m"] for result in trials])),
        }
    learned, forward = summary["controllers"]["model"], summary["controllers"]["forward"]
    summary["meaningful_result"] = (learned["verified_successes"] > 0 and
                                     learned["mean_progress_m"] > forward["mean_progress_m"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["meaningful_result"]:
        raise RuntimeError("Learned planning did not reach a verified goal and beat forward progress")
    (output / "SUCCESS").touch()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        try:
            app.close()
        finally:
            raise SystemExit(status)
