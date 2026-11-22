"""Evaluate learned-model planning and scripted forward motion in Isaac Sim."""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import shutil
import traceback
import time

import h5py
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--terrain", default="debug")
parser.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--checkpoint", type=Path, default=Path("/opt/navigation/model.pt"))
parser.add_argument("--output", type=Path, default=Path("/data/runs/rover-navigation"))
parser.add_argument("--jepa-run", type=Path)
parser.add_argument("--terrain-checkpoint", type=Path)
parser.add_argument("--encoder-source", type=Path, default=Path("/runtime/dinov2"))
parser.add_argument("--predictor-source", type=Path, default=Path("/opt/navigation/meta_vit.py"))
parser.add_argument("--test-plan", type=Path)
parser.add_argument("--replay-plan", type=Path)
parser.add_argument("--planner-source", type=Path)
parser.add_argument("--planner-role", choices=("original", "corrected"))
parser.add_argument("--runtime-receipt", type=Path)
parser.add_argument("--phase", choices=("development", "fresh"), default="development")
parser.add_argument("--record-decisions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
    parser.error("Steps must be positive and seeds must be distinct")
if any((args.jepa_run, args.terrain_checkpoint, args.test_plan)) and not all((args.jepa_run, args.terrain_checkpoint, args.test_plan)):
    parser.error("Forecast evaluation requires the JEPA run, terrain checkpoint, and frozen test plan")
if args.replay_plan and args.test_plan:
    parser.error("Use either a development replay or forecast test plan")
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
from src.rover_forecast import load_forecast
from src.evaluation_terrain import heldout_meshes
from rover_envs.assets.terrains.terrain_registry import register_terrain_from_folder
from rover_envs.assets.terrains.generator.terrain_generator import save_to_usd


def select_planner(source, plan, role):
    if source is None:
        if role is not None:
            raise ValueError("A planner role requires an explicit planner source")
        return plan_action
    if plan is None or role not in plan.get("planners", {}):
        raise ValueError("An explicit planner source requires its frozen planner role")
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["planners"][role]["sha256"]:
        raise ValueError("Frozen planner source changed")
    spec = importlib.util.spec_from_file_location("selected_rover_planner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_action


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


def audit_forecast_action(forecast, model, state, scan, align_heading, hold_reached_goal):
    mode, calls, candidates = forecast.mode, forecast.scoring_calls, forecast.scored_candidates
    try:
        forecast.mode = "transport" if mode == "jepa" else "jepa"
        return plan_action(model, state, align_heading=align_heading, height_scan=scan,
                           terrain_cost=forecast.costs, hold_reached_goal=hold_reached_goal)
    finally:
        forecast.mode, forecast.scoring_calls, forecast.scored_candidates = mode, calls, candidates


def run_trial(env, model, seed, controller, output, forecast=None, audit_forecasts=False):
    supports_diagnostics = "diagnostics" in inspect.signature(plan_action).parameters
    env.reset(seed=seed)
    use_forecast = controller in ("jepa", "transport")
    hold_reached_goal = controller != "forward" and (controller == "model-hold" or not args.replay_plan)
    if use_forecast:
        forecast.reset(controller)
    device = env.unwrapped.device
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(torch.zeros((1, 2), device=device))
        if bool((terminated | truncated)[0]):
            raise RuntimeError("A settling step terminated the episode before control")
    initial = env.unwrapped.navigation_sample
    if audit_forecasts or args.planner_role:
        robot = env.unwrapped.scene["robot"]
        initial = {**initial, "root_velocity": robot.data.root_link_vel_w.torch[0].detach().cpu().tolist(),
                   "joint_position": robot.data.joint_pos.torch[0].detach().cpu().tolist(),
                   "joint_velocity": robot.data.joint_vel.torch[0].detach().cpu().tolist()}
    samples = [initial]
    path_length = 0.0
    reward_sum = 0.0
    align_heading = False
    initial_scan = obs["policy"][0, 5:].detach().cpu().numpy().copy()
    stopped_steps = 0
    consecutive_stops = longest_stop = 0
    planning_times = []
    previous_action = None
    ready_steps = 0
    goal_hold_steps = 0
    comparison_steps = command_disagreements = 0
    result_path = output / f"seed-{seed}-{controller}"
    with ExitStack() as recording, result_path.with_suffix(".csv").open("w") as handle:
        decisions = recording.enter_context(h5py.File(str(result_path) + "-decisions.h5", "x")) if args.record_decisions else None
        if decisions is not None:
            decisions.attrs.update(schema="rover-plan-decisions-v1", steps_written=0, complete=False,
                                   csv_step_offset=1, policy_fields="previous_speed,previous_turn,distance*0.11,bearing/pi,heading/pi,height_scan")
            for name, shape in (("policy", (args.steps, obs["policy"].shape[1])), ("position_w", (args.steps, 3)),
                                ("yaw_rad", (args.steps,)), ("command", (args.steps, 2))):
                decisions.create_dataset(name, shape, dtype="f4", compression="lzf")
            decisions.create_dataset("diagnostics", (args.steps,), dtype=h5py.string_dtype())
        writer = csv.writer(handle)
        writer.writerow(["step", "time_s", "x_m", "y_m", "z_m", "yaw_rad", "goal_x_m", "goal_y_m",
                         "distance_m", "heading_error_rad", "speed_command", "turn_command", "native_success", "collision", "planning_s", "forecast_ready", "goal_hold",
                         "alternate_speed_command", "alternate_turn_command"])
        for step in range(args.steps):
            state = obs["policy"][:, :5].detach().cpu()
            distance = float(state[0, 2]) / 0.11
            heading = abs(float(state[0, 4])) * torch.pi
            holding = hold_reached_goal and distance < 0.18 and heading < 0.1
            if distance < 0.16:
                align_heading = True
            if distance > 0.25 or (distance > 0.18 and heading < 0.08):
                align_heading = False
            scan = obs["policy"][0, 5:].detach().cpu()
            start_planning = time.perf_counter()
            if use_forecast:
                forecast.observe(obs["policy"][0].detach(), previous_action)
            ready = use_forecast and forecast.ready
            ready_steps += int(ready)
            calls_before = forecast.scoring_calls if use_forecast else 0
            diagnostics = {} if decisions is not None else None
            if controller == "forward":
                action = torch.tensor([[0.65, 0.0]])
                if diagnostics is not None:
                    diagnostics.update(reason="scripted_forward", waypoint_m=None, forecast_applied=False)
            else:
                action = plan_action(model, state, align_heading=align_heading,
                                     height_scan=scan if controller in ("model", "model-hold", "jepa", "transport") else None,
                                     terrain_cost=forecast.costs if ready else None,
                                     hold_reached_goal=hold_reached_goal,
                                     **({"diagnostics": diagnostics} if supports_diagnostics else {}))
                if diagnostics is not None and not supports_diagnostics:
                    diagnostics.update(reason="legacy_planner", waypoint_m=None, forecast_applied=bool(ready))
            if use_forecast and forecast.device.type == "cuda":
                torch.cuda.synchronize(forecast.device)
            planning_seconds = time.perf_counter() - start_planning
            planning_times.append(planning_seconds)
            alternate = None
            if audit_forecasts and use_forecast and forecast.scoring_calls > calls_before:
                alternate = audit_forecast_action(forecast, model, state, scan, align_heading, hold_reached_goal)
                if forecast.device.type == "cuda":
                    torch.cuda.synchronize(forecast.device)
                comparison_steps += 1
                command_disagreements += int(not torch.equal(action, alternate))
            stopped = int(bool((action == 0).all()))
            holding = holding and bool(stopped)
            goal_hold_steps += int(holding)
            stopped_steps += stopped
            consecutive_stops = consecutive_stops + 1 if stopped else 0
            longest_stop = max(longest_stop, consecutive_stops)
            if not torch.isfinite(action).all() or action[:, 0].abs().max() > 0.7 or action[:, 1].abs().max() > 0.35:
                raise RuntimeError("Controller produced an invalid action")
            if decisions is not None:
                pre = samples[-1]
                waypoint = diagnostics["waypoint_m"]
                diagnostics["world_waypoint_m"] = None
                if waypoint is not None:
                    x, y = waypoint
                    c, s = np.cos(pre["yaw"]), np.sin(pre["yaw"])
                    diagnostics["world_waypoint_m"] = [pre["position"][0] + c * x - s * y,
                                                       pre["position"][1] + s * x + c * y]
                decisions["policy"][step] = obs["policy"][0].detach().cpu().numpy()
                decisions["position_w"][step] = pre["position"]
                decisions["yaw_rad"][step] = pre["yaw"]
                decisions["command"][step] = action[0].numpy()
                decisions["diagnostics"][step] = json.dumps(diagnostics, allow_nan=False)
                decisions.attrs["steps_written"] = step + 1
            obs, reward, terminated, truncated, _ = env.step(action.to(device))
            previous_action = action[0].detach().clone()
            sample = env.unwrapped.navigation_sample
            if not np.allclose(sample["goal"], initial["goal"], atol=1e-5, rtol=0):
                raise RuntimeError("The goal changed inside an evaluated episode")
            path_length += float(np.linalg.norm(np.asarray(sample["position"][:2]) - samples[-1]["position"][:2]))
            reward_sum += float(reward[0])
            samples.append(sample)
            term = sample["termination"]
            writer.writerow([step + 1, (step + 1) * env.unwrapped.step_dt, *sample["position"], sample["yaw"],
                             *sample["goal"][:2], sample["distance_m"], sample["heading_error_rad"],
                             *action[0].tolist(), term["is_success"], term["collision"], planning_seconds, ready, holding,
                             *(alternate[0].tolist() if alternate is not None else ["", ""])])
            if step % 50 == 0:
                handle.flush()
                print(f"seed={seed} controller={controller} step={step + 1} distance={sample['distance_m']:.3f}m heading={sample['heading_error_rad']:.3f}rad", flush=True)
            if bool((terminated | truncated)[0]):
                break
        if decisions is not None:
            decisions.attrs["complete"] = True
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
        "stopped_steps": stopped_steps, "max_consecutive_stop_steps": longest_stop,
        "stalled": longest_stop >= 25, "forecast_ready_steps": ready_steps,
        "goal_hold_steps": goal_hold_steps,
        "hold_reached_goal": hold_reached_goal,
        "forecast_comparison_steps": comparison_steps,
        "forecast_command_disagreements": command_disagreements,
        "planning_mean_s": float(np.mean(planning_times)), "planning_p95_s": float(np.percentile(planning_times, 95)),
        "forecast_scoring_calls": forecast.scoring_calls if use_forecast else 0,
        "forecast_scored_candidates": forecast.scored_candidates if use_forecast else 0,
    }
    if not np.isfinite([result["progress_m"], path_length, reward_sum]).all():
        raise RuntimeError("Non-finite trial result")
    result_path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    global plan_action
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "goal-motion-v1" or not checkpoint.get("use_actions"):
        raise ValueError("Navigation requires the trained goal-motion-v1 command model")
    model = model_for(checkpoint["obs_dim"], checkpoint["action_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    plan_path = args.replay_plan or args.test_plan
    plan = json.loads(plan_path.read_text()) if plan_path else None
    plan_action = select_planner(args.planner_source, plan, args.planner_role)
    comparison = bool(plan and plan.get("schema") == "rover-controller-validation-v1")
    if comparison and (not args.planner_source or not args.runtime_receipt):
        raise ValueError("Controller validation requires an explicit planner and runtime receipt")
    forecast = None
    if plan is not None:
        if args.replay_plan and not comparison and (args.phase != "development" or plan["controllers"] != ["model", "model-hold"]):
            raise ValueError("Goal-hold replay requires the two development controllers")
        if ({"terrain": args.terrain, "seeds": args.seeds} not in plan[args.phase]
                or args.steps != plan["steps"] or (not args.replay_plan and plan["forecast_weight"] != 0.25)):
            raise ValueError("Trial cases differ from the frozen test plan")
        for name, source in (("motion_checkpoint", args.checkpoint), ("planner_source", args.planner_source or Path(inspect.unwrap(plan_action).__code__.co_filename)),
                             ("runner_source", Path(__file__)), ("forecast_source", Path(load_forecast.__code__.co_filename)),
                             ("terrain_source", Path(heldout_meshes.__code__.co_filename)),
                             ("motion_source", Path(model_for.__code__.co_filename))):
            expected = plan["planners"][args.planner_role]["sha256"] if name == "planner_source" and args.planner_role else plan["sha256"][name]
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen navigation input changed: {name}")
        if not args.replay_plan:
            if plan["controllers"] not in (["model", "transport", "jepa"], ["transport", "jepa"]) and not (comparison and plan["controllers"] in (["transport"], ["jepa"])):
                raise ValueError("Forecast plans require the paired transport and JEPA controls")
            forecast = load_forecast(args.jepa_run, args.terrain_checkpoint, args.encoder_source,
                                     args.predictor_source, plan, args.device)
    output = args.output / os.environ.get("HOSTNAME", "local")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    source_hashes = {}
    for source in (__file__, str(args.planner_source) if args.planner_source else inspect.unwrap(plan_action).__code__.co_filename, model_for.__code__.co_filename,
                   load_forecast.__code__.co_filename, heldout_meshes.__code__.co_filename):
        source = Path(source)
        source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, source_dir / source.name)
    if args.runtime_receipt:
        shutil.copyfile(args.runtime_receipt, output / "runtime.json")
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seeds[0]
    if args.terrain.startswith("test-rocks-"):
        layout_seed = int(args.terrain.removeprefix("test-rocks-"))
        if plan is None:
            raise ValueError("Rock layouts require a frozen test plan")
        layout = output / "terrain"
        layout.mkdir()
        ground, faces, rocks, rock_faces = heldout_meshes(layout_seed)
        save_to_usd(str(layout / "terrain_only.usd"), ground, faces,
                    root_prim_name="ground", mesh_prim_name="ground", collision_approximation="none")
        save_to_usd(str(layout / "rocks_merged.usd"), rocks, rock_faces,
                    root_prim_name="obstacles", mesh_prim_name="obstacles", collision_approximation="none")
        register_terrain_from_folder(args.terrain, str(layout))
        np.savez_compressed(layout / "meshes.npz", ground=ground, faces=faces, rocks=rocks, rock_faces=rock_faces)
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
              "controllers": plan["controllers"] if plan is not None else ["model", "goal-only", "forward"],
              "controller_order": {str(seed): list(reversed(plan["controllers"])) if
                                   plan.get("counterbalance_order") and seed % 2 else plan["controllers"]
                                   for seed in args.seeds} if plan else None,
              "development_replay": bool(args.replay_plan),
              "planner_role": args.planner_role,
              "planner_source_sha256": hashlib.sha256((args.planner_source or Path(inspect.unwrap(plan_action).__code__.co_filename)).read_bytes()).hexdigest(),
              "runtime_sha256": hashlib.sha256(args.runtime_receipt.read_bytes()).hexdigest() if args.runtime_receipt else None,
              "record_decisions": args.record_decisions,
              "frozen_test_plan": plan, "phase": args.phase,
              "forecast_scoring": {"weight": 0.25, "max_seconds": 5, "patches": "central 2x2 maximum", "alignment": "scan only"},
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
    if forecast is not None and abs(env.unwrapped.step_dt - 0.2) > 1e-6:
        env.close()
        raise ValueError("Forecast history requires a 0.2 second control interval")
    results = []
    try:
        with torch.inference_mode():
            for seed in args.seeds:
                pair = []
                controllers = config["controller_order"][str(seed)] if plan else config["controllers"]
                for controller in controllers:
                    result = run_trial(env, model, seed, controller, output, forecast,
                                       audit_forecasts=bool(plan and plan.get("audit_forecasts")))
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
            "stalled_trials": sum(result["stalled"] for result in trials),
            "mean_path_length_m": float(np.mean([result["path_length_m"] for result in trials])),
            "mean_planning_s": float(np.mean([result["planning_mean_s"] for result in trials])),
        }
    if forecast is None and not args.replay_plan and not comparison:
        learned, forward = summary["controllers"]["model"], summary["controllers"]["forward"]
        summary["meaningful_result"] = (learned["verified_successes"] > 0 and
                                         learned["mean_progress_m"] > forward["mean_progress_m"])
    else:
        summary["meaningful_result"] = None
        summary["evaluation_completed"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if forecast is None and not args.replay_plan and not comparison and not summary["meaningful_result"]:
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
