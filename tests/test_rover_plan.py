import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.rover_plan import plan_action, scan_costmap, scan_waypoint
from scripts.train_rover_baseline import GoalDynamics, predict, relative_motion


class RoverPlannerTests(unittest.TestCase):
    def test_captured_reverse_arrival_preserves_heading_during_acknowledgement(self):
        folder = Path(__file__).parent / "fixtures"
        model = GoalDynamics()
        with np.load(folder / "arrival-613-112.npz") as weights:
            model.load_state_dict({key: torch.from_numpy(weights[key]) for key in model.state_dict()})
        with np.load(folder / "arrival-heading-501-30.npz") as capture:
            state = torch.from_numpy(capture["policy"][-1:, :5])
            scan = torch.from_numpy(capture["policy"][-1, 5:])
            recorded = torch.from_numpy(capture["command"][-1:])
            endpoint = json.loads(str(capture["csv_rows_json"]))[-1]
        terminal = state.new_tensor([[*recorded[0], float(endpoint["distance_m"]) * .11,
                                      (math.atan2(float(endpoint["goal_y_m"]) - float(endpoint["y_m"]),
                                                  float(endpoint["goal_x_m"]) - float(endpoint["x_m"]))
                                       - float(endpoint["yaw_rad"])) / math.pi,
                                      float(endpoint["heading_error_rad"]) / math.pi]])
        model.eval()

        class CapturedArrival(GoalDynamics):
            def motion(self, commands):
                previous = commands.clone()
                previous[:, 2:] = recorded
                return relative_motion(state, terminal) + model.motion(commands) - model.motion(previous)

        with torch.inference_mode():
            response = CapturedArrival()
            reproduced = predict(response, state, recorded)
            self.assertAlmostEqual(float(reproduced[0, 4]) * math.pi, .10174131393432617, places=6)
            action = plan_action(model, state, height_scan=scan)
            next_state = predict(response, state, action)
        self.assertLess(float(next_state[0, 2]) / .11, .18)
        self.assertLess(abs(float(next_state[0, 4])) * math.pi, .1)

    def test_captured_611_reachable_waypoint_prevents_turn_reversal(self):
        folder = Path(__file__).parent / "fixtures"
        model = GoalDynamics()
        with np.load(folder / "arrival-613-112.npz") as weights:
            model.load_state_dict({key: torch.from_numpy(weights[key]) for key in model.state_dict()})
        model.eval()
        with np.load(folder / "waypoint-611-110.npz") as capture:
            policy = torch.from_numpy(capture["policy"])
            delta = capture["world_waypoint_m"][0] - capture["position_w"][1, :2]
            c, s = np.cos(capture["yaw_rad"][1]), np.sin(capture["yaw_rad"][1])
            previous = policy.new_tensor([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
            before = plan_action(model, policy[:1, :5], height_scan=policy[0, 5:])
            diagnostics = {}
            baseline = plan_action(model, policy[1:, :5], height_scan=policy[1, 5:], diagnostics=diagnostics)
            torch.testing.assert_close(before[0], torch.from_numpy(capture["command"][0]))
            torch.testing.assert_close(baseline[0], torch.from_numpy(capture["command"][1]))
        self.assertLess(float(before[0, 1] * baseline[0, 1]), 0)
        retained = {}
        actual = plan_action(model, policy[1:, :5], height_scan=policy[1, 5:], diagnostics=retained,
                             previous_waypoint=previous)
        self.assertGreater(float(before[0, 1] * actual[0, 1]), 0)
        torch.testing.assert_close(policy.new_tensor(retained["waypoint_m"]), previous)
        self.assertEqual(retained["candidate_count"], diagnostics["candidate_count"])
        self.assertEqual(retained["safe_count"], diagnostics["safe_count"])
        self.assertLess(retained["selected_measured_risk"], 100)

    def test_previous_waypoint_requires_current_fine_cell_and_graph_reachability(self):
        cost = torch.zeros((101, 101))
        goal = torch.tensor([9., 0.])
        previous = torch.tensor([1.05, .05])
        torch.testing.assert_close(scan_waypoint(cost, goal, previous), previous)
        for hint in ([2.500001, 0.], [float("nan"), 0.], [float("inf"), 0.], [.6, 0.],
                     [0., 0.], [1.], [[1., 0.]], ["invalid", 0.]):
            with self.subTest(hint=hint):
                torch.testing.assert_close(scan_waypoint(cost, goal, hint), scan_waypoint(cost, goal))
        fine_blocked = cost.clone()
        fine_blocked[51, 71] = 100
        disconnected = cost.clone()
        disconnected[:, 60] = 100
        for changed in (fine_blocked, disconnected):
            torch.testing.assert_close(scan_waypoint(changed, goal, previous), scan_waypoint(changed, goal))
        for nearby in (torch.tensor([1.2, 0.]), torch.tensor([.5, 0.])):
            torch.testing.assert_close(scan_waypoint(cost, nearby, previous), scan_waypoint(cost, nearby))
        self.assertIsNone(scan_waypoint(torch.full_like(cost, 100), goal, previous))

    def test_waypoint_hint_keeps_precision_modes_and_unsafe_stop_unchanged(self):
        model = GoalDynamics()
        scan = torch.zeros(10201)
        previous = torch.tensor([-1., 1.])
        for distance, heading, align in ((1.2, .4, False), (.15, .4, False), (9., .4, True),
                                         (.179, .01, False)):
            state = torch.tensor([[0., 0., distance * .11, .2, heading]])
            torch.testing.assert_close(plan_action(model, state, height_scan=scan, align_heading=align,
                                                   previous_waypoint=previous),
                                       plan_action(model, state, height_scan=scan, align_heading=align))

        class OutsideModel(GoalDynamics):
            def motion(self, commands):
                return commands.new_tensor([10., 0., 0.]).expand(len(commands), -1)

        for measured, reason in ((scan, "no_safe_candidate"), (torch.full_like(scan, float("nan")), "start_blocked")):
            diagnostics = {}
            action = plan_action(OutsideModel(), torch.tensor([[0., 0., .99, 0., 0.]]),
                                 height_scan=measured, previous_waypoint=previous, diagnostics=diagnostics)
            torch.testing.assert_close(action, torch.zeros((1, 2)))
            self.assertEqual(diagnostics["reason"], reason)

    def test_optional_decision_diagnostics_preserve_action_and_forecast_calls(self):
        class IdealModel(GoalDynamics):
            def motion(self, commands):
                return torch.stack((commands[:, 2] * .11 * .2, torch.zeros_like(commands[:, 2]),
                                    commands[:, 3] * .2), dim=1)

        model = IdealModel()
        scan = torch.zeros(10201)
        state = torch.tensor([[0., 0., .99, .1, .2]])
        calls = []
        def forecast(commands, lengths):
            calls.append((commands.clone(), lengths.clone()))
            return lengths.new_zeros(len(lengths), dtype=torch.float32)
        expected = plan_action(model, state, height_scan=scan, terrain_cost=forecast)
        diagnostics = {"stale": True}
        actual = plan_action(model, state, height_scan=scan, terrain_cost=forecast, diagnostics=diagnostics)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        for a, b in zip(calls[0], calls[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertNotIn("stale", diagnostics)
        self.assertEqual(diagnostics["reason"], "selected")
        self.assertEqual(diagnostics["candidate_count"], 79)
        self.assertGreater(diagnostics["safe_count"], 0)
        self.assertTrue(diagnostics["forecast_applied"])
        self.assertIsNotNone(diagnostics["waypoint_m"])
        self.assertIn(diagnostics["turn_steps"], (0, 10, 20, 40, 60))
        self.assertGreaterEqual(diagnostics["top_cost_gap"], 0)
        centering = torch.tensor([[.15, 0., .179 * .11, 0., .01]])
        plan_action(model, centering, height_scan=scan, terrain_cost=forecast, diagnostics=diagnostics)
        self.assertEqual(len(calls), 2)
        self.assertEqual(diagnostics["forecast_reason"], "reached_goal")
        for current, measured, reason in ((torch.tensor([[0., 0., .01, 0., 0.]]), scan, "goal_hold"),
                                          (state, torch.full_like(scan, float("nan")), "start_blocked")):
            plan_action(model, current, height_scan=measured, diagnostics=diagnostics)
            self.assertEqual(diagnostics["reason"], reason)
            self.assertFalse(diagnostics["forecast_applied"])
            self.assertEqual(diagnostics["candidate_count"], 0)

    def test_captured_arrivals_stay_inside_goal_during_acknowledgement(self):
        with np.load(Path(__file__).parent / "fixtures/arrival-613-112.npz") as capture:
            translating = torch.from_numpy(capture["states"])
            turning = torch.from_numpy(capture["prior_states"])
            scan = torch.from_numpy(capture["scan"])
            model = GoalDynamics()
            model.load_state_dict({key: torch.from_numpy(capture[key]) for key in model.state_dict()})
        model.eval()
        cases = ((translating, scan, False, 0.1802847683429718),
                 (translating, scan, True, 0.1802847683429718),
                 (turning, None, True, 0.1783006489276886))
        scoring_calls = []

        def terrain_cost(commands, lengths):
            scoring_calls.append(len(commands))
            return torch.zeros_like(lengths, dtype=torch.float32)

        for states, measured_scan, align_heading, stopped_distance in cases:
            scoring_calls.clear()
            with self.subTest(stopped_distance=stopped_distance, align_heading=align_heading):
                state, terminal = states.split(1)

                class CapturedArrival(GoalDynamics):
                    def motion(self, commands):
                        zero = commands.clone()
                        zero[:, 2:] = 0
                        return relative_motion(state, terminal) + model.motion(commands) - model.motion(zero)

                with torch.inference_mode():
                    response = CapturedArrival()
                    stopped = predict(response, state, torch.zeros((1, 2)))
                    self.assertAlmostEqual(float(stopped[0, 2]) / 0.11, stopped_distance, places=7)
                    action = plan_action(model, state, height_scan=measured_scan, align_heading=align_heading,
                                         terrain_cost=terrain_cost if measured_scan is not None else None,
                                         previous_waypoint=torch.tensor([-1., 1.]))
                    next_state = predict(response, state, action)
                self.assertEqual(scoring_calls, [])
                self.assertLess(float(next_state[0, 2]) / 0.11, 0.18)
                self.assertLess(abs(float(next_state[0, 4])) * math.pi, 0.1)
                if states is turning:
                    torch.testing.assert_close(action, torch.zeros((1, 2)))
        stopped = plan_action(model, translating[:1], height_scan=torch.full((10201,), float("nan")))
        torch.testing.assert_close(stopped, torch.zeros((1, 2)))

    def test_local_route_avoids_a_blocked_straight_path(self):
        cost = torch.zeros((101, 101))
        cost[35:66, 60:80] = 100
        goal = torch.tensor([9., 0.])
        waypoint = scan_waypoint(cost, goal)
        self.assertGreater(float(waypoint[1].abs()), 0.5)
        index = ((waypoint + 2.5) / 0.05).round().long()
        self.assertLess(float(cost[index[1], index[0]]), 100)
        self.assertIsNone(scan_waypoint(torch.full_like(cost, 100), goal))

    def test_scan_calibration_preserves_slopes_and_marks_rocks_and_unknown(self):
        axis = torch.linspace(-2.5, 2.5, 101)
        scan = 9.0 - 0.1 * axis[None, :] - 0.15 * axis[:, None]
        cost = scan_costmap(scan.flatten())
        self.assertEqual(float(cost[50, 50]), 0)
        scan[48:53, 72:77] -= 0.4
        cost = scan_costmap(scan.flatten())
        self.assertGreaterEqual(float(cost[50, 74]), 100)
        self.assertGreaterEqual(float(cost[50, 62]), 100)
        self.assertLess(float(cost[25, 50]), 100)
        scan[20, 20] = float("nan")
        self.assertGreaterEqual(float(scan_costmap(scan.flatten())[20, 20]), 100)
        with self.assertRaises(ValueError):
            scan_costmap(torch.zeros(100))

    def test_scan_blocks_forward_rock_and_unknown_scan_stops(self):
        class IdealModel(GoalDynamics):
            def motion(self, commands):
                return torch.stack((commands[:, 2] * 0.11 * 0.2, torch.zeros_like(commands[:, 2]),
                                    commands[:, 3] * 0.2), dim=1)

        model = IdealModel()
        state = torch.tensor([[0., 0., 0.99, 0., 0.]])
        scan = torch.zeros((101, 101))
        scan[47:54, 71:78] = -0.4
        action = plan_action(model, state, height_scan=scan.flatten())
        self.assertEqual(float(action[0, 0]), 0)
        self.assertGreater(float(action[0, 1].abs()), 0)
        stopped = plan_action(model, state, height_scan=torch.full((10201,), float("nan")))
        torch.testing.assert_close(stopped, torch.zeros((1, 2)))

    def test_closed_loop_reaches_position_and_heading(self):
        class IdealModel(GoalDynamics):
            def motion(self, commands):
                return torch.stack((commands[:, 2] * 0.11 * 0.2, torch.zeros_like(commands[:, 2]),
                                    commands[:, 3] * 0.2), dim=1)

        model = IdealModel()
        state = torch.tensor([[0., 0., 0.22, 0.6, -0.4]])
        with torch.inference_mode():
            for _ in range(400):
                state = predict(model, state, plan_action(model, state))
                if state[0, 2] / 0.11 < 0.18 and state[0, 4].abs() * torch.pi < 0.1:
                    break
        self.assertLess(float(state[0, 2]) / 0.11, 0.18)
        self.assertLess(float(state[0, 4].abs()) * torch.pi, 0.1)

    def test_uses_model_predictions_to_turn_toward_goal(self):
        class TurningModel(torch.nn.Module):
            def forward(self, inputs):
                obs, action = inputs[:, :5], inputs[:, 5:7]
                delta = torch.zeros_like(obs)
                delta[:, :2] = action - obs[:, :2]
                delta[:, 3] = -action[:, 1] * 0.02
                delta[:, 4] = -action[:, 1] * 0.02
                return delta

        state = torch.tensor([[0., 0., 0.99, 0.6, 0.]])
        action = plan_action(TurningModel(), state)
        self.assertGreater(float(action[0, 1]), 0)
        self.assertLessEqual(float(action[0, 0].abs()), 0.7)
        self.assertLessEqual(float(action[0, 1].abs()), 0.35 + 1e-7)
        self.assertEqual(float(state[0, 3]), float(torch.tensor(0.6)))
        near_goal = torch.tensor([[0., 0., 0.015, 0., 0.6]])
        action = plan_action(TurningModel(), near_goal, align_heading=True)
        self.assertEqual(float(action[0, 0]), 0)
        self.assertGreater(float(action[0, 1]), 0)

    def test_copy_model_selects_stop_and_invalid_states_are_rejected(self):
        class CopyModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros_like(inputs[:, :5])

        state = torch.tensor([[0., 0., 0.99, 0., 0.]])
        torch.testing.assert_close(plan_action(CopyModel(), state), torch.zeros((1, 2)))
        with self.assertRaises(ValueError):
            plan_action(CopyModel(), state.repeat(2, 1))
        state[0, 2] = float("nan")
        with self.assertRaises(ValueError):
            plan_action(CopyModel(), state)

    def test_goal_hold_waits_only_inside_both_native_tolerances(self):
        class PlanningReached(torch.nn.Module):
            def forward(self, inputs):
                raise RuntimeError("Candidate planning reached")

        state = torch.tensor([[0., -0.23, 0.17944564 * 0.11, 0.7249015033576081, -0.08632588 / torch.pi]])
        scan = torch.zeros(10201)
        action = plan_action(PlanningReached(), state, height_scan=scan, hold_reached_goal=True)
        torch.testing.assert_close(action, torch.zeros((1, 2)))
        torch.testing.assert_close(plan_action(PlanningReached(), state, height_scan=scan), action)
        torch.testing.assert_close(plan_action(PlanningReached(), state, height_scan=scan, align_heading=True), action)
        inside = state.clone()
        inside[0, 0], inside[0, 2] = 0.05, 0.15 * 0.11
        torch.testing.assert_close(plan_action(PlanningReached(), inside, height_scan=scan), action)
        with self.assertRaisesRegex(RuntimeError, "Candidate planning reached"):
            plan_action(PlanningReached(), state, height_scan=scan, hold_reached_goal=False)
        for distance, heading in ((0.181, 0.08), (0.179, 0.101)):
            outside = state.clone()
            outside[0, 2], outside[0, 4] = distance * 0.11, heading / torch.pi
            with self.assertRaisesRegex(RuntimeError, "Candidate planning reached"):
                plan_action(PlanningReached(), outside, height_scan=scan, hold_reached_goal=True)
        with self.assertRaises(ValueError):
            plan_action(PlanningReached(), state, height_scan=torch.zeros(10), hold_reached_goal=True)
