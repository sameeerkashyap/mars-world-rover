import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts/train_rover_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class RoverBaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = [Path(self.directory.name) / f"run_{i}.hdf5" for i in range(2)]
        for path in self.paths:
            with h5py.File(path, "w") as data:
                for index in range(2):
                    episode = data.create_group(f"data/demo_{index}")
                    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
                    obs = np.zeros((6, 6), dtype=np.float32)
                    obs[1:, :2] = actions[:-1]
                    obs[:, 2] = 0.5 + index + np.arange(6) * 0.01
                    obs[:, 3:5] = 0.2
                    episode["obs"] = obs
                    episode["actions"] = actions
                    episode["dones"] = [0, 0, 0, 0, 0, 1]

    def test_alignment_and_whole_recording_split(self):
        train, validation = baseline.split_episodes(self.paths[:1], self.paths[1:])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)
        self.assertTrue(all(str(self.paths[1]) in episode[0] for episode in validation))
        _, obs, actions, target = train[0]
        np.testing.assert_array_equal(obs[1], target[0])
        np.testing.assert_array_equal(actions[0], [2, 3])
        self.assertEqual(len(obs), 4)
        self.assertLess(target.max(), 100)
        train, validation = baseline.split_episodes(self.paths[:1])
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)

    def test_rejects_reset_bad_done_and_nonfinite_observation(self):
        for key, index, value in (("dones", 2, 1), ("dones", 5, 2), ("obs", (0, 0), np.nan)):
            with self.subTest(key=key, value=value):
                with h5py.File(self.paths[0], "r+") as data:
                    dataset = data[f"data/demo_0/{key}"]
                    original = dataset[index]
                    dataset[index] = value
                with self.assertRaises(ValueError):
                    baseline.load_episodes(self.paths[0])
                with h5py.File(self.paths[0], "r+") as data:
                    data[f"data/demo_0/{key}"][index] = original

    def test_rejects_overlap_and_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            baseline.split_episodes(self.paths[:1], self.paths[:1])
        with h5py.File(self.paths[1], "r+") as data:
            for episode in data["data"].values():
                del episode["obs"]
                obs = np.ones((6, 7))
                obs[1:, :2] = episode["actions"][:-1]
                episode["obs"] = obs
        with self.assertRaises(ValueError):
            baseline.split_episodes(self.paths[:1], self.paths[1:])

    def test_training_beats_copy_on_action_driven_data(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        obs = torch.zeros((32, 5))
        obs[:, 2] = 0.8
        actions = torch.rand((32, 2))
        target = obs.clone()
        target[:, 2] -= actions[:, 0] * 0.02
        target[:, 3:5] = -actions[:, 1:2] * 0.05 / torch.pi
        model = baseline.model_for(5, 2)
        copy_mse = baseline.goal_delta(obs, target).square().mean().item()
        baseline.train(model, obs, actions, target, 1000)
        self.assertLess(baseline.mse(model, obs, actions, target), copy_mse / 10)
        self.assertGreater(baseline.mse(model, obs, actions.flip(0), target), copy_mse / 10)

    def test_rollout_uses_recorded_actions_and_stays_inside_episodes(self):
        class Delta(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, -2:-1].expand(-1, 5) * 0.01

        obs = np.arange(6, dtype=np.float32)[:, None] * np.ones((1, 5), dtype=np.float32) * 0.01
        actions = np.ones((5, 2), dtype=np.float32)
        episodes = [("one", obs[:-1], actions, obs[1:]),
                    ("two", obs[:-1] + 0.1, actions, obs[1:] + 0.1)]
        report = baseline.rollout_mse(Delta(), episodes)
        self.assertAlmostEqual(report["mse"], 0, places=12)
        self.assertAlmostEqual(report["copy_mse"], 0.0025)
        self.assertEqual(report["elements"], 6)
        self.assertIsNone(baseline.rollout_mse(Delta(), episodes, horizon=6)["mse"])

    def test_angles_wrap_at_pi_and_scan_is_not_a_training_target(self):
        obs = torch.tensor([[0., 0., 0.5, 0.99, -0.99]])
        target = torch.tensor([[0., 0., 0.5, -0.99, 0.99]])
        torch.testing.assert_close(baseline.goal_delta(target, obs), torch.tensor([[0., 0.02, -0.02]]))
        episode = baseline.load_episodes(self.paths[0])[0]
        state, actions, target = baseline.tensors([episode])
        self.assertEqual(state.shape[1], 5)
        model = baseline.model_for(5, 2)
        prediction = baseline.predict(model, state, actions)
        torch.testing.assert_close(prediction[:, :2], actions)
        self.assertTrue((prediction[:, 3:5].abs() <= 1).all())

    def test_rejects_shifted_actions(self):
        with h5py.File(self.paths[0], "r+") as data:
            data["data/demo_0/actions"][1] = [100, 100]
        with self.assertRaisesRegex(ValueError, "timing"):
            baseline.load_episodes(self.paths[0])

    def test_no_action_model_ignores_both_current_and_previous_commands(self):
        model = baseline.model_for(5, 2, use_actions=False)
        inputs = torch.zeros((2, 7))
        inputs[1, :2] = 1
        inputs[1, 5:7] = -1
        torch.testing.assert_close(model(inputs)[0, 2:5], model(inputs)[1, 2:5])

    def test_relative_motion_round_trip_and_checkpoint_reload(self):
        model = baseline.model_for(5, 2)
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.zero_()
            model.net[-1].bias.copy_(torch.tensor([0.01, -0.004, 0.1]))
        obs = torch.tensor([[0., 0., 0.9, 0.99, -0.99], [0., 0., 0.5, -0.2, 0.3]])
        actions = torch.tensor([[0.4, 0.1], [0.2, -0.1]])
        target = baseline.predict(model, obs, actions)
        motion = baseline.relative_motion(obs, target)
        torch.testing.assert_close(motion, torch.tensor([[0.01, -0.004, 0.1]]).expand(2, -1), atol=1e-6, rtol=1e-5)
        path = Path(self.directory.name) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = baseline.model_for(5, 2)
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(baseline.predict(restored, obs, actions), target)
