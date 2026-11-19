import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts.train_rover_jepa import RoverJEPA, advance_goal, decode_goal, encode_state, evaluate, forecast_loss, goal_error, prior_motion, quality_passed, rollout, scan_images, sequence_arrays, update_plateau, warp_features, windows
from scripts.train_rover_baseline import model_for


ROOT = Path(__file__).resolve().parents[1]


class RoverJEPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_evaluation_refuses_failed_development_before_loading_data(self):
        from scripts.evaluate_rover_jepa import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training").mkdir()
            (root / "training/metrics.json").write_text(json.dumps({"quality_passed": False}))
            with patch("sys.argv", ["evaluate", "--run", str(root), "--data", "missing20", "missing21",
                                    "--output", str(root / "final")]):
                with self.assertRaisesRegex(ValueError, "development quality target"):
                    main()
            self.assertFalse((root / "final").exists())

    def test_scan_scale_and_invalid_values(self):
        images = scan_images(torch.zeros(2, 10201))
        self.assertEqual(images.shape, (2, 3, 112, 112))
        self.assertTrue(torch.isfinite(images).all())
        self.assertFalse(torch.equal(images, scan_images(torch.ones(2, 10201))))
        with self.assertRaises(ValueError):
            scan_images(torch.full((1, 10201), float("nan")))

    def test_sequences_keep_action_timing_and_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rover.hdf5"
            with h5py.File(path, "w") as data:
                for episode in range(2):
                    group = data.create_group(f"data/demo_{episode}")
                    obs = np.zeros((42, 10206), np.float32)
                    actions = np.stack((np.arange(42), -np.arange(42)), axis=1).astype(np.float32)
                    obs[1:, :2] = actions[:-1]
                    obs[:, 5:] = np.arange(42)[:, None] + 100 * episode
                    group["obs"], group["actions"] = obs, actions
                    group["dones"] = np.zeros(42)
            sequences = sequence_arrays(path)
            self.assertEqual(len(sequences), 2)
            for episode, (_, obs, actions) in enumerate(sequences):
                np.testing.assert_array_equal(obs[:, 5], np.arange(1, 42, 5) + 100 * episode)
                np.testing.assert_array_equal(actions[0], np.array([[i, -i] for i in range(1, 6)]).ravel())

    def test_windows_do_not_cross_recordings(self):
        episodes = [(str(i), torch.full((8, 64, 384), float(i)), torch.zeros(8, 6), torch.zeros(7, 10))
                    for i in (0, 1)]
        features, states, actions, full_features = windows(episodes, torch.zeros(384), torch.ones(384))
        self.assertEqual(features.shape, (4, 7, 64, 384))
        self.assertTrue(all(torch.unique(sample).numel() == 1 for sample in features))

    def test_meta_predictor_trains_and_rollout_does_not_read_future_features(self):
        torch.manual_seed(0)
        prior = model_for(5, 2).eval().requires_grad_(False)
        model = RoverJEPA(ROOT / "external/jepa-wms/app/plan_common/models/vit.py", motion_prior=prior).eval()
        features = torch.randn(2, 7, 64, 64)
        state, actions = torch.zeros(2, 7, 6), torch.rand(2, 6, 10)
        state[..., 4] = 1
        prediction = rollout(model, features, state, actions, 5)
        changed = features.clone()
        changed[:, 2:] = 1000
        torch.testing.assert_close(prediction, rollout(model, changed, state, actions, 5))
        changed_state = state.clone()
        changed_state[:, 2:] = 1000
        torch.testing.assert_close(prediction, rollout(model, features, changed_state, actions, 5))
        self.assertFalse(torch.allclose(prediction[0], rollout(model, features, state, -actions, 5)[0]))
        (prediction[0].square().mean() + prediction[1].square().mean()).backward()
        self.assertIsNotNone(model.action_encoder.weight.grad)
        self.assertIsNotNone(model.readout.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in prior.parameters()))

    def test_quality_needs_prediction_gain_and_action_use(self):
        scores = {f"{prefix}{name}_step_mse": value for name in ("one", "five")
                  for prefix, value in (("", 0.8), ("copy_", 1.), ("shuffled_", 0.9), ("transport_", 0.95))}
        scores.update({"goal_" + key: value for key, value in list(scores.items())})
        self.assertTrue(quality_passed(scores))
        scores["shuffled_five_step_mse"] = 0.8
        self.assertFalse(quality_passed(scores))

    def test_projection_keeps_the_original_full_feature_scoring(self):
        class Persist(torch.nn.Module):
            def forward(self, features, states, actions):
                return features[:, -1], states[:, -1, 2:]

        projection = torch.eye(384)[:, :64]
        full_features = torch.randn(3, 7, 64, 384)
        dataset = (full_features @ projection, torch.zeros(3, 7, 6), torch.zeros(3, 6, 10), full_features)
        scores = evaluate(Persist(), dataset, "cpu", projection, torch.ones(64))
        self.assertAlmostEqual(scores["one_step_mse"], scores["copy_one_step_mse"])
        self.assertAlmostEqual(scores["five_step_mse"], scores["copy_five_step_mse"])
        self.assertFalse(quality_passed(scores))

    def test_pose_encoding_preserves_goal_geometry_and_angle_wrap(self):
        obs = torch.tensor([[0., 0., 0.9, 0.99, -0.99], [0., 0., 0.9, -0.99, 0.99]])
        encoded = encode_state(obs)
        torch.testing.assert_close(decode_goal(encoded), obs[:, 2:])
        self.assertLess(goal_error(encoded[:1], encoded[1:]).item(), 0.001)

    def test_transport_direction_and_goal_rotation(self):
        field = torch.arange(8, dtype=torch.float32).repeat(8, 1).reshape(1, 64, 1)
        torch.testing.assert_close(warp_features(field, torch.zeros(1, 3)), field)
        shifted = warp_features(field, torch.tensor([[0.11 * 5.05 / 8, 0., 0.]]))
        torch.testing.assert_close(shifted.reshape(8, 8)[:, :-1], field.reshape(8, 8)[:, 1:])
        state = torch.tensor([[0., 0., 1., 0., 1., 0.]])
        rotated = advance_goal(state, torch.tensor([[0., 0., torch.pi / 2]]))
        torch.testing.assert_close(rotated, torch.tensor([[0., -1., 0., -1.]]), atol=1e-6, rtol=0)

    def test_prior_composes_five_control_steps(self):
        class ConstantMotion(torch.nn.Module):
            def motion(self, command):
                return command.new_tensor([0.011, 0., 0.]).expand(len(command), -1)

        motion = prior_motion(ConstantMotion(), torch.zeros(2, 2), torch.zeros(2, 10))
        torch.testing.assert_close(motion, torch.tensor([[0.055, 0., 0.], [0.055, 0., 0.]]))

    def test_predictor_sees_both_observations_in_the_future_coordinate_frame(self):
        class ConstantMotion(torch.nn.Module):
            def motion(self, command):
                return command.new_tensor([0.011, 0., 0.]).expand(len(command), -1)

        class Capture(torch.nn.Module):
            def forward(self, value):
                self.value = value
                return value

        model = RoverJEPA(ROOT / "external/jepa-wms/app/plan_common/models/vit.py", motion_prior=ConstantMotion())
        model.predictor = Capture()
        features = torch.randn(1, 2, 64, 64)
        states, actions = torch.zeros(1, 2, 6), torch.zeros(1, 2, 10)
        model(features, states, actions)
        motion = torch.tensor([[0.055, 0., 0.]])
        expected = torch.stack((warp_features(warp_features(features[:, 0], motion), motion),
                                warp_features(features[:, 1], motion)), dim=1)
        torch.testing.assert_close(model.predictor.value[..., :64], expected.flatten(1, 2))

    def test_small_steady_improvements_do_not_count_as_a_plateau(self):
        reference, stale = float("inf"), 0
        for score in [1. - step * 0.0002 for step in range(20)]:
            reference, stale = update_plateau(score, reference, stale)
            self.assertLess(stale, 8)
        for _ in range(8):
            reference, stale = update_plateau(reference, reference, stale)
        self.assertGreaterEqual(stale, 8)

    def test_scan_resize_and_transport_share_the_physical_metre_scale(self):
        axis = torch.linspace(-2.5, 2.5, 101)
        scan = (-0.1 * axis[None, :].expand(101, -1)).reshape(1, -1)
        image = scan_images(scan)[:, :1]
        features = torch.nn.functional.avg_pool2d(image, 14).flatten(2).transpose(1, 2)
        distance = 5.05 / 8
        shifted = warp_features(features, torch.tensor([[distance * 0.11, 0., 0.]]))
        change = (shifted - features).reshape(8, 8)[2:-2, 2:-2]
        torch.testing.assert_close(change, torch.full_like(change, 0.1 * distance / (4 * 0.229)))

    def test_horizon_normalization_does_not_overweight_larger_future_changes(self):
        target = torch.tensor([1., 2.]).reshape(1, 2, 1, 1)
        reference = torch.tensor([1., 4.])
        torch.testing.assert_close(forecast_loss(torch.zeros_like(target), target, torch.ones(1), reference),
                                   torch.tensor(1.))
        torch.testing.assert_close(forecast_loss(target / 2, target, torch.ones(1), reference), torch.tensor(0.25))

    def test_split_accepts_only_the_documented_legacy_recording_without_metadata(self):
        spec = importlib.util.spec_from_file_location("split", ROOT / "deploy/nautilus-jepa-rover/prepare_split.py")
        split = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(split)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "original"
            root.mkdir()
            legacy = root / "mars-rover-training-data-tds6k"
            legacy.mkdir()
            (legacy / "rover_seed_0.hdf5").touch()
            for terrain, seed in [("mars", 1), ("mars", 2), ("mars", 3), ("debug", 1), ("debug", 2), ("debug", 3)]:
                folder = root / f"{terrain}-{seed}"
                folder.mkdir()
                (folder / f"rover_seed_{seed}.hdf5").touch()
                (folder / "metadata.json").write_text(json.dumps({"terrain": terrain, "seed": seed}))
            split.prepare(root, root)
            self.assertEqual(len((root / "train-files.txt").read_text().splitlines()), 5)
            extra = root.parent / "new"
            for seed in (16, 20):
                folder = extra / f"seed-{seed}-mars/pod"
                folder.mkdir(parents=True)
                (folder / f"rover_seed_{seed}.hdf5").touch()
                (folder / "metadata.json").write_text(json.dumps({
                    "seed": seed, "terrain": "mars", "steps": 3000,
                    "task": "AAURoverEnvSimple-v0", "action_policy": "varied"}))
            partial = extra / "seed-17-debug/pod"
            partial.mkdir(parents=True)
            (partial / "rover_seed_17.hdf5").touch()
            split.prepare(root, root, extra)
            selected = (root / "train-files.txt").read_text().splitlines()
            self.assertEqual(len(selected), 6)
            self.assertTrue(any("rover_seed_16.hdf5" in path for path in selected))
            self.assertFalse(any("rover_seed_17.hdf5" in path or "rover_seed_20.hdf5" in path for path in selected))
            marker = extra / "seed-16-mars/pod/metadata.json"
            invalid = json.loads(marker.read_text())
            invalid["steps"] = 1
            marker.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                split.prepare(root, root, extra)
            stray = root / "unknown"
            stray.mkdir()
            (stray / "rover_seed_0.hdf5").touch()
            with self.assertRaises(ValueError):
                split.prepare(root, root)


if __name__ == "__main__":
    unittest.main()
