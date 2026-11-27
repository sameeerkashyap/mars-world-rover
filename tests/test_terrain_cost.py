import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_rover_jepa import file_hash, sequence_arrays
from scripts.train_terrain_cost import evaluate_forecasts, fit_readout, recording_data, recording_split
from src.terrain_cost import TerrainCost, terrain_targets


class ScanEncoder(torch.nn.Module):
    def forward_features(self, images):
        patches = F.avg_pool2d(images[:, :1], 14).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": patches.expand(-1, -1, 384)}


class Persist(torch.nn.Module):
    motion_prior = None

    def forward(self, features, states, actions):
        return features[:, -1], states[:, -1, 2:]


class TerrainCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_labels_follow_scan_geometry_and_conservatively_mark_unknown(self):
        scan = torch.zeros(1, 101, 101)
        flat = terrain_targets(scan.flatten(1)).reshape(8, 8)
        self.assertEqual(flat[3, 3], 0)
        self.assertGreater(flat[0, 0], 0.9)
        scan[:, 48:53, 48:53] = -0.4
        rock = terrain_targets(scan.flatten(1)).reshape(8, 8)
        self.assertGreater(rock[3, 3], flat[3, 3])
        torch.testing.assert_close(terrain_targets(torch.full((1, 10201), float("nan"))), torch.ones(1, 64))
        with self.assertRaises(ValueError):
            terrain_targets(torch.zeros(1, 100))
        with self.assertRaises(ValueError):
            TerrainCost()(torch.full((1, 64, 64), float("nan")))

    def test_recordings_preserve_label_timing_and_reject_test_or_duplicate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.hdf5"
            with h5py.File(path, "w") as data:
                for episode in range(2):
                    group = data.create_group(f"data/demo_{episode}")
                    obs = np.zeros((42, 10206), dtype=np.float32)
                    obs[21:, 5:] = -0.02
                    if episode:
                        obs[:, 5 + 48 * 101:5 + 53 * 101] = -0.4
                    group["obs"] = obs
                    group["actions"] = np.zeros((42, 2), np.float32)
                    group["dones"] = np.zeros(42)
            checkpoint = {"mean": torch.zeros(384), "scale": torch.ones(384),
                          "projection": torch.eye(384)[:, :64], "latent_scale": torch.ones(64)}
            episodes = recording_data([path], ScanEncoder(), checkpoint, "cpu")
            sequences = sequence_arrays(path)
            self.assertEqual(len(episodes), 2)
            for episode, sequence in zip(episodes, sequences):
                self.assertEqual(episode[1].shape, (9, 64, 64))
                torch.testing.assert_close(episode[4], terrain_targets(torch.tensor(sequence[1][:, 5:])))
            validation, testing = root / "validation", root / "test"
            validation.write_text("validation")
            testing.write_text("reserved")
            manifest = {"files": {str(item): file_hash(item) for item in (path, validation, testing)}}
            split = {"train": [str(path)], "validation": [str(validation)], "test": [str(testing)]}
            recording_split([path], [validation], split, manifest)
            for training, held_out in (([testing], [validation]), ([path], [path]), ([path, path], [validation])):
                with self.assertRaises(ValueError):
                    recording_split(training, held_out, split, manifest)
            alias = root / "renamed-test"
            alias.write_bytes(testing.read_bytes())
            with self.assertRaises(ValueError):
                recording_split([alias], [validation], split, manifest)

    def test_readout_learns_reloads_and_forecasts_use_only_context(self):
        torch.manual_seed(0)
        features = torch.randn(128, 64, 64)
        targets = (features[..., 0] * 2).sigmoid()
        episode = ("train", features, torch.zeros(128, 6), torch.zeros(127, 10), targets)
        validation = ("validation", features[-16:], episode[2][-16:], episode[3][-15:], targets[-16:])
        training = ("train", features[:96], episode[2][:96], episode[3][:95], targets[:96])
        head = TerrainCost()
        before = F.mse_loss(head(validation[1]), validation[4]).item()
        metrics, constant = fit_readout(head, [training], [validation], 60, "cpu")
        self.assertLess(metrics["measured_validation_mse"], before * 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save(head.state_dict(), path)
            restored = TerrainCost()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(head(features), restored(features))
        scores = evaluate_forecasts(head, Persist(), [validation], constant, "cpu")
        self.assertEqual(scores["windows"], 10)
        expected = F.mse_loss(head(validation[1][1:11]), validation[4][6:16]).item()
        self.assertAlmostEqual(scores["five_seconds"]["jepa_mse"], expected)
        for horizon in ("one_second", "five_seconds"):
            self.assertAlmostEqual(scores[horizon]["jepa_mse"], scores[horizon]["copy_mse"])
            self.assertAlmostEqual(scores[horizon]["jepa_mse"], scores[horizon]["transport_mse"])
        json.dumps(scores, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
