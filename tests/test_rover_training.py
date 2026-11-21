from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.train_rover_jepa import evaluate, file_hash, windows
from src.rover_training import WindowDataset, cached_episodes, feature_statistics, restore_progress, save_progress


class Persist(torch.nn.Module):
    def forward(self, features, states, actions):
        return features[:, -1], states[:, -1, 2:]


class RoverTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def episodes(self):
        return [(str(index), torch.randn(9, 64, 384), torch.randn(9, 6), torch.randn(8, 10))
                for index in range(2)]

    def test_lazy_windows_and_scores_match_original_without_crossing_episodes(self):
        torch.manual_seed(0)
        episodes = self.episodes()
        mean, scale, projection, latent_scale = torch.zeros(384), torch.ones(384), torch.eye(384)[:, :64], torch.ones(64)
        lazy = WindowDataset(episodes, mean, scale, projection, latent_scale)
        original = windows(episodes, mean, scale, projection, latent_scale)
        self.assertEqual(len(lazy), 6)
        for actual, expected in zip(lazy.batch(range(6)), original):
            torch.testing.assert_close(actual, expected)
        lazy_scores = evaluate(Persist(), lazy.columns(), "cpu", projection, latent_scale)
        original_scores = evaluate(Persist(), original, "cpu", projection, latent_scale)
        self.assertEqual(lazy_scores, original_scores)
        with self.assertRaises(IndexError):
            lazy.location(6)

    def test_cache_reuses_features_and_invalidates_changed_data_or_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording"
            path.write_text("first")
            encode = Mock(return_value=self.episodes())
            def load(signature):
                return cached_episodes([path], None, "cpu", Path(directory) / "cache", signature, encode, file_hash)
            first = load("encoder-1")
            second = load("encoder-1")
            self.assertEqual(encode.call_count, 1)
            torch.testing.assert_close(first[0][1], second[0][1])
            path.write_text("second")
            load("encoder-1")
            load("encoder-2")
            self.assertEqual(encode.call_count, 3)

    def test_streaming_statistics_match_direct_training_statistics(self):
        torch.manual_seed(2)
        episodes = self.episodes()
        mean, scale, projection, latent_scale, _, retained = feature_statistics(episodes)
        features = torch.cat([episode[1] for episode in episodes])
        torch.testing.assert_close(mean, features.mean(dim=(0, 1)), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(scale, features.std(dim=(0, 1), unbiased=False))
        latent = ((features - mean) / scale) @ projection
        torch.testing.assert_close(latent_scale, latent.std(dim=(0, 1), unbiased=False))
        self.assertGreater(retained, 0)
        self.assertLessEqual(retained, 1)

    def test_full_feature_windows_skip_pca(self):
        episodes = self.episodes()
        with patch("torch.linalg.eigh", side_effect=AssertionError("Full features do not need PCA")):
            mean, scale, projection, latent_scale, _, retained = feature_statistics(episodes, project=False)
        self.assertIsNone(projection)
        self.assertIsNone(latent_scale)
        self.assertEqual(retained, 1.)
        dataset = WindowDataset(episodes, mean, scale, projection, latent_scale)
        features, _, _, full = dataset.batch([0, 1])
        self.assertEqual(features.shape, (2, 7, 64, 384))
        torch.testing.assert_close(features, full)

    def test_resume_reproduces_next_optimizer_step_and_rejects_changed_run(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters())
        def step():
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(5, 3)).square().mean().backward()
            optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_progress(path, model, optimizer, {"data": "fixed"}, 1, [], 1., 1., 0, "epoch_limit")
            expected_random = (random.random(), np.random.rand())
            step()
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            restore_progress(path, model, optimizer, {"data": "fixed"})
            self.assertEqual(expected_random, (random.random(), np.random.rand()))
            step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                restore_progress(path, model, optimizer, {"data": "changed"})



if __name__ == "__main__":
    unittest.main()
