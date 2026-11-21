from pathlib import Path
import re
import tempfile
import unittest

import torch
import yaml

from scripts.train_world_model_pilot import action_donors, candidate_passed, evaluate, selected_indices
from scripts.train_world_model_pilot import residual_forecasts
from scripts.evaluate_world_model_pilot import validate_recordings
from scripts.train_rover_jepa import file_hash
from src.rover_training import WindowDataset
from src.rover_world_model import RoverWorldModel


ROOT = Path(__file__).resolve().parents[1]


class RecordedCommands(RoverWorldModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.calls = []

    def forward(self, features, previous, actions):
        self.calls.append((previous.clone(), actions.clone()))
        return features + actions[..., :1, None]


class CopyFeatures(torch.nn.Module):
    def forecast(self, context, previous, blocks, steps=5):
        return context[:, -1:].expand(-1, steps, -1, -1)


class PersistLatent(torch.nn.Module):
    motion_prior = None

    def forward(self, features, states, actions):
        return features[:, -1], states[:, -1, 2:]


class RoverWorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_predictors_are_causal_action_sensitive_and_reloadable(self):
        for architecture in ("dino", "adaln"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(3)
                model = RoverWorldModel(architecture, ROOT / "external/jepa-wms").eval()
                context = torch.randn(1, 2, 64, 384)
                previous = torch.zeros(1, 2, 2)
                blocks = torch.zeros(1, 6, 10)
                baseline = model(context, previous, blocks[:, :2])
                changed_context, changed_blocks = context.clone(), blocks.clone()
                changed_context[:, 1] += 1
                changed_blocks[:, 1, ::2] = 0.5
                changed = model(changed_context, previous, changed_blocks[:, :2])
                torch.testing.assert_close(changed[:, 0], baseline[:, 0], atol=1e-6, rtol=1e-6)
                action_changed = model(context, previous, changed_blocks[:, :2])
                self.assertGreater((action_changed[:, 1] - baseline[:, 1]).abs().max().item(), 1e-5)
                self.assertEqual(model.forecast(context, previous, blocks).shape, (1, 5, 64, 384))
                loss = (baseline - context).square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.pt"
                    torch.save(model.state_dict(), path)
                    restored = RoverWorldModel(architecture, ROOT / "external/jepa-wms").eval()
                    restored.load_state_dict(torch.load(path, weights_only=True))
                    torch.testing.assert_close(restored(context, previous, blocks[:, :2]), baseline, atol=0, rtol=0)
                with self.assertRaises(ValueError):
                    model(context, previous, blocks[:, :2, :2])

    def test_rollout_uses_candidate_commands_as_previous_commands(self):
        model = RecordedCommands()
        context = torch.zeros(1, 2, 64, 384)
        previous = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        blocks = torch.arange(60.).reshape(1, 6, 10)
        forecast = model.forecast(context, previous, blocks)
        for step, (past, commands) in enumerate(model.calls):
            torch.testing.assert_close(commands, blocks[:, step:step + 2])
            expected = previous if step == 0 else torch.stack((previous[:, 1] if step == 1 else blocks[:, step - 1, -2:],
                                                              blocks[:, step, -2:]), dim=1)
            torch.testing.assert_close(past, expected)
        torch.testing.assert_close(forecast[:, :, 0, 0], torch.tensor([[10., 30., 60., 100., 150.]]))
        for steps in (1, 2):
            torch.testing.assert_close(model.forecast(context, previous, blocks[:, :steps + 1], steps), forecast[:, :steps])
        with self.assertRaises(ValueError):
            model.forecast(context, previous, blocks[:, 1:])

    def test_metrics_use_correct_future_frames_and_identical_sample_indices(self):
        features = torch.arange(9.).reshape(9, 1, 1).expand(-1, 64, 384)
        episodes = [(f"recording:episode-{index}", features, torch.zeros(9, 6), torch.zeros(8, 10)) for index in range(2)]
        dataset = WindowDataset(episodes, torch.zeros(384), torch.ones(384), torch.eye(384)[:, :64], torch.ones(64))
        indices = selected_indices(dataset, 128)
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4, 5])
        donors = action_donors(dataset, indices)
        torch.testing.assert_close(donors, action_donors(dataset, indices))
        self.assertTrue(all(dataset.location(index)[0][0] != dataset.location(donor)[0][0]
                            for index, donor in zip(indices, donors)))
        scores = evaluate(CopyFeatures(), dataset, indices, torch.ones(384), None, "cpu")
        self.assertEqual(scores["model"], [1., 25.])
        self.assertEqual(scores["copy"], scores["model"])
        self.assertEqual(scores["transport"], scores["model"])
        self.assertEqual(scores["action_shuffle_increase"], [0., 0.])
        self.assertEqual(scores["normalized_training_objective"], 2.5)
        self.assertEqual(len(scores["by_episode"]), 2)
        self.assertEqual(scores["near_rover"]["model"], scores["model"])
        with self.assertRaises(ValueError):
            action_donors(dataset, indices[:3])

    def test_pilot_deployment_contains_sources_and_no_elapsed_time_cutoff(self):
        folder = ROOT / "deploy/nautilus-world-model-pilot"
        job = yaml.safe_load((folder / "job.yaml").read_text())
        pod = job["spec"]["template"]["spec"]
        for spec in (job["spec"], pod):
            self.assertNotIn("activeDeadlineSeconds", spec)
            self.assertNotIn("ttlSecondsAfterFinished", spec)
        files = dict(re.findall(r'--from-file="([^=]+)=\$root/([^\"]+)"', (folder / "deploy.sh").read_text()))
        items = next(v["configMap"]["items"] for v in pod["volumes"] if v["name"] == "source")
        self.assertEqual(set(files), {item["key"] for item in items})
        self.assertTrue(all((ROOT / source).is_file() for source in files.values()))
        for item in items:
            if item["path"].startswith("upstream/"):
                self.assertEqual(files[item["key"]], "external/jepa-wms/" + item["path"].removeprefix("upstream/"))
        runner = (folder / "train.sh").read_text()
        self.assertNotIn("--test-data", runner)
        self.assertNotIn("timeout ", runner)

    def test_current_control_preserves_features_outside_its_pca_subspace(self):
        torch.manual_seed(4)
        features = torch.randn(2, 7, 64, 384)
        old = {"mean": torch.randn(384), "scale": torch.rand(384) + 0.1,
               "projection": torch.eye(384)[:, :64], "latent_scale": torch.rand(64) + 0.1}
        normalization = {"mean": torch.randn(384), "scale": torch.rand(384) + 0.1}
        outputs = residual_forecasts((PersistLatent(), old), features, torch.zeros(2, 2, 2), torch.zeros(2, 6, 10), normalization)
        for value in outputs.values():
            torch.testing.assert_close(value, features[:, 1:2].expand(-1, 5, -1, -1), atol=1e-5, rtol=1e-5)

    def test_evaluation_refuses_training_recordings_in_place_of_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            train, validation = Path(folder) / "train", Path(folder) / "validation"
            train.write_bytes(b"training")
            validation.write_bytes(b"validation")
            manifest = {"files": {str(path): file_hash(path) for path in (train, validation)}}
            split = {"validation": [str(validation)]}
            validate_recordings(manifest, [validation], split)
            with self.assertRaises(ValueError):
                validate_recordings(manifest, [train], split)
            manifest["validation_files"] = [str(validation)]
            validate_recordings(manifest, [validation], {"validation": []})

    def test_candidate_must_improve_current_predictor_and_copy_at_both_horizons(self):
        scores = {"model": [0.8, 0.8], "copy": [1., 1.], "current_residual": [0.9, 0.9],
                  "gain_over_transport": [0.2, 0.2], "action_shuffle_increase": [0.01, 0.01]}
        self.assertTrue(candidate_passed(scores))
        scores["current_residual"] = [0.7, 0.9]
        self.assertFalse(candidate_passed(scores))
        scores["current_residual"] = [0.9, 0.9]
        scores["copy"] = [1., 0.7]
        self.assertFalse(candidate_passed(scores))
        scores["copy"] = [1., 1.]
        scores["action_shuffle_increase"] = [0.01, 0.]
        self.assertFalse(candidate_passed(scores))


if __name__ == "__main__":
    unittest.main()
