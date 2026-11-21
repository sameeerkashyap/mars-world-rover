"""Frozen terrain forecasts for reranking measured-scan-safe rover paths."""

from collections import deque
from pathlib import Path

import torch

from scripts.train_rover_baseline import model_for
from scripts.train_rover_jepa import (
    RoverJEPA, encode_state, file_hash, prior_motion, rollout, scan_images, warp_features,
)
from src.terrain_cost import TerrainCost


class TerrainForecast:
    def __init__(self, encoder, predictor, readout, normalization, device):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        self.predictor = predictor.to(self.device).eval().requires_grad_(False)
        self.readout = readout.to(self.device).eval().requires_grad_(False)
        self.normalization = {key: value.to(self.device) for key, value in normalization.items()}
        self.reset()

    def reset(self, mode="jepa"):
        if mode not in ("jepa", "transport"):
            raise ValueError("Forecast mode must be jepa or transport")
        self.mode = mode
        self.features = deque(maxlen=6)
        self.states = deque(maxlen=6)
        self.actions = deque(maxlen=5)
        self.scoring_calls = self.scored_candidates = 0

    @property
    def ready(self):
        return len(self.features) == 6 and len(self.actions) == 5

    @torch.inference_mode()
    def observe(self, observation, applied_action=None):
        if observation.shape != (10206,) or not torch.isfinite(observation[:5]).all():
            raise ValueError("Forecast history requires one calibrated rover policy observation")
        if not torch.isfinite(observation[5:]).all():
            self.features.clear()
            self.states.clear()
            self.actions.clear()
            return
        if self.features:
            if applied_action is None or applied_action.shape != (2,) or not torch.isfinite(applied_action).all():
                raise ValueError("History needs the actual command between observations")
            if not torch.allclose(observation[:2].cpu(), applied_action.cpu(), atol=1e-6, rtol=0):
                raise ValueError("Forecast history observation/action timing mismatch")
            self.actions.append(applied_action.to(self.device).clone())
        scan = observation[5:].reshape(1, -1).to(self.device)
        features = self.encoder.forward_features(scan_images(scan))["x_norm_patchtokens"]
        if features.shape != (1, 64, 384) or not torch.isfinite(features).all():
            raise ValueError("Invalid live encoder features")
        norm = self.normalization
        latent = (features - norm["mean"]) / norm["scale"] @ norm["projection"] / norm["latent_scale"]
        self.features.append(latent[0])
        self.states.append(encode_state(observation[:5].reshape(1, 5).to(self.device))[0])

    @torch.inference_mode()
    def costs(self, commands, lengths):
        if (not self.ready or commands.ndim != 3 or commands.shape[1:] != (25, 2)
                or lengths.shape != (len(commands),) or not torch.isfinite(commands).all()
                or (commands[..., 0].abs() > 0.7).any() or (commands[..., 1].abs() > 0.350001).any()
                or (lengths < 5).any()):
            raise ValueError("Costs need valid history and bounded five-second candidate commands")
        count = len(commands)
        self.scoring_calls += 1
        self.scored_candidates += count
        steps = min(5, int(lengths.max()) // 5)
        context = torch.stack((self.features[0], self.features[-1]))[None].expand(count, -1, -1, -1)
        states = torch.stack((self.states[0], self.states[-1]))[None].expand(count, -1, -1)
        past = torch.stack(tuple(self.actions)).flatten()[None, None].expand(count, -1, -1)
        future = commands.to(self.device).reshape(count, 5, 10)
        actions = torch.cat((past, future), dim=1)
        if self.mode == "jepa":
            forecasts, _ = rollout(self.predictor, context, states, actions, steps)
        else:
            features, previous, forecasts = context[:, -1], states[:, -1, :2], []
            for step in range(steps):
                movement = prior_motion(self.predictor.motion_prior, previous, actions[:, step + 1])
                features = warp_features(features, movement)
                forecasts.append(features)
                previous = actions[:, step + 1, -2:]
            forecasts = torch.stack(forecasts, dim=1)
        costs = self.readout(forecasts).reshape(count, steps, 8, 8)[:, :, 3:5, 3:5].amax(dim=(-1, -2))
        active = torch.arange(1, steps + 1, device=self.device)[None] * 5 <= lengths.to(self.device)[:, None]
        return costs.masked_fill(~active, 0).amax(dim=1).to(commands.device)


def load_forecast(jepa_run, terrain_checkpoint, encoder_source, predictor_source, plan, device):
    path = jepa_run / "training/best.pt"
    expected = {"jepa_checkpoint": path, "terrain_checkpoint": terrain_checkpoint,
                "predictor_source": predictor_source,
                "jepa_source": Path(rollout.__code__.co_filename),
                "motion_source": Path(model_for.__code__.co_filename),
                "readout_source": Path(TerrainCost.forward.__code__.co_filename)}
    for name, source in expected.items():
        if file_hash(source) != plan["sha256"][name]:
            raise ValueError(f"Frozen navigation input changed: {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = torch.load(terrain_checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint["manifest"]["model"] != "rover-dino-wm-v6"
            or head["report"]["schema"] != "rover-terrain-cost-v1"
            or head["report"]["jepa_checkpoint_sha256"] != file_hash(path)):
        raise ValueError("Terrain readout does not belong to the frozen rover model")
    normalization = {key: checkpoint[key] for key in ("mean", "scale", "projection", "latent_scale")}
    for key, value in normalization.items():
        if not torch.equal(value, head[key]):
            raise ValueError(f"Terrain readout normalization changed: {key}")
    encoder = torch.hub.load(str(encoder_source), "dinov2_vits14", source="local", pretrained=True)
    weights = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth"
    if file_hash(weights) != plan["sha256"]["encoder_weights"]:
        raise ValueError("Frozen encoder weights changed")
    predictor = RoverJEPA(predictor_source, motion_prior=model_for(5, 2))
    predictor.load_state_dict(checkpoint["model"], strict=True)
    readout = TerrainCost()
    readout.load_state_dict(head["model"], strict=True)
    return TerrainForecast(encoder, predictor, readout, normalization, device)
