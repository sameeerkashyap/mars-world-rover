"""Unwarped rover feature prediction with the pinned JEPA-WMs architectures."""

import importlib.util
from pathlib import Path

import torch
from torch import nn

import src
from scripts.train_rover_jepa import file_hash


UPSTREAM_HASHES = {
    "app/plan_common/models/vit.py": "81d53872f60477ed3191f14ab3c71ee74e914945df3fac1a1ee378b9905d4087",
    "app/plan_common/models/AdaLN_vit.py": "77e20661849a33199c581b4ab47e2d0417ef2e81571e4998a30cc242b68abe97",
}


class RoverWorldModel(nn.Module):
    def __init__(self, architecture, upstream):
        super().__init__()
        if architecture not in ("adaln", "dino"):
            raise ValueError("Architecture must be adaln or dino")
        upstream = Path(upstream).resolve()
        relative = f"app/plan_common/models/{'AdaLN_vit.py' if architecture == 'adaln' else 'vit.py'}"
        path = upstream / relative
        if file_hash(path) != UPSTREAM_HASHES[relative]:
            raise ValueError(f"Pinned upstream predictor changed: {path}")
        if architecture == "adaln":
            dependency_path = str(upstream / "src")
            if dependency_path not in src.__path__:
                src.__path__.append(dependency_path)
        spec = importlib.util.spec_from_file_location(f"rover_upstream_{architecture}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.architecture = architecture
        self.previous_encoder = nn.Linear(2, 16)
        self.register_buffer("command_scale", torch.tensor([0.7, 0.35]))
        if architecture == "adaln":
            self.predictor = module.vit_predictor_AdaLN(
                img_size=112, patch_size=14, num_frames=2, tubelet_size=1,
                embed_dim=384, predictor_embed_dim=384, depth=2, num_heads=4,
                local_window=(2, -1, -1), use_rope=True, action_dim=10,
                use_proprio=True, proprio_emb_dim=16, proprio_encoder_inpred=False,
                action_encoder_inpred=True, init_scale_factor_adaln=10,
            )
        else:
            self.action_encoder = nn.Linear(10, 16)
            self.predictor = module.ViTPredictor(
                num_patches=64, num_frames=2, dim=416, depth=2, heads=4,
                dim_head=104, mlp_dim=2048, dropout=0.1, use_sdpa=True,
            )

    def forward(self, features, previous, actions):
        batch = len(features)
        if (features.shape != (batch, 2, 64, 384) or previous.shape != (batch, 2, 2)
                or actions.shape != (batch, 2, 10)
                or not all(torch.isfinite(x).all() for x in (features, previous, actions))):
            raise ValueError("Need two finite feature frames, previous commands, and aligned 10-value action blocks")
        past = self.previous_encoder(previous / self.command_scale)[:, :, None].expand(-1, -1, 64, -1)
        actions = actions / self.command_scale.repeat(5)
        if self.architecture == "adaln":
            output = self.predictor(features.reshape(batch, 2, 1, 8, 8, 384), actions, past)[0]
        else:
            command = self.action_encoder(actions)[:, :, None].expand(-1, -1, 64, -1)
            tokens = torch.cat((features, past, command), dim=-1)
            output = self.predictor(tokens.flatten(1, 2)).reshape(batch, 2, 64, 416)[..., :384]
        return output

    def forecast(self, context, previous, blocks, steps=5, detach=False):
        if not 1 <= steps <= 5 or blocks.shape != (len(context), steps + 1, 10):
            raise ValueError("Forecast needs a historical block followed by one to five future blocks")
        outputs = []
        for step in range(steps):
            prediction = self(context, previous, blocks[:, step:step + 2])[:, -1]
            outputs.append(prediction)
            context = torch.cat((context[:, -1:], (prediction.detach() if detach else prediction)[:, None]), dim=1)
            previous = torch.cat((previous[:, -1:], blocks[:, step + 1, -2:][:, None]), dim=1)
        return torch.stack(outputs, dim=1)
