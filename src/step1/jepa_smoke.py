"""Load the reference world model on CUDA; this is not a planning evaluation."""
import json
import importlib.util
import os
from pathlib import Path

import torch
import torchcodec
import yaml

from jepa_plan import SOURCE, prepare_config

if not torch.cuda.is_available():
    raise RuntimeError("The world-model Job requires a working CUDA GPU")
output = Path(os.environ["JEPAWM_LOGS"]) / os.environ["HOSTNAME"]
output.mkdir(parents=True, exist_ok=True)
config = prepare_config(yaml.safe_load(SOURCE.read_text()), output, 2)
config_path = output / "model-config.yaml"
config_path.write_text(yaml.safe_dump(config))
spec = importlib.util.spec_from_file_location("hubconf", "/runtime/jepa-wms/hubconf.py")
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)
# The public loader uses local decoder weights that this reference run omits.
model, preprocessor = hub._load_model_with_config(
    str(config_path), model_name="jepa_wm_pointmaze", device="cuda:0"
)
report = {
    "check": "pretrained model loading only",
    "model": type(model).__name__,
    "preprocessor": type(preprocessor).__name__,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "torchcodec": torchcodec.__version__,
    "checkpoint": config["model_kwargs"]["checkpoint"],
}
(output / "model.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
