"""Check starter setup without downloading assets or importing simulator code."""

import importlib
import json
import os
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []

    def check(label, ready, detail):
        print(f"{'OK' if ready else 'MISSING'}: {label}: {detail}")
        if not ready:
            failures.append(label)

    check("Python 3.10", sys.version_info[:2] == (3, 10), sys.version.split()[0])
    for name in ("external/jepa-wms/pyproject.toml", "external/RLRoverLab/README.md"):
        check(name, (root / name).is_file(), str(root / name))
    for name in ("data", "checkpoints", "runs", "notes"):
        check(name, (root / name).is_dir(), str(root / name))
    for name in ("JEPAWM_HOME", "JEPAWM_DSET", "JEPAWM_LOGS", "JEPAWM_CKPT"):
        value = os.environ.get(name, "")
        check(name, bool(value) and Path(value).is_dir(), value or "source scripts/activate.sh")
    for name in ("torch", "torchvision", "torchcodec"):
        try:
            module = importlib.import_module(name)
            check(name, True, getattr(module, "__version__", "import succeeded"))
            if name == "torch":
                print(f"INFO: CUDA available: {module.cuda.is_available()}")
        except Exception as exc:
            check(name, False, str(exc))
    print(f"INFO: Host: {platform.system()} {platform.machine()}")
    if platform.system() == "Darwin":
        print("INFO: Use a compatible Linux host for legacy PointMaze and Isaac/RLRoverLab.")
    print("INFO: Pinned source revisions:")
    print(json.dumps(json.loads((root / "notes/upstream.json").read_text()), indent=2))
    print("INFO: This check does not verify checkpoint loading or planning/simulation milestones.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
