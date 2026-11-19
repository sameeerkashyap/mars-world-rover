"""Freeze recording-level splits; exclude all concurrently collected data."""

import json
from pathlib import Path
import sys


def prepare(root, output, additional_root=None):
    selected = {"train": [], "validation": [], "test": []}
    seen = set()
    for path in sorted(root.glob("**/rover_seed_*.hdf5")):
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        elif path.relative_to(root) == Path("mars-rover-training-data-tds6k/rover_seed_0.hdf5"):
            # This original run predates metadata.json; its terrain/seed are in docs/results.md.
            metadata = {"terrain": "mars", "seed": 0}
        else:
            raise ValueError(f"Unidentified recording: {path}")
        key = (metadata["terrain"], metadata["seed"])
        split = "train" if (key[0] == "mars" and key[1] in (0, 1, 2, 3)) or key == ("debug", 3) else {
            ("debug", 1): "validation", ("debug", 2): "test"}.get(key)
        if split is None:
            continue
        if key in seen:
            raise ValueError(f"Duplicate recording for {key}")
        seen.add(key)
        selected[split].append(str(path))
    if len(selected["train"]) != 5 or len(selected["validation"]) != 1 or len(selected["test"]) != 1:
        raise ValueError(f"Incomplete original recording split: {selected}")
    if additional_root is not None:
        for seed in (16, 17, 18, 19):
            terrain = "mars" if seed % 2 == 0 else "debug"
            markers = list(additional_root.glob(f"seed-{seed}-{terrain}/*/metadata.json"))
            if len(markers) > 1:
                raise ValueError(f"Duplicate completed collection for seed {seed}")
            if not markers:
                continue
            metadata = json.loads(markers[0].read_text())
            if (metadata.get("seed") != seed or metadata.get("terrain") != terrain
                    or metadata.get("steps") != 3000 or metadata.get("task") != "AAURoverEnvSimple-v0"
                    or metadata.get("action_policy") != "varied"):
                raise ValueError(f"Unexpected collection metadata: {markers[0]}")
            path = markers[0].parent / f"rover_seed_{seed}.hdf5"
            if not path.is_file():
                raise ValueError(f"Completion marker has no recording: {path}")
            selected["train"].append(str(path))
    for split, paths in selected.items():
        (output / f"{split}-files.txt").write_text("\n".join(paths) + "\n")
    (output / "split.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    prepare(Path("/data/datasets/rover"), Path(sys.argv[1]), Path("/data/datasets/rover-jepa-collection"))
