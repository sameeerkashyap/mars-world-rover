"""Bound camera startup; preserve a physics position trace if rendering fails."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def run_bounded(command, seconds):
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return 124


if __name__ == "__main__":
    command = [sys.executable, str(Path(__file__).with_name("rover_record_worker.py")), *sys.argv[1:]]
    output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path("/data/runs/rover-video")
    summary = output / "summary.json"
    summary.unlink(missing_ok=True)
    trace_only = "--no-video" in sys.argv
    status = run_bounded(command, 240 if trace_only else 180)
    if not summary.exists() and not trace_only:
        print("Camera capture failed; recording the physics position trace without RTX rendering.", flush=True)
        status = run_bounded([*command, "--no-video"], 240)
    if summary.exists():
        report = json.loads(summary.read_text())
        if report["steps"] > 0:
            status = 0
    else:
        status = status or 1
    raise SystemExit(status)
