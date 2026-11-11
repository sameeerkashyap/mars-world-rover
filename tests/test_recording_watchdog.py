import importlib.util
from pathlib import Path
import sys
import unittest

path = Path(__file__).resolve().parents[1] / "deploy/nautilus/rover_record.py"
spec = importlib.util.spec_from_file_location("rover_record", path)
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class RecordingWatchdogTests(unittest.TestCase):
    def test_preserves_success_and_failure_exit_codes(self):
        self.assertEqual(record.run_bounded([sys.executable, "-c", "pass"], 5), 0)
        self.assertEqual(record.run_bounded([sys.executable, "-c", "raise SystemExit(3)"], 5), 3)

    def test_stops_a_hung_worker(self):
        self.assertEqual(record.run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], 0.1), 124)
