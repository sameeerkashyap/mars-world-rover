"""Exercise the real pinned CLI parser, without downloading the terrain archive."""
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch

try:
    import gdown.__main__ as cli
except ImportError:
    cli = None


@unittest.skipIf(cli is None, "Install gdown==6.5.1 to run the real CLI compatibility check")
class GdownCliTests(unittest.TestCase):
    def test_runtime_download_arguments_are_supported(self):
        setup = Path(__file__).resolve().parents[1] / "deploy/nautilus/runtime-setup.sh"
        line = next(line.strip() for line in setup.read_text().splitlines()
                    if line.strip().startswith("python -m gdown "))
        arguments = shlex.split(line)[3:]
        with patch.object(sys, "argv", ["gdown"] + arguments), patch.object(cli, "download", return_value="terrain.zip") as download:
            cli.main()
        self.assertEqual(download.call_args.kwargs["id"], "1VXFTD2OgHcsQL_ifO81AzD2HDkA98h93")
        self.assertEqual(download.call_args.kwargs["output"], "/runtime/terrain_assets.zip")
        self.assertEqual(download.call_args.kwargs["retries"], 3)


if __name__ == "__main__":
    unittest.main()
