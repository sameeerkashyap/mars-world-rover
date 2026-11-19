import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class NavigationExitTests(unittest.TestCase):
    def test_simulator_cleanup_cannot_turn_an_error_into_exit_zero(self):
        path = Path(__file__).resolve().parents[1] / "deploy/nautilus-navigation/rover_navigate.py"
        guard = ast.parse(path.read_text()).body[-1]
        code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
        for failure, expected in ((None, 0), (RuntimeError("failed trial"), 1)):
            with self.subTest(expected=expected):
                main = Mock(side_effect=failure)
                close = Mock(side_effect=SystemExit(0))
                with self.assertRaises(SystemExit) as caught:
                    exec(code, {"__name__": "__main__", "main": main,
                                "app": SimpleNamespace(close=close), "traceback": Mock()})
                self.assertEqual(caught.exception.code, expected)
                close.assert_called_once()
