"""Check numerical source consistency and retained trajectory data."""

from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduction_entry", ROOT / "scripts" / "reproduce.py")
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)
from run_nonlinear import compact_refinement_trajectories
from euler_path import figures, nonlinear_figures


class ReproductionTests(unittest.TestCase):
    def test_default_output_and_changed_source_reject_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            for name in ("config.json", "nonlinear_config.json", "equal_evaluation_config.json", "gaussian_higher_order_config.json"):
                (root / "experiments" / name).write_text("{}")
            for name in ("requirements.txt", "requirements.lock.txt"):
                (root / name).write_text("")
            outputs = []

            def changed_generation(config, output):
                output.mkdir(parents=True)
                outputs.append(output)
                config.write_text('{"changed": true}')
                return {}

            with patch.object(reproduction, "ROOT", root), \
                 patch("sys.argv", ["reproduce.py"]), \
                 patch.object(reproduction.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                 patch("run.generate", side_effect=changed_generation), \
                 patch("euler_path.figures.render"), \
                 patch("run_nonlinear.generate", return_value={}), \
                 patch("euler_path.equal_evaluation.generate", return_value={}), \
                 patch("euler_path.gaussian_higher_order.generate", return_value={}), \
                 patch("euler_path.nonlinear_figures.render", return_value=0), \
                 patch.object(reproduction, "write_manuscript_values"):
                with self.assertRaisesRegex(RuntimeError, "changed during reproduction"):
                    reproduction.main()
            self.assertEqual(outputs, [root / "build" / "reproduction"])
            self.assertFalse((root / "results").exists())

    def test_compaction_preserves_primary_paths_and_refinement_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("base_radius", "base_angle", "corrected_radius", "corrected_angle",
                     "reference_radius", "reference_angle")
            values = np.arange(6, dtype=float).reshape(3, 2)
            for order in (64, 128, 256):
                np.savez_compressed(root / f"q{order}_uniform_2.npz", times=np.arange(3),
                                    weights=np.array([0.5, 0.5]), base_w2=np.arange(3),
                                    **{name: values for name in names})
            quadrature = {}
            for order in (64, 128, 256):
                path = root / f"quadrature_{order}.npz"
                np.savez_compressed(path, weights=np.array([0.5, 0.5]))
                quadrature[path] = path.read_bytes()
            primary = (root / "q128_uniform_2.npz").read_bytes()
            compact_refinement_trajectories(root, 128)
            self.assertEqual(primary, (root / "q128_uniform_2.npz").read_bytes())
            for path, original in quadrature.items():
                self.assertEqual(original, path.read_bytes())
            for order in (64, 256):
                with np.load(root / f"q{order}_uniform_2.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["times"], np.arange(3))
                    np.testing.assert_array_equal(saved["base_w2"], np.arange(3))
                    for name in names:
                        self.assertNotIn(name, saved.files)
                        np.testing.assert_array_equal(saved["terminal_" + name], values[-1])


if __name__ == "__main__":
    unittest.main()
