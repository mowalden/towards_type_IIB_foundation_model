"""Run with: python -m unittest discover -s diffusion/tests -v."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusion.src.pipeline import (
    HODGE_PAIRS, MODES, REPO_ROOT, sample_and_save, to_evaluation_sample,
)


class EvaluationFormatTests(unittest.TestCase):
    def setUp(self):
        self.flux = torch.arange(-6, 6).reshape(1, 12)
        self.generated = {
            "flux": self.flux,
            "initializer": {
                "z": torch.tensor([[1.24 - 2.26j, -3.24 + 4.26j, 99j, 99j, 99j]]),
                "tau": torch.tensor([0.26 + 3.14j]),
            },
        }

    def test_integer_layout_and_scaling(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                flux, z, tau = to_evaluation_sample(self.generated, h12=2, mode=mode)
                np.testing.assert_array_equal(flux, self.flux.numpy())
                np.testing.assert_array_equal(z, [[12, -32, -23, 43]])
                np.testing.assert_array_equal(tau, [[3]] if mode == "gs_inv" else [[3, 31]])
                self.assertTrue(all(x.dtype == np.int64 for x in (flux, z, tau)))

    def test_existing_evaluator_conversion(self):
        # Execute the actual converter without importing unrelated model/solver
        # dependencies. NumPy supplies the same array operations for this test.
        path = REPO_ROOT / "transformer/evaluation_scripts/evaluation_utils.py"
        tree = ast.parse(path.read_text())
        converter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert_samples_to_data")
        namespace = {"jnp": np}
        exec(compile(ast.Module(body=[converter], type_ignores=[]), str(path), "exec"), namespace)
        for mode in MODES:
            parts = to_evaluation_sample(self.generated, h12=2, mode=mode)
            flux, z, tau = namespace["convert_samples_to_data"](
                parts, 10, gs_inv=40 if mode == "gs_inv" else None,
            )
            np.testing.assert_array_equal(flux, self.flux.numpy())
            np.testing.assert_allclose(z, [[1.2 - 2.3j, -3.2 + 4.3j]])
            np.testing.assert_allclose(tau, [0.3 + (4j if mode == "gs_inv" else 3.1j)])

    def test_prompt_order_matches_evaluator(self):
        tree = ast.parse((REPO_ROOT / "transformer/evaluate.py").read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "all_hodge_pairs" for target in n.targets
        ))
        expected = np.array(ast.literal_eval(assignment.value.args[0])).reshape(-1, 2)
        np.testing.assert_array_equal(HODGE_PAIRS, expected)

    def test_invalid_scale_and_nonfinite_initializer(self):
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                to_evaluation_sample(self.generated, h12=2, mode="nflux", discr=scale)
        self.generated["initializer"]["z"][0, 0] = complex(float("nan"), 0)
        with self.assertRaises(ValueError):
            to_evaluation_sample(self.generated, h12=2, mode="nflux")

    def test_existing_output_or_sidecar_prevents_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "samples.p"
            for existing in (output, output.with_suffix(".json")):
                existing.write_bytes(b"keep existing result")
                with patch("diffusion.src.pipeline.sample_for_evaluation") as sample:
                    with self.assertRaises(FileExistsError):
                        sample_and_save("nflux", output=output)
                    sample.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"keep existing result")
                existing.unlink()


if __name__ == "__main__":
    unittest.main()
