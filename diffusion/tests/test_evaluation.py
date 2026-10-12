"""Format and orchestration tests; physics algorithms remain in transformer/."""

import ast
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from diffusion import evaluate as evaluation


def panel(mode):
    return [
        [np.zeros((2, 4 * (h12 + 1)), dtype=np.int64),
         np.full((2, 2 * h12), 12, dtype=np.int64),
         np.full((2, 1 if mode == "gs_inv" else 2), 23, dtype=np.int64)]
        for h12, _ in evaluation.HODGE_PAIRS
    ]


class EvaluationTests(unittest.TestCase):
    def test_metadata_scale_and_geometry_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.p"
            path.write_bytes(pickle.dumps(panel("nflux")))
            sidecar = path.with_suffix(".json")
            metadata = {"mode": "nflux", "discr": 100,
                        "hodge_pairs_h12_h11": [list(p) for p in evaluation.HODGE_PAIRS]}
            sidecar.write_text(json.dumps(metadata))
            self.assertEqual(evaluation.load_samples(path, "nflux")[1], 100)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                evaluation.load_samples(path, "nflux", 10)
            with self.assertRaisesRegex(ValueError, "Requested"):
                evaluation.load_samples(path, "w0")
            metadata["hodge_pairs_h12_h11"].reverse()
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "geometry order"):
                evaluation.load_samples(path, "nflux")
            sidecar.unlink()
            self.assertEqual(evaluation.load_samples(path, "nflux")[1], 10)
            malformed = panel("nflux")
            malformed[0][1] = np.zeros((2, 3))
            path.write_bytes(pickle.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "shape"):
                evaluation.load_samples(path, "nflux")

    def test_all_modes_use_shared_converter_and_save_separate_results(self):
        # Use the actual transformer converter; mock only heavyweight imports
        # and physics solving so this can run without Flax or jaxvacua.
        utils_path = evaluation.REPO_ROOT / "transformer/evaluation_scripts/evaluation_utils.py"
        converter = next(n for n in ast.parse(utils_path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == "convert_samples_to_data")
        utils = types.ModuleType("evaluation_scripts.evaluation_utils")
        utils.jnp = np
        exec(compile(ast.Module(body=[converter], type_ignores=[]), str(utils_path), "exec"), utils.__dict__)
        calls = []

        def validate(flux, z, tau, indices, model, out, tol, use_scipy, *flags):
            calls.append((model.h12, tau.copy(), tol, use_scipy, flags))
            np.testing.assert_allclose(z, 1.2 + 1.2j)
            out.update(valid=2, valid_unique=1)
            return flux[:1], z[:1], tau[:1], indices[:1], out

        utils.validate_vacua = validate
        output = types.ModuleType("evaluation_scripts.output_utils")
        output.print_console_output = lambda *args: None
        jax = types.ModuleType("jax")
        jax.numpy = np
        jax.config = types.SimpleNamespace(update=lambda *args: None)
        jax.clear_caches = lambda: None

        class Finder:
            def __init__(self, h12, model_ID, maximum_degree):
                self.h12 = h12
                self.maximum_degree = maximum_degree

            def tadpole(self, flux):
                return np.full(len(flux), 40)

            def superpotential_gauge_invariant(self, z, tau, flux):
                return np.full(len(flux), 1.1)

        jvc = types.ModuleType("jaxvacua")
        jvc.FluxVacuaFinder = Finder
        modules = {"jax": jax, "jax.numpy": np, "jaxvacua": jvc,
                   "evaluation_scripts": types.ModuleType("evaluation_scripts"),
                   "evaluation_scripts.evaluation_utils": utils,
                   "evaluation_scripts.output_utils": output}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "transformer/data/data_multiple"
            data_dir.mkdir(parents=True)
            log = {f"{h12} {h11}": {"model_ID": i}
                   for i, (h12, h11) in enumerate(evaluation.HODGE_PAIRS)}
            (data_dir / "geometry_log_degree_10.p").write_bytes(pickle.dumps(log))
            for h12 in (2, 3, 4, 5):
                np.save(data_dir / f"train_data_raw_reference_{h12}.npy", np.zeros((1, 4 * (h12 + 1))))
            for run in set(evaluation.REFERENCE_RUNS.values()):
                config = root / f"transformer/logs/run_{run}/run_{run}.json"
                config.parent.mkdir(parents=True)
                config.write_text(json.dumps({"train_data_flux": "/original/train_data_reference.txt"}))
            samples_dir = root / "diffusion/samples"
            samples_dir.mkdir(parents=True)
            with patch.object(evaluation, "REPO_ROOT", root), patch.dict(sys.modules, modules), \
                 patch.object(sys, "path", list(sys.path)), patch.dict("os.environ", {}), \
                 contextlib.redirect_stdout(io.StringIO()):
                for mode in evaluation.REFERENCE_RUNS:
                    with self.subTest(mode=mode):
                        (samples_dir / f"diffusion_{mode}_samples.p").write_bytes(pickle.dumps(panel(mode)))
                        calls.clear()
                        result_path = evaluation.evaluate_samples(mode)
                        self.assertEqual(result_path.parent, root / "diffusion/evaluations")
                        summary = json.loads(result_path.with_name(f"diffusion_{mode}_samples_summary.json").read_text())
                        self.assertEqual((summary["total_samples"], summary["valid_unique"], summary["valid_unique_novel"]), (40, 20, 0))
                        self.assertEqual([c[0] for c in calls], [p[0] for p in evaluation.HODGE_PAIRS])
                        for _, tau, tol, scipy, flags in calls:
                            np.testing.assert_allclose(tau, 2.3 + (4j if mode == "gs_inv" else 2.3j))
                            self.assertEqual((tol, scipy, flags), (1e-10, True, (False, False, False)))
                        result = pickle.loads(result_path.read_bytes())
                        self.assertEqual(len(result), 20)
                        if mode != "unconditional":
                            prompt = {"nflux": 40, "w0": 11, "gs_inv": 40}[mode]
                            expected = {"nflux": 40, "w0": 1.1, "gs_inv": 4}[mode]
                            np.testing.assert_allclose(result[f"2 120 {prompt}"]["dist"], [expected])
                        with self.assertRaises(FileExistsError):
                            evaluation.evaluate_samples(mode)


if __name__ == "__main__":
    unittest.main()
