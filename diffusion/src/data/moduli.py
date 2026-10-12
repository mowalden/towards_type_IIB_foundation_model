"""Validate initializer normalization stored in inference checkpoints."""
from typing import Any
import numpy as np

NUM_MODULI_QUERIES = 6
PARAMETERIZATION = "z_real_asinh_imaginary__tau_real_log_imaginary"


def validate_moduli_statistics(stats: dict[str, Any]) -> None:
    if int(stats.get("format_version", -1)) != 3:
        raise ValueError("Unsupported moduli-statistics format")
    if stats.get("parameterization") != PARAMETERIZATION:
        raise ValueError("Moduli statistics use an incompatible parameterization")
    geometries = stats.get("geometries")
    if not isinstance(geometries, dict) or not geometries:
        raise ValueError("Moduli statistics must contain per-geometry entries")
    for geometry_id, row in geometries.items():
        mean = np.asarray(row.get("mean"), dtype=np.float64)
        std = np.asarray(row.get("std"), dtype=np.float64)
        quantile_low = np.asarray(row.get("quantile_low"), dtype=np.float64)
        quantile_high = np.asarray(row.get("quantile_high"), dtype=np.float64)
        mask = np.asarray(row.get("active_query_mask"), dtype=bool)
        if mean.shape != (NUM_MODULI_QUERIES, 2):
            raise ValueError(f"Geometry {geometry_id} has invalid mean shape {mean.shape}")
        if std.shape != mean.shape or mask.shape != (NUM_MODULI_QUERIES,):
            raise ValueError(f"Geometry {geometry_id} has invalid normalization shapes")
        if quantile_low.shape != mean.shape or quantile_high.shape != mean.shape:
            raise ValueError(f"Geometry {geometry_id} has invalid quantile shapes")
        if not all(
            np.all(np.isfinite(value))
            for value in (mean, std, quantile_low, quantile_high)
        ):
            raise ValueError(f"Geometry {geometry_id} statistics must be finite")
        if np.any(std <= 0.0) or not mask[-1]:
            raise ValueError(f"Geometry {geometry_id} statistics are invalid")
        if np.any(quantile_low[mask] > quantile_high[mask]):
            raise ValueError(f"Geometry {geometry_id} has reversed quantile bounds")
