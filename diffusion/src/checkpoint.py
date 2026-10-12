"""Versioned checkpoints for FluxVacuumTransformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .data.moduli import validate_moduli_statistics
from .masking import canonical_mask_schedule, canonical_time_conditioning
from .model.flux_transformer import FluxVacuumTransformer, FluxVacuumTransformerConfig


CHECKPOINT_FORMAT_VERSION = 5
CHECKPOINT_KIND = "flux_vacuum_transformer"


def checkpoint_mask_schedule(payload: Mapping[str, Any]) -> str:
    """Read the schedule recorded by supervised or policy training.

    Checkpoints created before this option existed used the sine schedule, so
    an absent value intentionally resolves to ``sine``.
    """

    run_config = payload.get("run_config", {})
    if isinstance(run_config, Mapping):
        diffusion = run_config.get("diffusion", {})
        if isinstance(diffusion, Mapping) and "mask_schedule" in diffusion:
            return canonical_mask_schedule(diffusion["mask_schedule"])
        sampling = run_config.get("sampling", {})
        if isinstance(sampling, Mapping) and "mask_schedule" in sampling:
            return canonical_mask_schedule(sampling["mask_schedule"])
    return "sine"


def checkpoint_time_conditioning(payload: Mapping[str, Any]) -> str:
    """Read the architecture's discrete, continuous, or absent time input."""

    model_config = payload.get("model_config", {})
    if isinstance(model_config, Mapping):
        return canonical_time_conditioning(
            model_config.get("time_conditioning", "discrete")
        )
    return "discrete"


def load_flux_transformer_model(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FluxVacuumTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"Not a FluxVacuumTransformer checkpoint: {path}")
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported FluxVacuumTransformer checkpoint format: {path}")
    validate_moduli_statistics(payload["moduli_statistics"])
    model = FluxVacuumTransformer(
        FluxVacuumTransformerConfig(**payload["model_config"]),
        moduli_statistics=payload["moduli_statistics"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, payload
