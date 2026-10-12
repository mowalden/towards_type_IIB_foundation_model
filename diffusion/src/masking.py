"""Mask-probability schedules shared by training and sampling."""

from __future__ import annotations

import math

import torch


MASK_SCHEDULES = ("sine", "linear")
TIME_CONDITIONING_MODES = ("discrete", "continuous", "none")


def canonical_mask_schedule(value: object = "sine") -> str:
    """Return a validated mask-schedule name.

    ``sine`` preserves the original sine-squared schedule. ``linear`` uses
    ``m_t = (t + 1) / T``, so uniformly sampled discrete timesteps approximate
    a uniform mask probability on ``(0, 1]``.
    """

    schedule = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sine": "sine",
        "sine_squared": "sine",
        "sin2": "sine",
        "linear": "linear",
        "uniform": "linear",
    }
    try:
        return aliases[schedule]
    except KeyError as error:
        raise ValueError(
            "mask_schedule must be one of: sine, linear"
        ) from error


def canonical_time_conditioning(value: object = "discrete") -> str:
    """Return a validated denoiser time-conditioning mode."""

    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "discrete": "discrete",
        "timestep": "discrete",
        "continuous": "continuous",
        "mask_probability": "continuous",
        "rho": "continuous",
        "none": "none",
        "no_time": "none",
        "disabled": "none",
    }
    try:
        return aliases[mode]
    except KeyError as error:
        raise ValueError(
            "time_conditioning must be one of: discrete, continuous, none"
        ) from error


def timestep_from_mask_fraction(
    masked_count: torch.Tensor,
    active_count: torch.Tensor | int,
    num_timesteps: int,
    mask_schedule: str = "sine",
) -> torch.Tensor:
    """Invert a mask schedule at the current fraction of masked coordinates."""

    if int(num_timesteps) <= 0:
        raise ValueError("num_timesteps must be positive")
    schedule = canonical_mask_schedule(mask_schedule)
    masked = torch.as_tensor(masked_count).float()
    active = torch.as_tensor(
        active_count, device=masked.device
    ).float().clamp_min(1.0)
    rho = (masked / active).clamp(0.0, 1.0)
    if schedule == "linear":
        continuous = int(num_timesteps) * rho - 1.0
    else:
        continuous = (
            int(num_timesteps)
            * (2.0 / math.pi)
            * torch.asin(torch.sqrt(rho))
            - 1.0
        )
    return continuous.round().clamp(0, int(num_timesteps) - 1).long()


def time_condition_from_mask_fraction(
    masked_count: torch.Tensor,
    active_count: torch.Tensor | int,
    *,
    time_conditioning: str = "discrete",
    num_timesteps: int | None = 200,
    mask_schedule: str = "sine",
) -> torch.Tensor | None:
    """Map the current masked fraction to the denoiser's time input."""

    mode = canonical_time_conditioning(time_conditioning)
    masked = torch.as_tensor(masked_count).float()
    active = torch.as_tensor(
        active_count, device=masked.device
    ).float().clamp_min(1.0)
    if mode == "none":
        return None
    if mode == "continuous":
        return (masked / active).clamp(0.0, 1.0)
    if num_timesteps is None:
        raise ValueError("Discrete time conditioning requires num_timesteps")
    return timestep_from_mask_fraction(
        masked,
        active,
        int(num_timesteps),
        mask_schedule,
    )


__all__ = [
    "MASK_SCHEDULES",
    "TIME_CONDITIONING_MODES",
    "canonical_mask_schedule",
    "canonical_time_conditioning",
    "time_condition_from_mask_fraction",
    "timestep_from_mask_fraction",
]
