"""Explicit observable tokens for FluxVacuumTransformer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ObservableConditioningContext:
    """Per-row physical targets and activation masks.

    Tensor-valued fields retain the original single-observable API.  Joint
    conditioning uses mappings keyed by the canonical observable names.
    """

    target: torch.Tensor | Mapping[str, torch.Tensor]
    geometry_id: torch.Tensor
    active: torch.Tensor | Mapping[str, torch.Tensor]

    @property
    def is_joint(self) -> bool:
        target_is_mapping = isinstance(self.target, Mapping)
        active_is_mapping = isinstance(self.active, Mapping)
        if target_is_mapping != active_is_mapping:
            raise ValueError(
                "Observable conditioning target and active must both be mappings "
                "or both be tensors"
            )
        return target_is_mapping

    def select(self, observable: str) -> "ObservableConditioningContext":
        """Return the single-observable view used by one token encoder."""

        if not self.is_joint:
            return self
        target = self.target
        active = self.active
        assert isinstance(target, Mapping) and isinstance(active, Mapping)
        if observable not in target or observable not in active:
            raise ValueError(
                f"Joint conditioning context is missing {observable!r}"
            )
        return ObservableConditioningContext(
            target=target[observable],
            geometry_id=self.geometry_id,
            active=active[observable],
        )

    def any_active(self) -> bool:
        if not self.is_joint:
            return bool(torch.as_tensor(self.active).bool().any())
        active = self.active
        assert isinstance(active, Mapping)
        return any(bool(torch.as_tensor(value).bool().any()) for value in active.values())


SUPPORTED_CONDITIONAL_OBSERVABLES = ("N_flux", "g_s", "W0_abs")

_OBSERVABLE_ALIASES = {
    "n_flux": "N_flux",
    "nflux": "N_flux",
    "g_s": "g_s",
    "gs": "g_s",
    "w0_abs": "W0_abs",
    "|w_0|": "W0_abs",
    "|w0|": "W0_abs",
}


def canonical_observable_name(observable: str) -> str:
    """Normalize config-friendly observable spellings to dataset field names."""

    value = str(observable).strip()
    if value in SUPPORTED_CONDITIONAL_OBSERVABLES:
        return value
    canonical = _OBSERVABLE_ALIASES.get(value.lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported conditional observable {observable!r}; expected one of "
            f"{SUPPORTED_CONDITIONAL_OBSERVABLES}"
        )
    return canonical


def observable_transform_name(observable: str) -> str:
    observable = canonical_observable_name(observable)
    if observable == "N_flux":
        return "identity"
    if observable == "g_s":
        return "log"
    if observable == "W0_abs":
        return "log1p_delta"
    raise ValueError(
        f"Unsupported conditional observable {observable!r}; expected one of "
        f"{SUPPORTED_CONDITIONAL_OBSERVABLES}"
    )


class ObservableTokenConditioner(nn.Module):
    """Encode one selected physical observable as one sequence token."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_width: int,
        num_geometries: int,
        observable: str = "N_flux",
        transform_delta: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_width, num_geometries) <= 0:
            raise ValueError("Observable-token conditioning dimensions must be positive")
        self.model_dim = int(model_dim)
        self.observable = canonical_observable_name(observable)
        self.transform = observable_transform_name(self.observable)
        self.transform_delta = float(transform_delta)
        self.encoder = nn.Sequential(
            nn.Linear(8, int(hidden_width)),
            nn.SiLU(),
            nn.Linear(int(hidden_width), self.model_dim),
            nn.SiLU(),
        )
        self.observable_type_embedding = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.observable_type_embedding, std=0.02)
        self.register_buffer(
            "observable_mean",
            torch.zeros(num_geometries, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "observable_std",
            torch.ones(num_geometries, dtype=torch.float32),
            persistent=True,
        )


    def forward(
        self, context: ObservableConditioningContext
    ) -> torch.Tensor:
        context = context.select(self.observable)
        target = torch.as_tensor(
            context.target, device=self.observable_mean.device
        ).reshape(-1)
        geometry_id = torch.as_tensor(
            context.geometry_id, device=target.device
        ).long().reshape(-1)
        active = torch.as_tensor(context.active, device=target.device).bool().reshape(-1)
        batch = target.shape[0]
        if any(
            tensor.shape[0] != batch
            for tensor in (geometry_id, active)
        ):
            raise ValueError("Observable conditioning tensors must have one value per row")
        if bool(
            ((geometry_id < 0) | (geometry_id >= self.observable_mean.numel())).any()
        ):
            raise ValueError("Observable conditioning contains an invalid geometry_id")
        dtype = self.observable_mean.dtype
        target = target.to(dtype)
        mean = self.observable_mean[geometry_id]
        std = self.observable_std[geometry_id]
        active_target = target[active]
        if not bool(torch.isfinite(active_target).all()):
            raise ValueError(f"Active {self.observable} targets must be finite")
        if self.transform == "identity":
            transformed_target = torch.where(active, target, mean)
        elif self.transform == "log":
            if bool((active_target <= 0.0).any()):
                raise ValueError("Physical g_s targets must be positive")
            safe_target = torch.where(active, target, torch.ones_like(target))
            transformed_target = torch.log(safe_target)
        else:
            if bool((active_target < 0.0).any()):
                raise ValueError("Physical W0_abs targets must be non-negative")
            safe_target = torch.where(active, target, torch.zeros_like(target))
            transformed_target = torch.log(safe_target + self.transform_delta)
        standardized = (transformed_target - mean) / std
        zeros = torch.zeros_like(standardized)
        features = torch.stack(
            (
                standardized,
                torch.sin(math.pi * standardized),
                torch.cos(math.pi * standardized),
                torch.sin(2.0 * math.pi * standardized),
                torch.cos(2.0 * math.pi * standardized),
                zeros,
                zeros,
                zeros,
            ),
            dim=-1,
        )
        representation = self.encoder(features) + self.observable_type_embedding
        return representation * active.to(representation.dtype).unsqueeze(1)


class MultiObservableTokenConditioner(nn.Module):
    """Encode a fixed ordered collection of observables as separate tokens."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_width: int,
        num_geometries: int,
        observables: tuple[str, ...],
        transform_delta: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.observables = tuple(
            canonical_observable_name(observable) for observable in observables
        )
        if len(self.observables) < 2:
            raise ValueError("Multi-observable conditioning requires at least two observables")
        if len(set(self.observables)) != len(self.observables):
            raise ValueError("Conditioned observables must be unique")
        self.model_dim = int(model_dim)
        self.conditioners = nn.ModuleDict(
            {
                observable: ObservableTokenConditioner(
                    model_dim=model_dim,
                    hidden_width=hidden_width,
                    num_geometries=num_geometries,
                    observable=observable,
                    transform_delta=transform_delta,
                )
                for observable in self.observables
            }
        )


    def forward(
        self, context: ObservableConditioningContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        active_masks: list[torch.Tensor] = []
        if context.is_joint:
            contexts = [context.select(observable) for observable in self.observables]
        else:
            # Backward-compatible scalar APIs target the first configured
            # observable; all remaining joint-token slots stay inactive.
            first_target = torch.as_tensor(context.target)
            first_active = torch.as_tensor(context.active).bool()
            contexts = [context]
            contexts.extend(
                ObservableConditioningContext(
                    target=torch.zeros_like(first_target),
                    geometry_id=context.geometry_id,
                    active=torch.zeros_like(first_active),
                )
                for _ in self.observables[1:]
            )
        for observable, selected in zip(self.observables, contexts, strict=True):
            tokens.append(self.conditioners[observable](selected))
            active_masks.append(
                torch.as_tensor(selected.active, device=tokens[-1].device)
                .bool()
                .reshape(-1)
            )
        return torch.stack(tokens, dim=1), torch.stack(active_masks, dim=1)


ObservableConditioner = ObservableTokenConditioner


__all__ = [
    "canonical_observable_name",
    "MultiObservableTokenConditioner",
    "ObservableConditioner",
    "ObservableConditioningContext",
    "ObservableTokenConditioner",
    "SUPPORTED_CONDITIONAL_OBSERVABLES",
    "observable_transform_name",
]
