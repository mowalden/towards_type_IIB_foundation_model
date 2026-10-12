"""Seeded sequential flux generation."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch

from ..data import DirectIntegerTokenizer
from ..masking import (
    canonical_mask_schedule,
    canonical_time_conditioning,
    time_condition_from_mask_fraction,
    timestep_from_mask_fraction,
)
from ..model.observable_conditioning import (
    ObservableConditioningContext,
    canonical_observable_name,
)


def classifier_free_guidance_logits(
    conditional_logits: torch.Tensor,
    unconditional_logits: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine conditional and null-path logits using the standard CFG rule."""

    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale must be a finite non-negative number")
    if conditional_logits.shape != unconditional_logits.shape:
        raise ValueError("Conditional and unconditional CFG logits must have one shape")
    return unconditional_logits + scale * (
        conditional_logits - unconditional_logits
    )


class StructuredFluxSampler:
    """Sequential masked sampling with seeded random-coordinate ordering."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_metadata: dict[str, Any],
        geometry_metadata: dict[str, Any],
        *,
        num_timesteps: int | None = 200,
        mask_schedule: str = "sine",
        time_conditioning: str = "discrete",
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        device: str | torch.device | None = None,
    ) -> None:
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not math.isfinite(float(guidance_scale)) or float(guidance_scale) < 0.0:
            raise ValueError("guidance_scale must be a finite non-negative number")
        self.model = model
        self.tokenizer = DirectIntegerTokenizer.from_metadata(tokenizer_metadata)
        self.geometry_by_id = {
            int(row["geometry_id"]): row for row in geometry_metadata["geometries"]
        }
        self.time_conditioning = canonical_time_conditioning(time_conditioning)
        self.num_timesteps = (
            None if num_timesteps is None else int(num_timesteps)
        )
        self.mask_schedule = canonical_mask_schedule(mask_schedule)
        if self.time_conditioning != "discrete":
            if self.num_timesteps is not None:
                raise ValueError(
                    "Continuous or absent time conditioning does not use "
                    "num_timesteps"
                )
            if self.mask_schedule != "linear":
                raise ValueError(
                    "Continuous or absent time conditioning requires "
                    "mask_schedule=linear"
                )
        elif self.num_timesteps is None or self.num_timesteps <= 0:
            raise ValueError("Discrete time conditioning requires num_timesteps")
        model_time_conditioning = canonical_time_conditioning(
            getattr(getattr(model, "config", None), "time_conditioning", "discrete")
        )
        if model_time_conditioning != self.time_conditioning:
            raise ValueError(
                "Sampler time conditioning does not match the model: "
                f"model={model_time_conditioning!r}, "
                f"sampler={self.time_conditioning!r}"
            )
        self.temperature = float(temperature)
        self.guidance_scale = float(guidance_scale)
        self.device = torch.device(device or next(model.parameters()).device)

    def _action_log_probabilities(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        *,
        observable_context: ObservableConditioningContext | None = None,
        time_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the deployed categorical policy at every flux coordinate."""

        if time_condition is None:
            masked_count = (
                (tokens == self.tokenizer.mask_token_id) & active_mask
            ).sum(1)
            time_condition = time_condition_from_mask_fraction(
                masked_count,
                active_mask.sum(1),
                time_conditioning=self.time_conditioning,
                num_timesteps=self.num_timesteps,
                mask_schedule=self.mask_schedule,
            )
        if observable_context is None:
            conditional_hidden = self.model.flux_hidden(
                tokens, active_mask, h11, h12, time_condition
            )
        else:
            conditional_hidden = self.model.flux_hidden(
                tokens,
                active_mask,
                h11,
                h12,
                time_condition,
                observable_context=observable_context,
            )
        action_logits = self.model.flux_head(conditional_hidden).float()
        if observable_context is not None and self.guidance_scale != 1.0:
            unconditional_hidden = self.model.flux_hidden(
                tokens, active_mask, h11, h12, time_condition
            )
            unconditional_logits = self.model.flux_head(unconditional_hidden).float()
            action_logits = classifier_free_guidance_logits(
                action_logits,
                unconditional_logits,
                self.guidance_scale,
            )
        return torch.log_softmax(action_logits / self.temperature, dim=-1)

    def _sample_action(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        position: torch.Tensor,
        generator: torch.Generator,
        *,
        observable_context: ObservableConditioningContext | None = None,
    ) -> torch.Tensor:
        log_probabilities = self._action_log_probabilities(
            tokens,
            active_mask,
            h11,
            h12,
            observable_context=observable_context,
        )
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return torch.multinomial(
            log_probabilities[rows, position].exp(), 1, generator=generator
        ).squeeze(1)

    def _coordinate_order(
        self,
        *,
        flux_dim: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        return torch.stack(
            [
                torch.randperm(
                    flux_dim,
                    generator=generator,
                    device=self.device,
                )
                for _ in range(batch_size)
            ]
        )

    def _sample_step(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        position: torch.Tensor | None,
        generator: torch.Generator,
        *,
        observable_context: ObservableConditioningContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position is None:
            raise RuntimeError("Random-coordinate sampling requires a coordinate order")
        action = self._sample_action(
            tokens,
            active_mask,
            h11,
            h12,
            position,
            generator,
            observable_context=observable_context,
        )
        return position, action

    def _fill_tokens(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        *,
        flux_dim: int,
        generator: torch.Generator,
        observable_context: ObservableConditioningContext | None,
        progress: Callable[[int, int], None] | None,
    ) -> None:
        """Fill active masks according to this sampler's reveal policy."""

        batch_size = int(tokens.shape[0])
        order = self._coordinate_order(
            flux_dim=flux_dim,
            batch_size=batch_size,
            generator=generator,
        )
        rows = torch.arange(batch_size, device=self.device)

        for step in range(flux_dim):
            position = None if order is None else order[:, step]
            selected_position, action = self._sample_step(
                tokens,
                active_mask,
                h11,
                h12,
                position,
                generator,
                observable_context=observable_context,
            )
            tokens[rows, selected_position] = action
            if progress is not None:
                progress(step + 1, flux_dim)

    @torch.no_grad()
    def sample_flux(
        self,
        *,
        geometry_id: int,
        batch_size: int,
        seed: int,
        predict_initializer: bool = True,
        progress: Callable[[int, int], None] | None = None,
        observable_target: float | torch.Tensor | None = None,
        observable_targets: Mapping[str, float | torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        geometry = self.geometry_by_id[int(geometry_id)]
        h11_value = int(geometry["h11"])
        h12_value = int(geometry["h12"])
        flux_dim = 4 * (h12_value + 1)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        active = torch.zeros(
            (batch_size, self.tokenizer.max_flux_dim),
            dtype=torch.bool,
            device=self.device,
        )
        active[:, :flux_dim] = True
        tokens = torch.full(
            active.shape,
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        tokens[active] = self.tokenizer.mask_token_id
        geometry_ids = torch.full(
            (batch_size,), int(geometry_id), dtype=torch.long, device=self.device
        )
        h11 = torch.full(
            (batch_size,), h11_value, dtype=torch.long, device=self.device
        )
        h12 = torch.full(
            (batch_size,), h12_value, dtype=torch.long, device=self.device
        )
        if observable_target is not None and observable_targets is not None:
            raise ValueError(
                "Choose either observable_target or observable_targets, not both"
            )
        configured_observables = tuple(
            getattr(
                self.model.config,
                "conditioned_observables",
                (getattr(self.model.config, "conditional_observable", "N_flux"),),
            )
        )
        requested_targets: dict[str, float | torch.Tensor] = {}
        if observable_target is not None:
            requested_targets[configured_observables[0]] = observable_target
        elif observable_targets is not None:
            for name, value in observable_targets.items():
                canonical = canonical_observable_name(name)
                if canonical in requested_targets:
                    raise ValueError(f"Duplicate observable target {name!r}")
                requested_targets[canonical] = value
        unknown = set(requested_targets) - set(configured_observables)
        if unknown:
            raise ValueError(
                "Targets were supplied for unconfigured observable(s): "
                + ", ".join(sorted(unknown))
            )
        if requested_targets and self.model.observable_conditioner is None:
            raise ValueError("This model has no native observable conditioner")

        target_tensors: dict[str, torch.Tensor] = {}
        for observable, target in requested_targets.items():
            target_tensor = torch.as_tensor(
                target, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if target_tensor.numel() == 1:
                target_tensor = target_tensor.expand(batch_size).clone()
            if target_tensor.shape != (batch_size,):
                raise ValueError(
                    "observable target must be scalar or have shape [batch_size]"
                )
            if not bool(torch.isfinite(target_tensor).all()):
                raise ValueError("observable target must be finite")
            if observable == "g_s" and bool(
                (target_tensor <= 0.0).any()
            ):
                raise ValueError("g_s targets must be strictly positive")
            if observable == "W0_abs" and bool(
                (target_tensor < 0.0).any()
            ):
                raise ValueError("W0_abs targets must be non-negative")
            target_tensors[observable] = target_tensor

        context: ObservableConditioningContext | None = None
        if target_tensors:
            if len(configured_observables) == 1:
                target_tensor = target_tensors[configured_observables[0]]
                context = ObservableConditioningContext(
                    target=target_tensor,
                    geometry_id=geometry_ids,
                    active=torch.ones_like(target_tensor, dtype=torch.bool),
                )
            else:
                targets = {
                    observable: target_tensors.get(
                        observable,
                        torch.zeros(batch_size, device=self.device, dtype=torch.float32),
                    )
                    for observable in configured_observables
                }
                active_targets = {
                    observable: torch.full(
                        (batch_size,),
                        observable in target_tensors,
                        device=self.device,
                        dtype=torch.bool,
                    )
                    for observable in configured_observables
                }
                context = ObservableConditioningContext(
                    target=targets,
                    geometry_id=geometry_ids,
                    active=active_targets,
                )
        self._fill_tokens(
            tokens,
            active,
            h11,
            h12,
            flux_dim=flux_dim,
            generator=generator,
            observable_context=context,
            progress=progress,
        )
        if bool(((tokens == self.tokenizer.mask_token_id) & active).any()):
            raise RuntimeError("Structured fill ended with masked active positions")

        result: dict[str, Any] = {
            "tokens": tokens,
            "active_mask": active,
            "flux": tokens[:, :flux_dim].to(torch.int64) + int(self.tokenizer.min_value),
        }
        if predict_initializer:
            result["initializer"] = self.model.predict_initializer(
                tokens, active, h11, h12, geometry_ids
            )
        return result


class ConfidenceOrderFluxSampler(StructuredFluxSampler):
    """Sequential sampling that commits the most confident coordinate next."""

    def _coordinate_order(
        self,
        *,
        flux_dim: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> None:
        del flux_dim, batch_size, generator
        return None

    def _sample_step(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        position: torch.Tensor | None,
        generator: torch.Generator,
        *,
        observable_context: ObservableConditioningContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position
        log_probabilities = self._action_log_probabilities(
            tokens,
            active_mask,
            h11,
            h12,
            observable_context=observable_context,
        )
        masked_positions = (
            (tokens == self.tokenizer.mask_token_id) & active_mask.bool()
        )
        coordinate_confidence = log_probabilities.amax(dim=-1).masked_fill(
            ~masked_positions, -torch.inf
        )
        selected_position = coordinate_confidence.argmax(dim=1)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        action = torch.multinomial(
            log_probabilities[rows, selected_position].exp(),
            1,
            generator=generator,
        ).squeeze(1)
        return selected_position, action


class MaskGITFluxSampler(StructuredFluxSampler):
    """Cosine-scheduled parallel MaskGIT decoding with sampled confidence."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_metadata: dict[str, Any],
        geometry_metadata: dict[str, Any],
        *,
        steps: int = 32,
        num_timesteps: int | None = 200,
        mask_schedule: str = "sine",
        time_conditioning: str = "discrete",
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        device: str | torch.device | None = None,
    ) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("MaskGIT steps must be a positive integer")
        super().__init__(
            model,
            tokenizer_metadata,
            geometry_metadata,
            num_timesteps=num_timesteps,
            mask_schedule=mask_schedule,
            time_conditioning=time_conditioning,
            temperature=temperature,
            guidance_scale=guidance_scale,
            device=device,
        )
        self.steps = int(steps)

    def _fill_tokens(
        self,
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        h11: torch.Tensor,
        h12: torch.Tensor,
        *,
        flux_dim: int,
        generator: torch.Generator,
        observable_context: ObservableConditioningContext | None,
        progress: Callable[[int, int], None] | None,
    ) -> None:
        batch_size = int(tokens.shape[0])
        for step in range(1, self.steps + 1):
            masked = (
                (tokens == self.tokenizer.mask_token_id) & active_mask.bool()
            )
            current = masked.sum(dim=1)
            if int(current.max()) == 0:
                break

            time_condition = time_condition_from_mask_fraction(
                current,
                active_mask.sum(1),
                time_conditioning=self.time_conditioning,
                num_timesteps=self.num_timesteps,
                mask_schedule=self.mask_schedule,
            )
            log_probabilities = self._action_log_probabilities(
                tokens,
                active_mask,
                h11,
                h12,
                observable_context=observable_context,
                time_condition=time_condition,
            )
            proposal = torch.multinomial(
                log_probabilities.exp().reshape(-1, log_probabilities.shape[-1]),
                1,
                generator=generator,
            ).view_as(tokens)
            proposal_confidence = log_probabilities.gather(
                -1, proposal.unsqueeze(-1)
            ).squeeze(-1)
            proposal_confidence = proposal_confidence.masked_fill(
                ~masked, -torch.inf
            )

            target_masked = (
                0
                if step == self.steps
                else int(
                    math.floor(
                        math.cos(0.5 * math.pi * step / self.steps) * flux_dim
                    )
                )
            )
            for row in range(batch_size):
                row_masked = int(current[row])
                if row_masked == 0:
                    continue
                number = min(row_masked, max(1, row_masked - target_masked))
                positions = torch.topk(
                    proposal_confidence[row], k=number
                ).indices
                tokens[row, positions] = proposal[row, positions]

            if progress is not None:
                remaining = int(
                    (
                        (tokens == self.tokenizer.mask_token_id)
                        & active_mask.bool()
                    )
                    .sum(dim=1)
                    .max()
                )
                progress(flux_dim - remaining, flux_dim)
