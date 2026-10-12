"""Bound evaluation sampling memory while preserving complete prompt results."""

from __future__ import annotations

from typing import Any, Callable

import torch


def concatenate_sample_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate CPU sample tensors and weight scalar means by batch size."""

    sizes = [len(batch["tokens"]) for batch in batches]

    def combine(values: list[Any]) -> Any:
        if isinstance(values[0], dict):
            return {key: combine([value[key] for value in values]) for key in values[0]}
        if values[0].ndim == 0:
            return sum(value * size for value, size in zip(values, sizes)) / sum(sizes)
        return torch.cat(values, dim=0)

    return combine(batches)


def sample_flux_in_batches(
    sampler: Any,
    *,
    num_samples: int,
    sampling_batch_size: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """Sample one prompt in bounded chunks, returning all outputs on CPU.

    Chunk zero keeps the prompt seed. Later chunks use a large seed stride so
    adjacent prompt seeds do not reuse the same random stream. Reproducibility
    requires both the seed and batch size to remain fixed. Progress counts
    completed sampling coordinates across all chunks.
    """

    for name, value in (("num_samples", num_samples), ("sampling_batch_size", sampling_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    num_batches = (num_samples + sampling_batch_size - 1) // sampling_batch_size
    batches = []

    def to_cpu(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        return value.detach().cpu()

    for batch_index, start in enumerate(range(0, num_samples, sampling_batch_size)):
        def update(completed: int, total: int) -> None:
            if progress is not None:
                progress(batch_index * total + completed, num_batches * total)

        generated = sampler.sample_flux(
            batch_size=min(sampling_batch_size, num_samples - start),
            seed=(seed + batch_index * 1_000_000_007) % (2**64),
            progress=update if progress is not None else None,
            **sample_kwargs,
        )
        batches.append(to_cpu(generated))
        del generated
    return concatenate_sample_batches(batches)
