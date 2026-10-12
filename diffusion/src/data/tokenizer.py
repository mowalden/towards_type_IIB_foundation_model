from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DirectIntegerTokenizer:
    min_value: int
    max_value: int
    max_flux_dim: int = 24

    def __post_init__(self) -> None:
        if int(self.min_value) > int(self.max_value):
            raise ValueError("min_value must not exceed max_value")
        if int(self.max_flux_dim) <= 0:
            raise ValueError("max_flux_dim must be positive")

    @property
    def num_value_tokens(self) -> int:
        return int(self.max_value) - int(self.min_value) + 1

    @property
    def pad_token_id(self) -> int:
        return self.num_value_tokens

    @property
    def mask_token_id(self) -> int:
        return self.num_value_tokens + 1

    @property
    def input_vocab_size(self) -> int:
        return self.num_value_tokens + 2

    @property
    def output_vocab_size(self) -> int:
        return self.num_value_tokens

    def encode_batch(self, fluxes: np.ndarray, flux_dim: int) -> tuple[np.ndarray, np.ndarray]:
        fluxes = np.asarray(fluxes)
        if fluxes.ndim != 2 or fluxes.shape[1] != int(flux_dim):
            raise ValueError(f"Expected fluxes shape (N, {flux_dim}), got {fluxes.shape}")
        rounded = np.rint(fluxes)
        if not np.allclose(fluxes, rounded, atol=1e-6, rtol=0.0):
            raise ValueError("Flux array contains non-integer values")
        values = rounded.astype(np.int64)
        if values.size and (values.min() < self.min_value or values.max() > self.max_value):
            raise ValueError("Flux values exceed the tokenizer range")
        tokens = np.full(
            (values.shape[0], self.max_flux_dim),
            self.pad_token_id,
            dtype=np.int32,
        )
        active = np.zeros_like(tokens, dtype=np.uint8)
        tokens[:, : int(flux_dim)] = (values - int(self.min_value)).astype(np.int32)
        active[:, : int(flux_dim)] = 1
        return tokens, active

    def decode_batch(self, tokens: np.ndarray, flux_dim: int) -> np.ndarray:
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] != self.max_flux_dim:
            raise ValueError(
                f"Expected token shape (N, {self.max_flux_dim}), got {tokens.shape}"
            )
        active = tokens[:, : int(flux_dim)]
        if np.any(active < 0) or np.any(active >= self.num_value_tokens):
            raise ValueError("Active positions contain PAD, MASK, or invalid IDs")
        return (active.astype(np.int64) + int(self.min_value)).astype(np.int32)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "direct_integer",
            "format_version": 1,
            "min_value": int(self.min_value),
            "max_value": int(self.max_value),
            "num_value_tokens": self.num_value_tokens,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size,
            "max_flux_dim": int(self.max_flux_dim),
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "DirectIntegerTokenizer":
        tokenizer = cls(
            min_value=int(payload["min_value"]),
            max_value=int(payload["max_value"]),
            max_flux_dim=int(payload.get("max_flux_dim", 24)),
        )
        expected = tokenizer.to_metadata()
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Inconsistent tokenizer metadata field {key}")
        return tokenizer
