from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .models import ModelSpec


@dataclass(frozen=True)
class EncoderConfig:
    max_length: int = 256
    batch_size: int = 16
    device: str = "cpu"
    dtype: str = "float32"
    include_embedding_layer: bool = True
    model_cache_dir: str | None = None


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last = attention_mask.sum(dim=1).clamp_min(1) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, last]


def array_hash(*arrays: np.ndarray, length: int = 20) -> str:
    """Hash arrays without materializing a second full-size bytes object.

    Chunked updates preserve the exact SHA-256 byte stream used by the previous
    ``arr.tobytes()`` implementation while keeping peak memory bounded.
    """
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024 * 1024
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        raw = memoryview(arr).cast("B")
        for offset in range(0, len(raw), chunk_bytes):
            digest.update(raw[offset : offset + chunk_bytes])
    return digest.hexdigest()[:length]


class LayerwiseEncoder:
    def __init__(self, spec: ModelSpec, cfg: EncoderConfig):
        self.spec = spec
        self.cfg = cfg
        cache_dir = cfg.model_cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_name,
            trust_remote_code=spec.trust_remote_code,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            spec.hf_name,
            trust_remote_code=spec.trust_remote_code,
            cache_dir=cache_dir,
            output_hidden_states=True,
        )
        self.model.to(cfg.device)
        self.model.eval()

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooling = self.spec.pooling.lower()
        if pooling == "mean":
            return _mean_pool(hidden, attention_mask)
        if pooling == "cls":
            return hidden[:, 0]
        if pooling == "last_token":
            return _last_token_pool(hidden, attention_mask)
        raise ValueError(f"Unsupported pooling mode: {self.spec.pooling}")

    def encode(self, texts: Iterable[str], show_progress: bool = True) -> np.ndarray:
        """Encode all valid layers with bounded host-memory usage.

        The previous implementation retained one CPU tensor per layer *per batch*
        and concatenated them only after the full collection had been encoded. On
        long augmentation runs this caused allocator fragmentation and hundreds of
        gigabytes of committed memory.

        This implementation instead:
          1. uses fixed-length padding so every forward pass reuses the same tensor
             shapes (important for the Windows/PyTorch allocator),
          2. preallocates the final float32 NumPy array after the first batch, and
          3. writes each pooled batch directly into its final slice.

        The representation semantics are unchanged: padding tokens remain masked,
        the same valid model layers are retained, and output ordering is identical.
        """
        values = [str(x) for x in texts]
        if self.spec.prompt:
            values = [self.spec.prompt + text for text in values]
        if not values:
            raise ValueError("Cannot encode an empty text collection.")

        batch_starts = range(0, len(values), self.cfg.batch_size)
        iterator = tqdm(
            batch_starts,
            total=(len(values) + self.cfg.batch_size - 1) // self.cfg.batch_size,
            desc=f"Encoding {self.spec.alias}",
            disable=not show_progress,
        )

        result: np.ndarray | None = None
        expected_layers: int | None = None

        with torch.inference_mode():
            for batch_number, start in enumerate(iterator, start=1):
                stop = min(start + self.cfg.batch_size, len(values))
                batch = values[start:stop]

                # Fixed-shape padding avoids the severe CPU allocator fragmentation
                # seen with a different longest sequence length in every batch. The
                # attention mask ensures padded tokens do not affect pooling.
                tokens = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                ).to(self.cfg.device)

                output = self.model(
                    **tokens,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = output.hidden_states

                # The raw embedding-layer CLS vector is input-independent before
                # any Transformer block. It is therefore not a valid candidate
                # representation for models using CLS pooling.
                drop_raw_embedding_layer = (
                    not self.cfg.include_embedding_layer
                    or self.spec.pooling.lower() == "cls"
                )
                if drop_raw_embedding_layer and len(hidden_states) > 1:
                    hidden_states = hidden_states[1:]

                if expected_layers is None:
                    expected_layers = len(hidden_states)
                elif len(hidden_states) != expected_layers:
                    raise RuntimeError(
                        "Model returned an inconsistent number of hidden layers "
                        "across batches."
                    )

                for layer_index, layer in enumerate(hidden_states):
                    pooled_cpu = (
                        self._pool(layer, tokens["attention_mask"])
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )

                    if result is None:
                        result = np.empty(
                            (len(hidden_states), len(values), pooled_cpu.shape[-1]),
                            dtype=np.float32,
                        )

                    result[layer_index, start:stop, :] = pooled_cpu.numpy()
                    del pooled_cpu

                # Release the large token-level hidden-state tuple immediately.
                # Periodic GC is deliberately infrequent to avoid slowing every
                # batch while still breaking reference cycles in remote-code models.
                del hidden_states, output, tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_number % 8 == 0:
                    gc.collect()

        gc.collect()
        if result is None:
            raise RuntimeError("No embeddings were generated.")
        if result.shape[1] != len(values):
            raise RuntimeError(
                f"Embedding/sample mismatch: embeddings={result.shape[1]}, texts={len(values)}"
            )
        return result
