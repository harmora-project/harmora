from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .encoder import EncoderConfig, LayerwiseEncoder, array_hash
from .io_utils import safe_name
from .models import ModelSpec
from .sampling import stable_hash


def cache_path(output_dir: str | Path, model_alias: str, task_name: str) -> Path:
    return Path(output_dir) / "embedding_cache" / safe_name(model_alias) / f"{safe_name(task_name)}.npz"


def encoder_fingerprint(model_spec: ModelSpec, encoder_cfg: EncoderConfig) -> str:
    # Device and batch size do not change the intended representation semantics.
    return stable_hash({
        "alias": model_spec.alias,
        "hf_name": model_spec.hf_name,
        "trust_remote_code": bool(model_spec.trust_remote_code),
        "pooling": model_spec.pooling,
        "prompt": model_spec.prompt,
        "max_length": int(encoder_cfg.max_length),
        "dtype": str(encoder_cfg.dtype),
        "include_embedding_layer": bool(encoder_cfg.include_embedding_layer),
    })


def _read_scalar(archive: Any, key: str) -> str:
    value = archive[key]
    if np.asarray(value).shape == ():
        return str(np.asarray(value).item())
    return str(np.asarray(value).reshape(-1)[0])


def load_embedding_cache(
    output_dir: str | Path,
    model_alias: str,
    task_name: str,
    expected_sample_hash: str,
    expected_encoder_fingerprint: str,
) -> Dict[str, Any] | None:
    path = cache_path(output_dir, model_alias, task_name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        sample_hash = _read_scalar(archive, "sample_hash")
        stored_fingerprint = _read_scalar(archive, "encoder_fingerprint") if "encoder_fingerprint" in archive.files else ""
        if sample_hash != expected_sample_hash:
            raise RuntimeError(
                f"Embedding cache sample hash mismatch for {model_alias}/{task_name}: "
                f"cache={sample_hash}, expected={expected_sample_hash}."
            )
        if stored_fingerprint != expected_encoder_fingerprint:
            raise RuntimeError(
                f"Embedding cache encoder/model settings mismatch for {model_alias}/{task_name}. "
                "Rerun with --overwrite-embeddings and --overwrite-results."
            )
        payload: Dict[str, Any] = {
            "sample_hash": sample_hash,
            "embedding_hash": _read_scalar(archive, "embedding_hash"),
            "encoder_fingerprint": stored_fingerprint,
            "num_layers": int(np.asarray(archive["num_layers"]).item()),
        }
        for key in ["embeddings", "embeddings_a", "embeddings_b"]:
            if key in archive.files:
                payload[key] = np.asarray(archive[key], dtype=np.float32)
        return payload


def build_or_load_embeddings(
    output_dir: str | Path,
    task_payload: Dict[str, Any],
    model_spec: ModelSpec,
    encoder_cfg: EncoderConfig,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    path = cache_path(output_dir, model_spec.alias, task_payload["task"])
    fingerprint = encoder_fingerprint(model_spec, encoder_cfg)
    if not overwrite:
        cached = load_embedding_cache(
            output_dir,
            model_spec.alias,
            task_payload["task"],
            task_payload["sample_hash"],
            fingerprint,
        )
        if cached is not None:
            return cached

    encoder = LayerwiseEncoder(model_spec, encoder_cfg)
    family = task_payload["probe_family"]
    arrays: Dict[str, np.ndarray]
    if family in {"classification", "clustering"}:
        arrays = {"embeddings": encoder.encode(task_payload["texts"], show_progress=show_progress)}
    elif family in {"pair_classification", "sts"}:
        arrays = {
            "embeddings_a": encoder.encode(task_payload["sentence1"], show_progress=show_progress),
            "embeddings_b": encoder.encode(task_payload["sentence2"], show_progress=show_progress),
        }
        if arrays["embeddings_a"].shape != arrays["embeddings_b"].shape:
            raise RuntimeError(
                f"Pair embedding shape mismatch for {model_spec.alias}/{task_payload['task']}: "
                f"{arrays['embeddings_a'].shape} vs {arrays['embeddings_b'].shape}"
            )
    else:
        raise ValueError(f"Unsupported family: {family}")

    all_arrays = [arrays[key] for key in sorted(arrays)]
    embedding_hash = array_hash(*all_arrays)
    num_layers = int(all_arrays[0].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        sample_hash=np.asarray(task_payload["sample_hash"]),
        embedding_hash=np.asarray(embedding_hash),
        encoder_fingerprint=np.asarray(fingerprint),
        num_layers=np.asarray(num_layers),
    )
    return {
        **arrays,
        "sample_hash": task_payload["sample_hash"],
        "embedding_hash": embedding_hash,
        "encoder_fingerprint": fingerprint,
        "num_layers": num_layers,
    }
