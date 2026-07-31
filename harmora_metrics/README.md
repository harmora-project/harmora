# Harmora Metrics Package

A small modular package for unsupervised layer-wise representation evaluation.

## Canonical input shapes

Single-view metrics:

```python
hidden_states.shape == [L, N, D]
```

Augmentation-based metrics:

```python
augmented_states.shape == [L, N, A, D]
```

where:

- `L`: number of layers
- `N`: number of samples
- `D`: representation dimension
- `A`: number of augmentations/views

## Implemented metrics

### Single-view

1. `harmora`
2. `matrix_entropy`
3. `participation_ratio`
4. `anisotropy`
5. `intrinsic_dimension`
6. `curvature`
7. `spectral_gap`

### Augmentation-based

8. `infonce`
9. `dime`
10. `lidar`

## Quick usage

```python
from metrics import MetricConfig, compute_all_metrics

config = MetricConfig(
    harmora_sigma_l2=1.0,
    harmora_K_l=None,
    graph_bandwidth="median",
    graph_k_nn=None,
    entropy_alpha=1.0,
    entropy_normalizations=("maxEntropy",),
    infonce_temperature=0.1,
)

results = compute_all_metrics(
    hidden_states=hidden_states,          # [L, N, D]
    augmented_states=augmented_states,    # [L, N, A, D], optional
    config=config,
)

print(results["harmora"]["score"])
print(results["participation_ratio"]["normalized_rank"])
print(results["anisotropy"]["anisotropy"])
print(results["spectral_gap"]["lambda2"])
```

## Select a subset

```python
results = compute_all_metrics(
    hidden_states=hidden_states,
    metrics=["harmora", "participation_ratio", "anisotropy", "spectral_gap"],
)
```

## Notes

- `anisotropy` follows the average random-pair cosine similarity convention used in contextual embedding geometry analysis. It does not mean-center representations, because the common mean direction is part of the anisotropy signal.
- `curvature` is meaningful when sample order has structure, such as sequence, trajectory, or time-series order.
- `dime`, `matrix_entropy`, and `lidar` require `repitl`.
- `intrinsic_dimension` requires `dadapy`.
