# Output map

## Fast exact reproduction

`python scripts/reproduce_paper.py`

creates:

```text
artifacts/paper/
├── figures/main/
├── figures/supplement/
├── tables/main/
├── tables/supplement/
├── data/
├── RESULTS.json
└── MANIFEST_SHA256.json
```

These are the exact frozen artifacts and result tables associated with the submitted manuscript.

## Full experiment

`python scripts/run_full_pipeline.py`

creates raw and aggregated results under:

```text
outputs/full_experiment/
├── sample_cache/
├── embedding_cache/
├── seed_results/
├── metrics/
├── metric_csv/
├── correlations/
└── analysis/
```

`python scripts/run_paper_analyses.py` then creates only the paper-focused analyses under:

```text
artifacts/generated/
├── within_model/
├── cross_model/
├── spectral/
│   ├── figures/
│   ├── tables/
│   ├── data/
│   └── metadata/
├── adaptive_precision/
│   ├── figures/
│   ├── tables/
│   ├── data/
│   └── metadata/
└── runtime/
```
