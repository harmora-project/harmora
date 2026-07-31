# Output map

## Full experiment

Run:

```bash
python scripts/run_full_pipeline.py
```

The pipeline writes its outputs under:

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

## Paper analyses

After the full experiment finishes, run:

```bash
python scripts/run_paper_analyses.py
```

The analysis scripts write their outputs under:

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
