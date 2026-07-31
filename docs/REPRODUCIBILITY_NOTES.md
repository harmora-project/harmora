# Reproducibility notes

- The fast reproduction path uses frozen task-level and seed-level result files so that the submitted figures and LaTeX tables are available without downloading models or datasets.
- The full rerun path downloads the seven Hugging Face models and eleven MTEB tasks, then reconstructs embeddings, downstream utilities, label-free metrics, correlations, and paper analyses.
- Matplotlib output may differ slightly across operating systems because fonts and rendering backends differ. For exact manuscript assets, use `reference/figures/` or the files materialized by `scripts/reproduce_paper.py`.
- Pair-classification and STS graphs use the concatenation of both endpoint representation matrices; their downstream evaluators operate on paired endpoints.
- Target labels are used only for downstream evaluation. Harmora and all label-free baselines are computed from unlabeled representations.
