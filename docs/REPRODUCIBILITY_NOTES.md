# Reproducibility notes

- The full experiment downloads the seven configured Hugging Face models and eleven MTEB tasks, then computes embeddings, downstream utilities, label-free metrics, correlations, and paper analyses.
- Matplotlib output may differ slightly across operating systems because fonts and rendering backends can differ.
- The manuscript figures are stored in `reference/figures/`.
- Pair-classification and STS graphs use the concatenation of both endpoint representation matrices. Their downstream evaluators operate on paired endpoints.
- Target labels are used only for downstream evaluation. Harmora and the label-free baselines are computed from unlabeled representations.
