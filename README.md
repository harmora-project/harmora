# Harmora

Official reproducibility package for:

> **Harmora: Label-Free Selection of Representations in Language Embedding Models via Laplacian Harmonics**

This repository is intentionally focused on the experiments and artifacts used in the paper. Historical backups, obsolete generalization workflows, duplicate scripts, intermediate debugging outputs, and unused figures have been removed.

## What is included

- the locked **7-model × 11-task × 4-seed** experiment configuration;
- Harmora and the eight label-free baselines used in the paper;
- layer-wise downstream evaluation and metric extraction;
- within-model and cross-model selection analyses;
- bandwidth and spectral-depth analyses;
- adaptive-precision ablation;
- controlled runtime and eigensolver validation;
- exact submitted figures, tables, and frozen result CSV files;
- unit tests, integrity checks, and SHA-256 artifact manifests.

## Repository layout

```text
harmora-paper-code/
├── configs/
│   └── paper.yaml                 # locked paper experiment
├── src/harmora_downstream/        # core data, encoding, metrics, evaluation
├── harmora_metrics/metrics/       # Harmora + baseline metric implementations
├── scripts/
│   ├── check_environment.py
│   ├── run_full_pipeline.py
│   ├── run_paper_analyses.py
│   ├── reproduce_paper.py
│   ├── validate_reproduction.py
│   └── run_smoke_test.py
├── experiments/                   # focused paper-analysis generators
├── reference/                     # exact paper data, figures, and LaTeX tables
├── paper/                         # manuscript, supplement, checklist
├── tests/
├── docs/
└── artifacts/                     # generated locally; ignored by Git
```

---

# 1. Fastest path: reproduce the submitted paper artifacts

This path does **not** download models or datasets. It materializes the exact frozen figures, LaTeX tables, and result CSV files associated with the submitted manuscript.

```bash
python scripts/reproduce_paper.py
python scripts/validate_reproduction.py
```

Outputs:

```text
artifacts/paper/
```

The validation checks the experiment dimensions and the principal reported results, including:

- within-model Harmora Spearman: `0.432387`;
- strongest within-model baseline: `0.270963`;
- selected-candidate percentile: `0.845779`;
- cross-model NDCG@1: `0.843242`;
- cross-model NDCG@3: `0.795474`;
- Top-5 overlap: `0.372727`;
- exhaustive runtime: `762.298868 s`;
- Harmora-shortlist runtime: `166.995998 s`;
- speedup: `4.564773×`.

To recreate the artifact directory from scratch:

```bash
python scripts/reproduce_paper.py --force
```

---

# 2. Installation

Python 3.10 or newer is required. Python 3.11 is recommended.

## Virtual environment

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

A Conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate harmora
```

## Environment check

```bash
python scripts/check_environment.py
```

This verifies the locked model/task set, required libraries, device selection, and the bundled metric implementation.

---

# 3. Smoke test

Run this before a full experiment:

```bash
python scripts/run_smoke_test.py
```

It performs:

1. unit tests for configuration, sampling, deterministic splits, evaluators, aggregation, and metric computation;
2. exact paper-artifact materialization;
3. validation of the key paper results.

---

# 4. Full experiment rerun

The full rerun downloads the configured Hugging Face models and MTEB datasets. It caches task samples and embeddings, so interrupted runs can normally resume without repeating completed work.

```bash
python scripts/run_full_pipeline.py
```

The stages are:

1. `samples` — resolve the eleven tasks and build one fixed sample per task;
2. `downstream` — evaluate every model-layer candidate with four fixed seeds;
3. `metrics` — compute Harmora and eight baselines on the same cached representations;
4. `aggregate` — aggregate downstream utilities across seeds;
5. `correlate` — compute within-model and cross-model evaluation quantities.

Outputs are written to:

```text
outputs/full_experiment/
```

## Resume from a stage

```bash
python scripts/run_full_pipeline.py --from-stage metrics
```

Run only a range:

```bash
python scripts/run_full_pipeline.py --from-stage downstream --to-stage aggregate
```

Run a small subset for debugging:

```bash
python scripts/run_full_pipeline.py \
  --models minilm_l6 \
  --tasks Banking77Classification.v2 \
  --seeds 11 \
  --to-stage correlate
```

PowerShell equivalent:

```powershell
python scripts/run_full_pipeline.py `
  --models minilm_l6 `
  --tasks Banking77Classification.v2 `
  --seeds 11 `
  --to-stage correlate
```

Use `--overwrite` only when cached samples, embeddings, and completed results must be rebuilt.

---

# 5. Generate the paper analyses from a full rerun

After the core experiment finishes:

```bash
python scripts/run_paper_analyses.py
```

This generates only the analyses used in the manuscript:

- within-model ranking figure and tables;
- cross-model selection figure and tables;
- bandwidth and spectral-depth diagnostics;
- adaptive-precision ablation;
- runtime and eigensolver analyses.

Outputs:

```text
artifacts/generated/
```

Individual analyses can be selected:

```bash
python scripts/run_paper_analyses.py --only within cross spectral
python scripts/run_paper_analyses.py --only precision --device cuda
python scripts/run_paper_analyses.py --only runtime --skip-real-runtime
```

The exact submitted vector/raster assets remain under `reference/figures/`. Small visual differences can occur when plots are rerendered on another operating system because of fonts and Matplotlib backends.

---

# 6. Exact paper configuration

The complete configuration is in [`configs/paper.yaml`](configs/paper.yaml).

## Models

- MiniLM-L6
- MPNet-base
- E5-base-v2
- E5-large-v2
- BGE-base-en-v1.5
- BGE-large-en-v1.5
- Snowflake Arctic Embed M

## Tasks

- Classification: Banking77, Emotion, HUMEEmotion
- Clustering: ArXiv-P2P, ArXiv-S2S, Biorxiv-P2P
- Pair classification: LegalBenchPC
- STS: STS15, STS16, STSBenchmark, SICK-R

## Fixed settings

- downstream seeds: `11, 22, 33, 44`;
- shared sample seed: `2025`;
- maximum task sample size: `512`;
- Harmora bandwidth: `K=10`;
- Harmora variance: `sigma²=1`;
- candidate pool: `106` model-layer candidates per task;
- shortlist size: `5`.

The configuration loader rejects silent changes to the seven models or eleven tasks when `lock_exact_experiment_set: true`.

---

# 7. Reproducibility modes

## Exact artifact reproduction

Use:

```bash
python scripts/reproduce_paper.py
```

This is the appropriate command when the goal is to obtain the exact figures and tables used in the manuscript immediately.

## Independent computational rerun

Use:

```bash
python scripts/run_full_pipeline.py
python scripts/run_paper_analyses.py
```

This is the appropriate command when the goal is to independently recompute embeddings, downstream utilities, label-free scores, and analyses.

---

# 8. Data and cache policy

The repository does not commit downloaded datasets, model weights, or large embedding caches.

Local-only directories:

```text
cache/models/
cache/data/
outputs/full_experiment/
artifacts/generated/
```

These paths are excluded by `.gitignore`.

The frozen result files required to verify the paper are intentionally tracked under:

```text
reference/data/
```

---

# 9. Tests

```bash
PYTHONPATH=src pytest -q tests
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
pytest -q tests
```

---

# 10. Release validation

The packaged repository was checked with:

```bash
python -m compileall -q src harmora_metrics scripts experiments tests
PYTHONPATH=src pytest -q tests
python scripts/reproduce_paper.py --force
python scripts/validate_reproduction.py
python scripts/run_smoke_test.py
```

The lightweight release checks pass, and the within-model and cross-model
figure/table generators were also exercised against the frozen seed-level
reference inputs. See [`PACKAGE_VALIDATION.md`](PACKAGE_VALIDATION.md).

The complete seven-model, eleven-task run was not repeated while assembling
this archive because it requires external model/dataset downloads and
substantial compute. The commands and locked configuration for that independent
rerun are included above.

---

# 11. Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

The code is released under the MIT License. Dataset and pretrained-model licenses remain governed by their original providers.
