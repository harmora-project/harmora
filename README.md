# Harmora

Official implementation and reproducibility package for:

> **Harmora: Label-Free Selection of Representations in Language Embedding Models via Laplacian Harmonics**

This repository contains the code, experiment configurations, analyses, and reference results used in the paper.

## Main features

The repository includes:

- the fixed **7-model × 11-task × 4-seed** experiment configuration;
- the Harmora metric;
- eight label-free baseline metrics;
- layer-wise downstream evaluation;
- within-model representation selection;
- cross-model representation selection;
- bandwidth and spectral-depth analyses;
- adaptive-precision experiments;
- runtime and eigensolver analyses;
- reference figures, tables, and result CSV files;
- unit tests and reproducibility checks;
- SHA-256 artifact manifests.

## Repository structure

```text
harmora-paper-code/
├── configs/
│   └── paper.yaml                 # Main experiment configuration
├── src/harmora_downstream/        # Data, encoding, metrics, and evaluation
├── harmora_metrics/metrics/       # Harmora and baseline implementations
├── scripts/
│   ├── check_environment.py
│   ├── run_full_pipeline.py
│   ├── run_paper_analyses.py
│   ├── reproduce_paper.py
│   ├── validate_reproduction.py
│   └── run_smoke_test.py
├── experiments/                   # Paper analysis scripts
├── reference/                     # Reference data, figures, and LaTeX tables
├── paper/                         # Manuscript, supplement, and checklist
├── tests/                         # Unit tests
├── docs/                          # Documentation
└── artifacts/                     # Locally generated files
```

## Quick reproduction

To generate the reference figures, LaTeX tables, and result CSV files, run:

```bash
python scripts/reproduce_paper.py
python scripts/validate_reproduction.py
```

The generated files are written to:

```text
artifacts/paper/
```

This process does not download models or datasets.

To recreate the artifact directory and replace existing files, run:

```bash
python scripts/reproduce_paper.py --force
```

### Reference results

The validation script checks the experiment dimensions and the main reported results:

| Result | Value |
|---|---:|
| Within-model Harmora Spearman | `0.432387` |
| Strongest within-model baseline | `0.270963` |
| Selected-candidate percentile | `0.845779` |
| Cross-model NDCG@1 | `0.843242` |
| Cross-model NDCG@3 | `0.795474` |
| Top-5 overlap | `0.372727` |
| Exhaustive runtime | `762.298868 s` |
| Harmora-shortlist runtime | `166.995998 s` |
| Speedup | `4.564773×` |

## Installation

Python 3.10 or newer is required. Python 3.11 is recommended.

### Linux and macOS

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

### Windows PowerShell

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

### Development installation

To install the development and testing dependencies, run:

```bash
pip install -e ".[dev]"
```

### Conda installation

A Conda environment file is also available:

```bash
conda env create -f environment.yml
conda activate harmora
```

## Environment check

Before running the experiments, check the environment:

```bash
python scripts/check_environment.py
```

This script checks:

- the configured models and tasks;
- the required Python libraries;
- the selected computing device;
- the included metric implementations.

## Smoke test

Run the smoke test before starting the full experiment:

```bash
python scripts/run_smoke_test.py
```

The smoke test performs:

1. unit tests for configuration and sampling;
2. tests for deterministic data splits;
3. tests for evaluators and aggregation;
4. tests for metric computation;
5. reference artifact generation;
6. validation of the main paper results.

## Full experiment

The full experiment downloads the configured Hugging Face models and MTEB datasets.

Run the complete pipeline with:

```bash
python scripts/run_full_pipeline.py
```

The pipeline uses cached task samples and embeddings. An interrupted run can normally continue without repeating completed work.

### Pipeline stages

The pipeline contains five stages:

1. `samples`  
   Resolves the eleven tasks and creates one fixed sample for each task.

2. `downstream`  
   Evaluates every model-layer candidate using four fixed seeds.

3. `metrics`  
   Computes Harmora and the eight baseline metrics on the cached representations.

4. `aggregate`  
   Aggregates downstream utilities across the four seeds.

5. `correlate`  
   Computes within-model and cross-model evaluation results.

The outputs are written to:

```text
outputs/full_experiment/
```

### Resume from a specific stage

To continue from the metrics stage:

```bash
python scripts/run_full_pipeline.py --from-stage metrics
```

### Run a selected range of stages

For example, to run from downstream evaluation to aggregation:

```bash
python scripts/run_full_pipeline.py \
  --from-stage downstream \
  --to-stage aggregate
```

### Run a small experiment

A smaller run can be used for testing or debugging:

```bash
python scripts/run_full_pipeline.py \
  --models minilm_l6 \
  --tasks Banking77Classification.v2 \
  --seeds 11 \
  --to-stage correlate
```

Windows PowerShell:

```powershell
python scripts/run_full_pipeline.py `
  --models minilm_l6 `
  --tasks Banking77Classification.v2 `
  --seeds 11 `
  --to-stage correlate
```

Use `--overwrite` only when the cached samples, embeddings, and completed results must be rebuilt.

## Paper analyses

After the full experiment finishes, generate the paper analyses with:

```bash
python scripts/run_paper_analyses.py
```

This command generates:

- within-model ranking figures and tables;
- cross-model selection figures and tables;
- bandwidth analyses;
- spectral-depth analyses;
- adaptive-precision experiments;
- runtime analyses;
- eigensolver analyses.

The generated files are written to:

```text
artifacts/generated/
```

### Run selected analyses

To run only the within-model, cross-model, and spectral analyses:

```bash
python scripts/run_paper_analyses.py --only within cross spectral
```

To run the adaptive-precision analysis on a CUDA device:

```bash
python scripts/run_paper_analyses.py --only precision --device cuda
```

To run the runtime analysis without the real-runtime experiment:

```bash
python scripts/run_paper_analyses.py --only runtime --skip-real-runtime
```

The reference vector and raster figures are stored in:

```text
reference/figures/
```

Small visual differences may appear when figures are generated on different operating systems. These differences can be caused by fonts or Matplotlib backends.

## Experiment configuration

The complete experiment configuration is available in:

[`configs/paper.yaml`](configs/paper.yaml)

### Models

The experiment uses seven language embedding models:

- MiniLM-L6
- MPNet-base
- E5-base-v2
- E5-large-v2
- BGE-base-en-v1.5
- BGE-large-en-v1.5
- Snowflake Arctic Embed M

### Tasks

The experiment uses eleven tasks.

#### Classification

- Banking77
- Emotion
- HUMEEmotion

#### Clustering

- ArXiv-P2P
- ArXiv-S2S
- Biorxiv-P2P

#### Pair classification

- LegalBenchPC

#### Semantic textual similarity

- STS15
- STS16
- STSBenchmark
- SICK-R

### Fixed settings

| Setting | Value |
|---|---|
| Downstream seeds | `11, 22, 33, 44` |
| Shared sample seed | `2025` |
| Maximum task sample size | `512` |
| Harmora bandwidth | `K = 10` |
| Harmora variance | `sigma² = 1` |
| Candidate pool | `106` model-layer candidates per task |
| Shortlist size | `5` |

When the following option is enabled, the configuration loader does not allow silent changes to the seven models or eleven tasks:

```yaml
lock_exact_experiment_set: true
```

## Reproducibility options

There are two main ways to use this repository.

### Reference result reproduction

Use this option to generate the reference figures, tables, and result files without downloading models or datasets:

```bash
python scripts/reproduce_paper.py
python scripts/validate_reproduction.py
```

### Independent computational rerun

Use this option to independently recompute the embeddings, downstream utilities, label-free scores, and analyses:

```bash
python scripts/run_full_pipeline.py
python scripts/run_paper_analyses.py
```

## Data and cache policy

Downloaded datasets, pretrained model weights, and large embedding caches are not stored in the repository.

The following directories are used only for local files:

```text
cache/models/
cache/data/
outputs/full_experiment/
artifacts/generated/
```

These paths are excluded through `.gitignore`.

The reference result files required for validation are stored in:

```text
reference/data/
```

## Tests

### Linux and macOS

Run the test suite with:

```bash
PYTHONPATH=src pytest -q tests
```

### Windows PowerShell

Run:

```powershell
$env:PYTHONPATH = "src"
pytest -q tests
```

## Citation

Citation metadata is provided in:

[`CITATION.cff`](CITATION.cff)

Please use this file when citing the repository or the paper.

## License

The code is released under the MIT License.

The datasets and pretrained models are covered by the licenses of their original providers.
