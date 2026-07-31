# Harmora

Official implementation and experiment package for:

> **Harmora: Label-Free Selection of Representations in Language Embedding Models via Laplacian Harmonics**

This repository contains the code, experiment configuration, analysis scripts, tests, and reference files used in the paper.

## Main features

The repository includes:

- a fixed **7-model × 11-task × 4-seed** experiment configuration;
- the Harmora metric and eight label-free baseline metrics;
- layer-wise downstream evaluation;
- within-model and cross-model representation selection;
- bandwidth and spectral-depth analyses;
- adaptive-precision experiments;
- runtime and eigensolver analyses;
- tests for the main components.

## Repository structure

```text
.
├── configs/
│   └── paper.yaml                 # Main experiment configuration
├── src/
│   └── harmora_downstream/        # Data, encoding, evaluation, and analysis pipeline
├── harmora_metrics/
│   └── metrics/                   # Harmora and baseline metric implementations
├── scripts/
│   ├── check_environment.py       # Check the local environment
│   ├── run_full_pipeline.py       # Run the full experiment pipeline
│   └── run_paper_analyses.py      # Generate the paper analyses
├── experiments/
│   ├── make_within_model_results.py
│   ├── make_cross_model_results.py
│   ├── make_spectral_results.py
│   ├── run_adaptive_precision_ablation.py
│   └── run_runtime_analysis.py
├── reference/                     # Reference data, figures, and LaTeX tables
├── paper/                         # Manuscript and supplementary files
├── tests/                         # Unit tests
├── docs/                          # Additional documentation
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── CITATION.cff                   # Citation information
└── LICENSE                        # License
```

Generated files are written to local directories under:

```text
cache/
outputs/
artifacts/
```

## Requirements

Python 3.10 or newer is required. Python 3.11 is recommended.

Run all commands from the root directory of the repository.

## Installation

### Linux and macOS

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the package:

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

Upgrade `pip` and install the package:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

### Development installation

To install the testing dependencies, run:

```bash
pip install -e ".[dev]"
```

### Conda installation

A Conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate harmora
```

## Environment check

Before running the experiment, check the local environment:

```bash
python scripts/check_environment.py
```

This command reports the Python version, operating system, selected device, configured models and tasks, output path, required libraries, and the bundled metric package.

## Full experiment

The full experiment downloads the configured Hugging Face models and MTEB datasets.

Run the complete pipeline with:

```bash
python scripts/run_full_pipeline.py
```

The pipeline uses cached task samples, embeddings, splits, and completed results when they are available.

### Pipeline stages

The pipeline contains five stages:

1. `samples` — creates one fixed sample for each selected task;
2. `downstream` — evaluates each model-layer candidate using the selected seeds;
3. `metrics` — computes Harmora and the eight baseline metrics;
4. `aggregate` — aggregates downstream results across seeds;
5. `correlate` — computes within-model and cross-model evaluation results.

The main outputs are written to:

```text
outputs/full_experiment/
```

### Resume from a stage

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

### Run a smaller experiment

A smaller run can be used for testing:

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

Use `--overwrite` only when existing samples, embeddings, splits, metrics, and completed results must be rebuilt.

## Paper analyses

After the full experiment finishes, generate the analyses used in the paper:

```bash
python scripts/run_paper_analyses.py
```

This command runs five analysis groups:

- within-model analysis;
- cross-model analysis;
- bandwidth and spectral-depth analysis;
- adaptive-precision analysis;
- runtime and eigensolver analysis.

The generated files are written to:

```text
artifacts/generated/
```

### Run selected analyses

Run the within-model, cross-model, and spectral analyses:

```bash
python scripts/run_paper_analyses.py --only within cross spectral
```

Run the adaptive-precision analysis with automatic device selection:

```bash
python scripts/run_paper_analyses.py --only precision --device auto
```

Request CUDA for the adaptive-precision analysis:

```bash
python scripts/run_paper_analyses.py --only precision --device cuda
```

Run the runtime analysis without the real-cache benchmark:

```bash
python scripts/run_paper_analyses.py --only runtime --skip-real-runtime
```

Small visual differences may appear when figures are generated on different systems because fonts and Matplotlib backends can differ.

## Experiment configuration

The main experiment configuration is stored in:

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

### Main settings

| Setting | Value |
|---|---|
| Downstream seeds | `11, 22, 33, 44` |
| Shared sampling seed | `2025` |
| Maximum task sample size | `512` |
| Harmora bandwidth | `K = 10` |
| Harmora variance parameter | `sigma² = 1` |
| Candidate pool | `106` model-layer representations per task |
| Shortlist size | `5` |

The configuration contains:

```yaml
lock_exact_experiment_set: true
```

When this option is enabled, the configuration loader rejects changes to the fixed seven-model and eleven-task experiment set.

## Data and generated files

Downloaded datasets, pretrained model weights, caches, and generated outputs are not tracked by Git.

The main local paths are:

```text
cache/
outputs/full_experiment/
artifacts/generated/
```

Reference data and manuscript assets are stored under:

```text
reference/
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

Citation information is provided in:

[`CITATION.cff`](CITATION.cff)

Please cite the Harmora paper when using this code or its artifacts.

## License

The code is released under the MIT License.

The datasets and pretrained models are covered by the licenses of their original providers.
