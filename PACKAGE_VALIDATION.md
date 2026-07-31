# Package validation

Validation date: **2026-07-29**

## Completed checks

- Python compilation completed without syntax errors.
- Unit tests: **13 passed**.
- Exact frozen artifact reproduction completed successfully.
- Principal paper values, task/model/seed counts, adaptive-precision validation,
  and required figure/table inventory passed automated validation.
- The within-model and cross-model analysis generators completed successfully
  on the frozen seed-level reference inputs.
- Editable package import was verified with the local build environment.

## Validated principal values

- Harmora within-model mean Spearman: `0.4323868963`
- Strongest within-model baseline: `0.270963`
- Selected-candidate percentile: `0.8457792208`
- NDCG@1: `0.8432421363`
- NDCG@3: `0.7954741032`
- Top-5 overlap: `0.3727272727`
- Exhaustive post-embedding runtime: `762.2988681 s`
- Harmora-shortlist runtime: `166.9959977 s`
- Runtime speedup: `4.5647733x`
- Adaptive-precision NDCG@5 at 25%: `0.8300904248`
- Adaptive-precision utility Spearman at 25%: `0.4911958493`
- Official Harmora reconstruction: `1166` candidates, maximum absolute error
  below `1e-6`

## Scope

The full seven-model, eleven-task experiment was not rerun during package
assembly because it requires external Hugging Face/MTEB downloads and
substantial compute. The repository contains the complete locked configuration,
core pipeline, focused analysis scripts, exact submitted artifacts, and commands
needed for that independent rerun.
