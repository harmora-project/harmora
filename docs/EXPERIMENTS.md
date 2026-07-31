# Locked paper experiment

The paper configuration is intentionally fixed to prevent silent drift.

## Models

1. `sentence-transformers/all-MiniLM-L6-v2`
2. `sentence-transformers/all-mpnet-base-v2`
3. `intfloat/e5-base-v2`
4. `intfloat/e5-large-v2`
5. `BAAI/bge-base-en-v1.5`
6. `BAAI/bge-large-en-v1.5`
7. `Snowflake/snowflake-arctic-embed-m`

## MTEB tasks

- Classification: Banking77, Emotion, HUMEEmotion
- Clustering: ArXiv-P2P, ArXiv-S2S, Biorxiv-P2P
- Pair classification: LegalBenchPC
- Semantic textual similarity: STS15, STS16, STSBenchmark, SICK-R

## Primary settings

- Downstream seeds: `11, 22, 33, 44`
- Shared sampling seed: `2025`
- Maximum task sample: `512`
- Harmora bandwidth: `K=10`
- Harmora variance parameter: `sigma^2=1`
- Candidate pool: `106` model-layer representations per task
- Shortlist size: `5`

The exact values are stored in `configs/paper.yaml` and checked by the config loader.
