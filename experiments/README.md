# Experiments

Each run family has the same shape:

- `metrics/` — small, versionable outputs: JSON summaries, CSV tables, per-seed result files under `metrics/experiments/` when applicable.
- `artifacts/` — local-only: model checkpoints, plots, learning curves, visualization folders (gitignored).

After pulling the repo, regenerate `artifacts/` by running the corresponding notebooks under `notebooks/`.
