# Data layout

Everything under `data/` is **local-only** (see root `.gitignore`): large images, tables, and checkpoints are not committed.

## `raw dataset/`

Original **FashionStyle14-style images**, organized as **one subfolder per style** (e.g. `natural/`, `street/`, …). Paths in caption CSVs should stay consistent with how you reference these files from the repo root (or set `PROJECT_ROOT` / working directory accordingly).

Note: the folder name includes a **space** (`raw dataset`), so use quotes in shell paths.

## `processed/`

Current **tabular / config** artifacts used by training and evaluation notebooks:

- `caption_dataset_final_full.csv` — full caption table (main path used in code: `…/data/processed/caption_dataset_final_full.csv`)
- `caption_train.csv`, `caption_val.csv`, `caption_test.csv` — splits
- `caption_style_mapping.json`, `class_weights.json`
- `seeds_list.txt` — optional shared seeds list for robustness runs (also duplicated under `experiments/…/metrics/` where versioned)

## `processed/achived/`

Local **legacy / backup** material (older CSVs, notebooks, experiment exports, checkpoints). Safe to delete on your machine if you no longer need it; not part of the main pipeline.

## Removed layout

The older subdirectory **`processed/caption_pipeline/`** is no longer used; processed caption files live **directly** under `processed/`.
