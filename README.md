# Fashion multimodal fusion (DATA255)

## Layout

| Path | Purpose |
|------|---------|
| `notebooks/preprocessing/` | EDA and dataset understanding |
| `notebooks/captioning/` | Caption generation (e.g. LLaVA workflows) |
| `notebooks/training/` | Hyperparameter search, phase-2 checks, ResNet baseline |
| `notebooks/robustness/` | Multi-seed robustness sweeps |
| `notebooks/evaluation/` | Caption evaluation and per-class metric analysis |
| `scripts/` | Python helpers and one-off automation |
| `config/paths.py` | Optional shared path constants |
| `data/raw dataset/` | Original images by style folder (large; gitignored; name has a space) |
| `data/processed/` | Caption CSVs, splits, `class_weights.json`, `seeds_list.txt` (large; gitignored) |
| `data/processed/achived/` | Local legacy backups (gitignored) |
| `experiments/<name>/metrics` | Lightweight results tracked in git |
| `experiments/<name>/artifacts` | Checkpoints and plots (gitignored) |

## Running notebooks

Open Jupyter from the **repository root**, or run cells as-is: the robustness and training notebooks detect `PROJECT_ROOT` and `chdir` there automatically.

## LLaVA

The upstream `LLaVA/` tree is gitignored if present; install or clone separately.
