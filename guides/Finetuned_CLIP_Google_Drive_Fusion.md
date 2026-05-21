# Fine-tuned CLIP weights (Google Drive) → fusion ablation

**Google Drive (model weights — start here):** _[[paste Shared drive link here](https://drive.google.com/drive/folders/19I0tzKzFbXCqNUvk7G3jJUvs1QTIt1Uq?usp=drive_link)]_

When you replace weights after a new training run, bump a **version suffix** in the Drive folder name and update the link above.

---

This project **does not** store large checkpoints in Git (see `.gitignore`: `*.pth`).  
The team maintains the **Google Drive** copy described above, with the same layout as:

`experiments/imageonly_clip_finetuned_robustness/models/`

Expected files (one per matched split seed), for example:

```text
seed_13_best_model.pth
seed_14_best_model.pth
...
```

(Exact seed list should match your `data/splits/seed_*` robustness seeds.)

### Planned workflow for fusion ablation

Teammates will **download the per-seed checkpoints** from Google Drive, **load the fine-tuned CLIP weights** for the matching split seed (see §4), and **run fusion experiments** (attention, concat, gated, etc.) using the existing pattern: **CLIP `encode_image` on preprocessed pixels** plus the text branch, then the fusion head—implemented in the fusion notebooks by wiring in the load step from §5.

---

## 1. Share approach (single source of truth)

| Layer | What lives there |
|--------|------------------|
| **Google Drive** | Entire `models/` folder from `imageonly_clip_finetuned_robustness` (~large total size). Use a **Shared drive** or team-owned folder so access does not depend on one person’s account. |
| **GitHub (this repo)** | Notebooks, configs, split CSVs policy, **this guide** (Drive link at the top). |

Teammates: **clone GitHub**, **sync or download Drive `models/` once**, then point fusion code at the **local** path.

---

## 2. What each file is

Each `seed_<N>_best_model.pth` is a **`torch.save`** of the **full** `ImageOnlyFashionClassifier` `state_dict` from `notebooks/robustness/ImageOnly_CLIP_Finetune.ipynb`:

- Keys prefixed with **`clip_model.`** — OpenAI **CLIP ViT-B/32** visual + text stack (fusion only needs the **image tower** loaded into a CLIP instance; easiest path: load all `clip_model.*` weights into `clip.load("ViT-B/32", …)`).
- Keys prefixed with **`classifier.`** — small MLP head for **image-only** training (ignored for standard fusion, which has its own head).

Backbone name in training: **`ViT-B/32`**, `jit=False`.

---

## 3. Local setup after download

### Option A — Mirror the repo path (recommended)

Place files so this path exists under your clone:

```text
<PROJECT_ROOT>/experiments/imageonly_clip_finetuned_robustness/models/seed_<N>_best_model.pth
```

Then fusion notebooks can use the **same relative paths** as the image-only robustness notebook (`ARTIFACTS_DIR / "models" / f"seed_{seed}_best_model.pth"`).

### Option B — Arbitrary directory + environment variable

1. Download Drive `models/` to e.g. `~/fashion-data/clip_finetuned_robustness_models/`.
2. Export a base path your notebooks read:

   ```bash
   export FASHION_CLIP_FINETUNED_MODELS_DIR="$HOME/fashion-data/clip_finetuned_robustness_models"
   ```

3. In fusion code, resolve:

   `Path(os.environ["FASHION_CLIP_FINETUNED_MODELS_DIR"]) / f"seed_{seed}_best_model.pth"`

Pick **one** convention for the whole team and document the Drive folder name/version (e.g. `clip_finetuned_robustness_models_v1`).

---

## 4. Matched-seed rule (avoid subtle bugs)

For **strict matched-seed** experiments (aligned with per-seed partial CLIP training):

- Split CSVs: `data/splits/seed_<N>/train.csv` (and val/test).
- Checkpoint: **`seed_<N>_best_model.pth`** trained on that seed’s **train** split only.

**Do not** use checkpoint from seed A with splits from seed B unless you explicitly label that as a **separate ablation**.

---

## 5. Loading fine-tuned CLIP inside fusion (minimal pattern)

After the same base load as frozen CLIP fusion:

```python
import os
from pathlib import Path
import torch
import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13  # must match the split seed you are training/evaluating

root = Path(os.environ.get(
    "FASHION_CLIP_FINETUNED_MODELS_DIR",
    str(Path.cwd() / "experiments" / "imageonly_clip_finetuned_robustness" / "models"),
))
ckpt_path = root / f"seed_{SEED}_best_model.pth"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model = clip_model.float()

sd = torch.load(ckpt_path, map_location=DEVICE)
clip_sd = {k[len("clip_model.") :]: v for k, v in sd.items() if k.startswith("clip_model.")}
clip_model.load_state_dict(clip_sd, strict=True)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
```

Use this **`clip_model`** and **`clip_preprocess`** in attention / concat / gated fusion in place of “pretrained only” CLIP.

If `load_state_dict` fails, you are on the wrong backbone, wrong file, or a corrupted download — re-sync from Drive.

---

## 6. Sanity checks (before long runs)

1. **Path:** `ckpt_path.exists()` is True.  
2. **Strict load:** no missing/unexpected keys after stripping `clip_model.`.  
3. **Smoke forward:** one batch through fusion with loaded weights; loss finite, no shape errors.

---

## 7. Checklist for teammates

- [ ] Drive `models/` downloaded or `rclone` sync’d to the agreed local path.  
- [ ] `SEED` in fusion matches `data/splits/seed_<SEED>/` and `seed_<SEED>_best_model.pth`.  
- [ ] Same Python env / `clip` package as training (ViT-B/32).  
- [ ] Fusion notebook updated to **load** checkpoint (snippet in §5) instead of only `clip.load` pretrained weights.
