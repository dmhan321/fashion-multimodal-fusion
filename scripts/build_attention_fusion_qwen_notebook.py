#!/usr/bin/env python3
"""One-off generator for AttentionFusion_FinetunedCLIP_Qwen_Robustness.ipynb"""
import json
import re
from pathlib import Path
from textwrap import dedent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(s).strip().split("\n")}


def code(s):
    lines = dedent(s).strip().split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in lines]}


# Bump when FashionMultiModalDataset / image-path logic changes (must match runner gate).
ATTN_FUSION_DATASET_CELL_VERSION = 2


cells = []

cells.append(
    md(
        """
# Attention fusion — fine-tuned CLIP + Qwen captions (matched CSV splits)

This notebook trains **attention-based multimodal fusion** (CLIP image encoder + BERT text encoder + self-attention fusion) using:

- **Per-seed CSV splits** under `data/splits/seed_<N>/` (same protocol as partial CLIP fine-tuning).
- **Per-seed fine-tuned CLIP ViT-B/32** checkpoints (`seed_<N>_best_model.pth`) from `experiments/imageonly_clip_finetuned_robustness/models/` or `FASHION_CLIP_FINETUNED_MODELS_DIR`.
- **Qwen** captions: `data/captions/qwen25vl_caption_full.csv` (LLaVA-style columns; `status == success` for rows with captions).

**After git pull or regenerating this file:** use **Kernel → Restart**, then **Run All** from the top. Otherwise an old `FashionMultiModalDataset` can stay in memory while the setup cell changes, which yields an empty training set and a `num_samples=0` / `DataLoader` error.

**Prerequisites:** `clip`, `torch`, `transformers`, split CSVs under `data/splits/`, Qwen caption CSV, per-seed CLIP checkpoints, and image files. Split rows use `dataset/...` paths; if there is no top-level `dataset/` folder, images are expected under `data/raw dataset/...` (same as `ImageOnly_CLIP_Finetune.ipynb`). The first code cell runs a **path audit** and fails fast if required paths are missing.

**Outputs:** `experiments/attention_fusion_finetuned_clip_qwen_v2/` with `metrics/` and `artifacts/` (phase3-style layout). Trained **fusion + classifier** are saved as small **`seed_<N>_best_fusion_head.pth`** files (not full CLIP/BERT weights).

**Implementation notes:** accuracies on **0–1** scale; optimizer trains **fusion + classifier** only; CLIP/BERT frozen + **eval** via `train()` override; class weights from **filtered** train rows; **deterministic** train `DataLoader` seed; best val macro-F1 starts at **-1** so epoch 1 can save; if no file was written, a **fallback** fusion-head save runs before test load.
"""
    )
)

cells.append(
    code(
        r"""
import os
import re
import json
import shutil
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# --- project root ---
_walk = os.path.abspath(os.getcwd())
for _ in range(10):
    if os.path.isdir(os.path.join(_walk, "experiments")) and os.path.isdir(os.path.join(_walk, "data")):
        PROJECT_ROOT = _walk
        break
    _walk = os.path.dirname(_walk)
else:
    PROJECT_ROOT = os.path.abspath(os.getcwd())
os.chdir(PROJECT_ROOT)
print("PROJECT_ROOT:", PROJECT_ROOT)

# --- toggles ---
RUN_ALL_SEEDS = False   # set True for full 30-seed loop
SMOKE_SEED = 13         # used when RUN_ALL_SEEDS is False

# --- training hyperparameters ---
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
MODEL_INIT_SEED = 42
NUM_WORKERS = 0


def pick_training_device():
    # Prefer NVIDIA CUDA, then Apple MPS, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_torch_compute_diagnostics():
    print("torch:", torch.__version__)
    vcuda = getattr(torch.version, "cuda", None)
    print("torch.version.cuda (wheel build):", vcuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device 0:", torch.cuda.get_device_name(0))
        return
    if vcuda:
        print(
            "This PyTorch build includes CUDA, but the GPU is not usable from this process. "
            "Typical causes: NVIDIA driver/kernel vs user-space library mismatch (often fixed by a reboot), "
            "no discrete GPU, VM/WSL without GPU passthrough, or a driver/GPU pairing that rejects this CUDA build."
        )
    nv = shutil.which("nvidia-smi")
    if nv:
        try:
            r = subprocess.run([nv, "-L"], capture_output=True, text=True, timeout=10)
            msg = ((r.stdout or "") + (r.stderr or "")).strip()
            if msg:
                print("nvidia-smi -L:\n", msg)
            low = msg.lower()
            if "version mismatch" in low or "failed to initialize nvml" in low:
                print(
                    "\n>>> Driver fix: reboot so the loaded NVIDIA kernel module matches libnvidia-ml.so, "
                    "or reinstall/repair the NVIDIA driver until `nvidia-smi` works outside Jupyter."
                )
        except Exception as ex:
            print("nvidia-smi check failed:", ex)
    else:
        print("nvidia-smi not on PATH (install NVIDIA driver + tools, or use a CUDA-capable machine).")


device = pick_training_device()
print_torch_compute_diagnostics()
print("Selected device:", device)

CAPTION_CSV = os.path.join(PROJECT_ROOT, "data", "captions", "qwen25vl_caption_full.csv")
SPLITS_ROOT = os.path.join(PROJECT_ROOT, "data", "splits")
CLIP_FINETUNED_ROOT = os.environ.get(
    "FASHION_CLIP_FINETUNED_MODELS_DIR",
    os.path.join(PROJECT_ROOT, "experiments", "imageonly_clip_finetuned_robustness", "models"),
)
EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "experiments", "attention_fusion_finetuned_clip_qwen_v2")
METRICS_DIR = os.path.join(EXPERIMENT_ROOT, "metrics")
ARTIFACTS_DIR = os.path.join(EXPERIMENT_ROOT, "artifacts")
for d in [
    METRICS_DIR,
    os.path.join(METRICS_DIR, "experiments"),
    os.path.join(ARTIFACTS_DIR, "models"),
    os.path.join(ARTIFACTS_DIR, "learning_curves"),
    os.path.join(ARTIFACTS_DIR, "comparison_plots"),
]:
    os.makedirs(d, exist_ok=True)

print("CAPTION_CSV:", CAPTION_CSV)
print("CLIP_FINETUNED_ROOT:", CLIP_FINETUNED_ROOT)
print("EXPERIMENT_ROOT:", EXPERIMENT_ROOT)


def discover_split_seeds(splits_root: str):
    seeds = []
    for name in os.listdir(splits_root):
        m = re.match(r"seed_(\d+)$", name)
        if m and os.path.isdir(os.path.join(splits_root, name)):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


SEEDS = discover_split_seeds(SPLITS_ROOT)
print("Discovered split seeds:", SEEDS, "count:", len(SEEDS))

with open(os.path.join(METRICS_DIR, "seeds_list.txt"), "w") as f:
    f.write("Seeds from data/splits/seed_*\n")
    for s in SEEDS:
        f.write(f"{s}\n")


def resolve_split_image_path(row_image_path, base_dir):
    # Map split CSV image_path to on-disk file (same rules as ImageOnly_CLIP_Finetune).
    base = base_dir or "."
    p = str(row_image_path)
    if not os.path.isabs(p):
        rel = p.replace(chr(92), "/")
        dataset_top = os.path.join(base, "dataset")
        if rel.startswith("dataset/") and not os.path.isdir(dataset_top):
            p = os.path.join(base, "data", "raw dataset", rel[len("dataset/") :])
        else:
            p = os.path.join(base, p)
    if "%" in p:
        parts = p.replace(chr(92), "/").split("/")
        p = "/".join(unquote(part) if "%" in part else part for part in parts)
    return os.path.normpath(p)


def load_qwen_captions(csv_path: str, base_dir: str):
    # Map image_path -> Qwen caption (success rows only). Register raw, normpath, naive join, and resolve_split_image_path.
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()
    if "caption" not in df.columns:
        raise ValueError("Expected column 'caption' in caption CSV")

    def register(keys_dict, raw_path, caption_text):
        cap = caption_text.strip()
        p = str(raw_path)
        keys_dict[p] = cap
        keys_dict[os.path.normpath(p)] = cap
        if base_dir and not os.path.isabs(p):
            abs_p = os.path.normpath(os.path.join(base_dir, p))
            keys_dict[abs_p] = cap
        res = resolve_split_image_path(p, base_dir or ".")
        keys_dict[res] = cap

    d = {}
    for _, row in df.iterrows():
        c = row["caption"]
        if not isinstance(c, str) or not c.strip():
            continue
        register(d, row["image_path"], c)
    return d


BASE_DIR = PROJECT_ROOT
captions_dict = load_qwen_captions(CAPTION_CSV, BASE_DIR)
print("Qwen caption dict size (keys may include path variants):", len(captions_dict))

_dfq = pd.read_csv(CAPTION_CSV)
if "style" in _dfq.columns:
    all_styles = sorted(_dfq["style"].dropna().astype(str).unique())
else:
    raise ValueError("Qwen CSV must have 'style' for num_classes")
style_to_idx = {s: i for i, s in enumerate(all_styles)}
num_classes = len(all_styles)
print("num_classes:", num_classes)


def print_fusion_path_audit(smoke_seed: int):
    # Fail fast with readable checks (caption CSV, splits, images, CLIP ckpt).
    print("=== Path audit (required files / dirs) ===")
    items = [
        ("CAPTION_CSV", os.path.isfile(CAPTION_CSV)),
        ("SPLITS_ROOT", os.path.isdir(SPLITS_ROOT)),
        ("CLIP_FINETUNED_ROOT", os.path.isdir(CLIP_FINETUNED_ROOT)),
        (
            "repo dataset/ OR data/raw dataset/",
            os.path.isdir(os.path.join(PROJECT_ROOT, "dataset"))
            or os.path.isdir(os.path.join(PROJECT_ROOT, "data", "raw dataset")),
        ),
    ]
    for label, ok in items:
        print(f"  [{'x' if ok else ' '}] {label}")
    if not all(ok for _, ok in items):
        raise RuntimeError(
            "Path audit failed. Open repo root as cwd or fix paths (need data/, experiments/, and images)."
        )

    train_csv = os.path.join(SPLITS_ROOT, f"seed_{smoke_seed}", "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Missing split train.csv: {train_csv}")
    tr = pd.read_csv(train_csv, nrows=5)
    ok_files = 0
    for i, row in tr.iterrows():
        raw = str(row["image_path"])
        res = resolve_split_image_path(raw, BASE_DIR)
        ex = os.path.isfile(res)
        ok_files += int(ex)
        print(f"  [{'x' if ex else ' '}] smoke row {i} file exists")
        print(f"      raw={raw}")
        print(f"      resolved={res}")
    if ok_files == 0:
        raise RuntimeError(
            "Path audit: smoke-split rows resolve to no image files. "
            "If images live elsewhere, set BASE_DIR / symlink dataset/ or data/raw dataset/ under PROJECT_ROOT."
        )

    ck = os.path.join(CLIP_FINETUNED_ROOT, f"seed_{smoke_seed}_best_model.pth")
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"Missing CLIP checkpoint for smoke seed: {ck}")
    print(f"  [x] smoke CLIP ckpt seed_{smoke_seed}_best_model.pth")
    print("=== Path audit OK ===")


print_fusion_path_audit(SMOKE_SEED)
"""
    )
)

cells.append(md("## Load CLIP + BERT"))

cells.append(
    code(
        r"""
import clip
from transformers import AutoTokenizer, AutoModel

print("Loading CLIP ViT-B/32 …")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip_model = clip_model.float()
clip_model.eval()

print("Loading BERT …")
fashionbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fashionbert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
fashionbert_model.eval()


def load_finetuned_clip_weights(clip_model, ckpt_path, map_location):
    # Load partial fine-tuned CLIP from ImageOnlyFashionClassifier checkpoint.
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=map_location)
    clip_sd = {k[len("clip_model.") :]: v for k, v in sd.items() if k.startswith("clip_model.")}
    if len(clip_sd) == 0:
        raise ValueError(f"No clip_model.* keys in {ckpt_path} (wrong checkpoint format?)")
    clip_model.load_state_dict(clip_sd, strict=True)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("  Loaded fine-tuned CLIP:", ckpt_path.name, "|", len(clip_sd), "tensor keys")


print("✅ Encoders ready (reload CLIP weights per seed before training).")
"""
    )
)

cells.append(md("## Dataset + attention fusion model"))

cells.append(
    code(
        r"""
class FashionMultiModalDataset(Dataset):
    # Split CSV rows + Qwen captions + CLIP preprocess. Excludes rows without file or caption.

    def _resolve_path(self, row_image_path):
        return resolve_split_image_path(str(row_image_path), self.base_dir or ".")

    def _caption_lookup(self, resolved, raw):
        for k in (resolved, raw, os.path.normpath(str(raw))):
            if k in self.captions_dict:
                return self.captions_dict[k]
        return None

    def __init__(self, df, captions_dict, style_to_idx, clip_preprocess, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.captions_dict = captions_dict
        self.style_to_idx = style_to_idx
        self.clip_preprocess = clip_preprocess
        self.base_dir = base_dir

        self.valid_indices = []
        missing_file = 0
        missing_cap = 0
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            raw = str(row["image_path"])
            resolved = self._resolve_path(raw)
            cap = self._caption_lookup(resolved, raw)
            if cap is None:
                missing_cap += 1
                continue
            if not os.path.isfile(resolved):
                missing_file += 1
                continue
            self.valid_indices.append(idx)

        print(
            f"  Dataset: {len(self.valid_indices)} / {len(self.df)} usable | missing caption: {missing_cap} | missing file: {missing_file}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        raw_key = str(row["image_path"])
        image_path = self._resolve_path(raw_key)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Missing image after filter: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self.clip_preprocess(image)
        caption = self._caption_lookup(image_path, raw_key)
        if caption is None:
            raise RuntimeError(f"Missing caption for {image_path}")
        style = row["style"]
        label = self.style_to_idx[str(style)]
        return {
            "image": image,
            "caption": caption,
            "label": label,
            "style": style,
            "image_path": image_path,
        }


class AttentionFusion(nn.Module):
    def __init__(self, visual_dim, textual_dim, hidden_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.textual_proj = nn.Linear(textual_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, visual_features, textual_features):
        visual_proj = self.visual_proj(visual_features)
        textual_proj = self.textual_proj(textual_features)
        combined = torch.stack([visual_proj, textual_proj], dim=1)
        attended, _ = self.attention(combined, combined, combined)
        attended = self.layer_norm(attended)
        fused = torch.mean(attended, dim=1)
        return self.final_proj(fused)


class MultiModalFashionClassifier(nn.Module):
    def __init__(self, clip_model, fashionbert_model, num_classes, dropout=0.5, visual_dim=512, textual_dim=768):
        super().__init__()
        self.clip_model = clip_model
        self.fashionbert_model = fashionbert_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for p in self.fashionbert_model.parameters():
            p.requires_grad = False
        self.fusion = AttentionFusion(visual_dim, textual_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.fashionbert_model.eval()
        return self

    def forward(self, images, captions):
        with torch.no_grad():
            visual_features = self.clip_model.encode_image(images).float()
            inputs = fashionbert_tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(images.device)
            outputs = self.fashionbert_model(**inputs)
            textual_features = outputs.last_hidden_state[:, 0, :]
        fused = self.fusion(visual_features, textual_features)
        return self.classifier(fused)


def train_epoch(model, train_loader, criterion, optimizer, dev):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc="train", leave=False):
        images = batch["image"].to(dev)
        captions = batch["caption"]
        labels = batch["label"].to(dev)
        optimizer.zero_grad()
        logits = model(images, captions)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    acc = correct / max(total, 1)
    return total_loss / max(len(train_loader), 1), acc


def validate_epoch(model, val_loader, criterion, dev, collect_meta=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions, all_labels = [], []
    all_paths, all_styles = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", leave=False):
            images = batch["image"].to(dev)
            captions = batch["caption"]
            labels = batch["label"].to(dev)
            logits = model(images, captions)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_predictions.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            if collect_meta:
                all_paths.extend(list(batch["image_path"]))
                all_styles.extend(list(batch["style"]))
    macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0) if all_predictions else 0.0
    acc = correct / max(total, 1)
    if collect_meta:
        return total_loss / max(len(val_loader), 1), acc, all_predictions, all_labels, macro, all_paths, all_styles
    return total_loss / max(len(val_loader), 1), acc, all_predictions, all_labels, macro


print("✅ Dataset + model classes defined")
ATTN_FUSION_DATASET_CELL_VERSION = ___DSCELLVER___
""".replace("___DSCELLVER___", str(ATTN_FUSION_DATASET_CELL_VERSION)),
    )
)

cells.append(md("## Runner: load splits, reload CLIP per seed, train"))

cells.append(
    code(
        r"""
def load_split_csvs(seed: int):
    split_dir = os.path.join(SPLITS_ROOT, f"seed_{seed}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(split_dir)
    assert f"seed_{seed}" in split_dir.replace("\\", "/")
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    return split_dir, train_df, val_df, test_df


def make_seed_worker(base_seed):
    def _fn(worker_id):
        w = int(base_seed) + int(worker_id)
        np.random.seed(w)
        torch.manual_seed(w)

    return _fn


def run_robustness_experiment(seed_value, seed_idx):
    _exp_ds_ver = ___DSCELLVER___
    if globals().get("ATTN_FUSION_DATASET_CELL_VERSION") != _exp_ds_ver:
        raise RuntimeError(
            "Dataset cell is missing or out of date (FashionMultiModalDataset / path remap). "
            "Use **Kernel -> Restart**, then **Run All** from the first cell.\n"
            f"Expected ATTN_FUSION_DATASET_CELL_VERSION == {_exp_ds_ver!r}, "
            f"got {globals().get('ATTN_FUSION_DATASET_CELL_VERSION')!r}."
        )
    print(f"\n{'='*70}\nExperiment {seed_idx}/{len(SEEDS)}: seed {seed_value}\n{'='*70}")

    ckpt_path = Path(CLIP_FINETUNED_ROOT) / f"seed_{seed_value}_best_model.pth"
    assert str(seed_value) in ckpt_path.name

    result_file = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
    if os.path.exists(result_file):
        print("  ⏭️  Result exists, skipping")
        with open(result_file) as f:
            return json.load(f)

    split_dir, train_df, val_df, test_df = load_split_csvs(seed_value)
    print("  Split sizes:", len(train_df), len(val_df), len(test_df))
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        ds_tmp = FashionMultiModalDataset(part, captions_dict, style_to_idx, clip_preprocess, base_dir=BASE_DIR)
        print(f"    {name}: with captions {len(ds_tmp)} / {len(part)}")

    load_finetuned_clip_weights(clip_model, ckpt_path, map_location=device)

    torch.manual_seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)

    train_ds = FashionMultiModalDataset(train_df, captions_dict, style_to_idx, clip_preprocess, base_dir=BASE_DIR)
    val_ds = FashionMultiModalDataset(val_df, captions_dict, style_to_idx, clip_preprocess, base_dir=BASE_DIR)
    test_ds = FashionMultiModalDataset(test_df, captions_dict, style_to_idx, clip_preprocess, base_dir=BASE_DIR)

    if len(train_ds) == 0:
        r0 = str(train_df.iloc[0]["image_path"]) if len(train_df) else "<empty train_df>"
        res0 = resolve_split_image_path(r0, BASE_DIR) if len(train_df) else ""
        cap0 = (
            any(k in captions_dict for k in (res0, r0, os.path.normpath(str(r0))))
            if len(train_df)
            else False
        )
        file0 = os.path.isfile(res0) if len(train_df) else False
        if len(train_df) and file0 and cap0:
            raise RuntimeError(
                "Training dataset is empty even though the first train row resolves to an existing file "
                "and has a Qwen caption. The FashionMultiModalDataset class in memory is almost certainly "
                "out of date (old image-path logic).\n\n"
                "Fix: Jupyter menu **Kernel -> Restart**, then **Run All** from the first cell so the "
                "Dataset class and resolve_split_image_path stay in sync.\n"
                f"(Debug: resolved first row -> {res0!r})"
            )
        raise RuntimeError(
            "Training dataset is empty (0 rows with Qwen caption + existing image). Common causes: "
            "stale kernel after notebook edits, wrong PROJECT_ROOT, or caption/path mismatch.\n"
            f"  PROJECT_ROOT={PROJECT_ROOT}\n"
            f"  BASE_DIR={BASE_DIR}\n"
            f"  first train image_path={r0!r}\n"
            f"  resolved={res0!r} exists={os.path.isfile(res0)}\n"
            f"  dataset/ at repo root: {os.path.isdir(os.path.join(BASE_DIR, 'dataset'))}\n"
            f"  data/raw dataset/: {os.path.isdir(os.path.join(BASE_DIR, 'data', 'raw dataset'))}"
        )

    loader_seed = MODEL_INIT_SEED + int(seed_value)
    g = torch.Generator()
    g.manual_seed(loader_seed)
    try:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=make_seed_worker(loader_seed) if NUM_WORKERS > 0 else None,
            generator=g,
        )
    except ValueError as e:
        if "num_samples" in str(e):
            raise RuntimeError(
                "Training DataLoader failed because the training set has length 0 (RandomSampler). "
                "If you updated this notebook, use **Kernel -> Restart** then **Run All** from the top "
                "so the FashionMultiModalDataset definition matches the setup cell (resolve_split_image_path / "
                "data/raw dataset remap).\n"
                f"len(train_ds)={len(train_ds)!r}\n"
                f"Original error: {e!r}"
            ) from e
        raise
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_valid_df = train_ds.df.iloc[train_ds.valid_indices]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(num_classes),
        y=train_valid_df["style"].map(style_to_idx).values,
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = MultiModalFashionClassifier(clip_model, fashionbert_model, num_classes=num_classes, dropout=DROPOUT).to(device)
    n_trainable = sum(p.numel() for p in model.fusion.parameters()) + sum(p.numel() for p in model.classifier.parameters())
    print("  Trainable parameters (fusion + classifier only):", n_trainable)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        list(model.fusion.parameters()) + list(model.classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("  Trainable parameter names (first 25):", trainable_names[:25])
    print("  Total trainable tensors:", len(trainable_names))
    assert all(
        n.startswith("fusion.") or n.startswith("classifier.") for n in trainable_names
    ), f"Unexpected trainable params: {trainable_names}"

    fusion_head_path = os.path.join(ARTIFACTS_DIR, "models", f"seed_{seed_value}_best_fusion_head.pth")

    train_losses, val_losses, train_accs, val_accs, val_macro_f1s, learning_rates = [], [], [], [], [], []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_val_loss = float("inf")
    early_stopped = False
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _, va_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        val_macro_f1s.append(va_f1)

        if va_f1 > best_val_macro_f1:
            best_val_macro_f1 = va_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "fusion": model.fusion.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "seed": int(seed_value),
                    "clip_checkpoint": str(ckpt_path),
                    "caption_csv": CAPTION_CSV,
                    "best_epoch": int(best_epoch),
                    "best_val_macro_f1": float(best_val_macro_f1),
                },
                fusion_head_path,
            )
        else:
            patience_counter += 1
        if va_loss < best_val_loss:
            best_val_loss = va_loss
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(f"  Early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_macro_f1={va_f1:.4f}")

    total_time = time.time() - start_time
    if not os.path.exists(fusion_head_path):
        torch.save(
            {
                "fusion": model.fusion.state_dict(),
                "classifier": model.classifier.state_dict(),
                "seed": int(seed_value),
                "clip_checkpoint": str(ckpt_path),
                "caption_csv": CAPTION_CSV,
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_val_macro_f1),
                "fallback_save": True,
            },
            fusion_head_path,
        )
        print("  Warning: no fusion-head checkpoint during training; saved fallback from last weights.")

    ck = torch.load(fusion_head_path, map_location=device)
    model.fusion.load_state_dict(ck["fusion"])
    model.classifier.load_state_dict(ck["classifier"])
    model.eval()
    te_loss, te_acc, te_pred, te_lab, te_f1, te_paths, te_styles = validate_epoch(
        model, test_loader, criterion, device, collect_meta=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(train_losses, label="train")
    axes[0, 0].plot(val_losses, label="val")
    axes[0, 0].legend()
    axes[0, 1].plot(train_accs, label="train acc")
    axes[0, 1].plot(val_accs, label="val acc")
    axes[0, 1].set_ylabel("accuracy (0–1)")
    axes[0, 1].legend()
    axes[1, 0].plot(val_macro_f1s, label="val macro F1")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.5,
        f"seed={seed_value}\nbest_val_macro_f1={best_val_macro_f1:.4f}\ntest_macro_f1={te_f1:.4f}\ntest_acc={te_acc:.4f}",
        fontsize=11,
        family="monospace",
    )
    plt.suptitle(f"Attention fusion + finetuned CLIP + Qwen — seed {seed_value}")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "learning_curves", f"seed_{seed_value}_learning_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    results = {
        "experiment_id": f"seed_{seed_value}",
        "seed_value": seed_value,
        "seed_index": seed_idx,
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "splits": "data/splits/seed_<N>/*.csv",
            "clip_checkpoint": str(ckpt_path),
            "caption_csv": CAPTION_CSV,
            "fusion": "attention",
            "fusion_head_checkpoint": fusion_head_path,
        },
        "configuration": {
            "learning_rate": float(LEARNING_RATE),
            "batch_size": BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "dropout": DROPOUT,
            "weight_decay": float(WEIGHT_DECAY),
            "max_epochs": MAX_EPOCHS,
            "model_init_seed": MODEL_INIT_SEED,
        },
        "validation_metrics": {
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_epoch": int(best_epoch),
            "best_val_accuracy": float(val_accs[best_epoch - 1]) if best_epoch > 0 else 0.0,
        },
        "test_metrics": {
            "test_macro_f1": float(te_f1),
            "test_accuracy": float(te_acc),
            "test_loss": float(te_loss),
            "test_predictions": [int(x) for x in te_pred],
            "test_labels": [int(x) for x in te_lab],
            "test_image_paths": [str(x) for x in te_paths],
            "test_styles": [str(x) for x in te_styles],
        },
        "caption_coverage": {
            "train_rows_csv": int(len(train_df)),
            "val_rows_csv": int(len(val_df)),
            "test_rows_csv": int(len(test_df)),
            "train_with_caption_and_file": int(len(train_ds)),
            "val_with_caption_and_file": int(len(val_ds)),
            "test_with_caption_and_file": int(len(test_ds)),
        },
        "training_info": {
            "total_epochs": len(train_losses),
            "early_stopped": early_stopped,
            "total_time_minutes": float(total_time / 60.0),
        },
        "data_split_info": {
            "split_dir": split_dir,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        },
    }
    out_json = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Done seed {seed_value} | best val macro F1={best_val_macro_f1:.4f} | test macro F1={te_f1:.4f}")
    return results


if RUN_ALL_SEEDS:
    all_results = []
    failed = []
    for i, sv in enumerate(SEEDS, 1):
        try:
            all_results.append(run_robustness_experiment(sv, i))
        except Exception as e:
            print(f"FAIL seed {sv}: {e}")
            failed.append((sv, str(e)))
    print("Completed:", len(all_results), "Failed:", len(failed))
else:
    run_robustness_experiment(SMOKE_SEED, 1)
""".replace("___DSCELLVER___", str(ATTN_FUSION_DATASET_CELL_VERSION)),
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "robustness" / "AttentionFusion_FinetunedCLIP_Qwen_Robustness.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote", out)
