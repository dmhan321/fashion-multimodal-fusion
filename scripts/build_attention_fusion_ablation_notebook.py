#!/usr/bin/env python3
"""Generate notebooks/robustness/AttentionFusion_Ablation_FrozenCLIP_LLaVA.ipynb

Frozen CLIP ViT-B/32 + bert-base-uncased + LLaVA caption table — architecture ablation
over fusion hidden_dim and MultiheadAttention num_heads only (same protocol as
Attention_based_fusion_Robustness_Experiments.ipynb, separate output root).
"""
import json
from pathlib import Path
from textwrap import dedent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(s).strip().split("\n")}


def code(s):
    lines = dedent(s).strip().split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in lines]}


cells = []

cells.append(
    md(
        """
# Attention fusion — architecture ablation (frozen CLIP + LLaVA captions)

This notebook is **separate** from `Attention_based_fusion_Robustness_Experiments.ipynb`. It reproduces the **same multimodal setting** (frozen OpenAI CLIP **ViT-B/32**, frozen **BERT** (`bert-base-uncased`), captions from **`data/processed/LLaVA_caption_dataset_final_full.csv`**, stratified **70 / 15 / 15** splits) but **only** varies:

- **`hidden_dim`** — shared projection width into `nn.MultiheadAttention`
- **`num_heads`** — subject to `hidden_dim % num_heads == 0`

**Ablation grid (5 settings):** see `ABLATION_CONFIGS` in the next cell.

**Seeds (exactly 5):** fixed split seeds in `ABLATION_SEEDS` (default `13, 14, 16, 17, 45` — declare in the paper **before** reading metrics).

**Hardware:** this notebook **requires CUDA** (GPU). It will raise if no GPU is available.

**Metrics (aligned with other fusion experiments):** test and validation summaries use **macro-averaged** precision, recall, and F1 from `sklearn` (`average='macro'`, `zero_division=0`), plus **accuracy** on the test set (`test_accuracy` as a fraction in 0–1 in JSON; multiply by 100 for a percent column in the paper).

**Outputs (this ablation only):** all metrics and checkpoints go under **`experiments/attention_fusion_ablation_frozen_clip_llava/<config_name>/`** — nothing is read from or written to other experiment trees.

**Images:** CSV paths like `dataset/...` are resolved the same way as the Qwen fusion notebook: if the repo has a top-level `dataset/` directory, files are read from there; if not, paths are remapped under `data/raw dataset/` (space in folder name).

**Artifacts:** `metrics/experiments/seed_<N>_results.json` and `artifacts/models/seed_<N>_best_fusion_head.pth`.

**Controls:** set `RUN_ABLATION_GRID = True` to execute all runs; use `RUN_SMOKE_ONLY = True` for one config × one seed. Hyperparameters match the main attention-fusion recipe (LR, batch size, dropout, etc.).
"""
    )
)

cells.append(
    code(
        r"""
import os
import json
import random
import time
import warnings
from datetime import datetime
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

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


def resolve_split_image_path(row_image_path, base_dir):
    # Map CSV image_path to disk: use repo dataset/ or data/raw dataset/ fallback (same as Qwen fusion notebook).
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


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this ablation notebook. "
        "Use a GPU machine or enable GPU in your runtime."
    )
device = torch.device("cuda")
print("device:", device, "|", torch.cuda.get_device_name(0))

# --- same training recipe as Attention_based_fusion_Robustness_Experiments ---
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
MODEL_INIT_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Five ablation settings (hidden_dim, num_heads); names are folder-safe
ABLATION_CONFIGS = [
    {"name": "proj256_h4", "label": "Smaller projection", "hidden_dim": 256, "num_heads": 4},
    {"name": "default_512_h8", "label": "Default", "hidden_dim": 512, "num_heads": 8},
    {"name": "proj768_h8", "label": "Larger projection", "hidden_dim": 768, "num_heads": 8},
    {"name": "heads4_512", "label": "Fewer heads", "hidden_dim": 512, "num_heads": 4},
    {"name": "head1_512", "label": "Single-head", "hidden_dim": 512, "num_heads": 1},
]

# Exactly five fixed split seeds for the ablation table
ABLATION_SEEDS = [13, 14, 16, 17, 45]
assert len(ABLATION_SEEDS) == 5, "Use exactly 5 seeds for the ablation table"

# All runs write only under this tree (metrics + fusion-head checkpoints per setting)
EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "experiments", "attention_fusion_ablation_frozen_clip_llava")
CAPTION_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "LLaVA_caption_dataset_final_full.csv")

RUN_SMOKE_ONLY = False  # True: first config + first seed only
RUN_ABLATION_GRID = False  # True: full nested loops (25 runs if smoke off)

for cfg in ABLATION_CONFIGS:
    if cfg["hidden_dim"] % cfg["num_heads"] != 0:
        raise ValueError(f"Invalid pair: {cfg}")

print("ABLATION_CONFIGS:", len(ABLATION_CONFIGS), "| seeds (n=5):", ABLATION_SEEDS)
print("EXPERIMENT_ROOT (all outputs):", EXPERIMENT_ROOT)
"""
    )
)

cells.append(
    code(
        r"""
# Load caption table (LLaVA-style)
print("Loading caption dataset...")
df = pd.read_csv(CAPTION_CSV)
if "status" in df.columns:
    df_success = df[df["status"] == "success"].copy()
else:
    df_success = df.copy()
print("Rows (success if status present):", len(df_success))

all_styles = sorted(df_success["style"].dropna().astype(str).unique())
style_to_idx = {s: i for i, s in enumerate(all_styles)}
num_classes = len(all_styles)
print("num_classes:", num_classes)

captions_dict = {}
for _, row in df_success.iterrows():
    raw = str(row["image_path"])
    cap = str(row["caption"])
    keys = {raw, os.path.normpath(raw)}
    if not os.path.isabs(raw):
        keys.add(os.path.normpath(os.path.join(PROJECT_ROOT, raw)))
    keys.add(resolve_split_image_path(raw, PROJECT_ROOT))
    for k in keys:
        captions_dict[k] = cap
print("caption dict size:", len(captions_dict))
_ex = str(df_success.iloc[0]["image_path"])
_ex_res = resolve_split_image_path(_ex, PROJECT_ROOT)
print("Sample image_path resolve:", _ex, "->", _ex_res, "| exists:", os.path.isfile(_ex_res))
"""
    )
)

cells.append(
    code(
        r"""
import clip
from transformers import AutoTokenizer, AutoModel

print("Loading CLIP ViT-B/32 (frozen)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("Loading BERT...")
fashionbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fashionbert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
fashionbert_model.eval()
print("Encoders ready.")
"""
    )
)

cells.append(
    code(
        r"""
class FashionMultiModalDataset(Dataset):
    def __init__(self, df, captions_dict, style_to_idx, transform=None, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.captions_dict = captions_dict
        self.style_to_idx = style_to_idx
        self.transform = transform
        self.base_dir = base_dir
        self.valid_indices = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            raw_key = str(row["image_path"])
            image_path = resolve_split_image_path(raw_key, self.base_dir or ".")
            has_file = os.path.isfile(image_path)
            nk = os.path.normpath(raw_key)
            has_caption = (
                raw_key in captions_dict
                or nk in captions_dict
                or image_path in captions_dict
            )
            if has_file and has_caption:
                self.valid_indices.append(idx)
        print(f"  Dataset: {len(self.valid_indices)} / {len(self.df)} usable rows")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        raw_key = str(row["image_path"])
        image_path = resolve_split_image_path(raw_key, self.base_dir or ".")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Missing image file (should be filtered): {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.captions_dict.get(
            raw_key,
            self.captions_dict.get(os.path.normpath(raw_key), self.captions_dict.get(image_path, "")),
        )
        if not caption:
            raise ValueError(f"Missing caption for {raw_key}")
        label = self.style_to_idx[row["style"]]
        return {
            "image": image,
            "caption": caption,
            "label": label,
            "style": row["style"],
            "image_path": image_path,
        }


class AttentionFusion(nn.Module):
    def __init__(self, visual_dim, textual_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.textual_proj = nn.Linear(textual_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, visual_features, textual_features):
        v = self.visual_proj(visual_features)
        t = self.textual_proj(textual_features)
        combined = torch.stack([v, t], dim=1)
        attended, _ = self.attention(combined, combined, combined)
        attended = self.layer_norm(attended)
        fused = torch.mean(attended, dim=1)
        return self.final_proj(fused)


class MultiModalFashionClassifier(nn.Module):
    def __init__(
        self,
        clip_model,
        fashionbert_model,
        num_classes,
        dropout=0.5,
        visual_dim=512,
        textual_dim=768,
        fusion_hidden_dim=512,
        fusion_num_heads=8,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.fashionbert_model = fashionbert_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for p in self.fashionbert_model.parameters():
            p.requires_grad = False
        self.fusion = AttentionFusion(
            visual_dim, textual_dim, hidden_dim=fusion_hidden_dim, num_heads=fusion_num_heads
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 256),
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
            vf = self.clip_model.encode_image(images).float()
            inputs = fashionbert_tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(images.device)
            out = self.fashionbert_model(**inputs)
            tf = out.last_hidden_state[:, 0, :]
        fused = self.fusion(vf, tf)
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
        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss / max(len(train_loader), 1), correct / max(total, 1)


def validate_epoch(model, val_loader, criterion, dev):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_pred, all_lab = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", leave=False):
            images = batch["image"].to(dev)
            captions = batch["caption"]
            labels = batch["label"].to(dev)
            logits = model(images, captions)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_pred.extend(pred.cpu().numpy().tolist())
            all_lab.extend(labels.cpu().numpy().tolist())
    if all_lab:
        macro_f1 = f1_score(all_lab, all_pred, average="macro", zero_division=0)
        macro_p = precision_score(all_lab, all_pred, average="macro", zero_division=0)
        macro_r = recall_score(all_lab, all_pred, average="macro", zero_division=0)
    else:
        macro_f1 = macro_p = macro_r = 0.0
    acc = correct / max(total, 1)
    return total_loss / max(len(val_loader), 1), acc, all_pred, all_lab, macro_f1, macro_p, macro_r


print("Model classes and train/val helpers defined.")
"""
    )
)

cells.append(
    code(
        r"""
def make_seed_worker(base_seed):
    def seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def run_one_ablation(ablation_cfg, seed_value, df_full, base_dir):
    name = ablation_cfg["name"]
    hd = int(ablation_cfg["hidden_dim"])
    nh = int(ablation_cfg["num_heads"])
    root = os.path.join(EXPERIMENT_ROOT, name)
    metrics_dir = os.path.join(root, "metrics", "experiments")
    artifacts_dir = os.path.join(root, "artifacts", "models")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    result_path = os.path.join(metrics_dir, f"seed_{seed_value}_results.json")
    if os.path.isfile(result_path):
        print(f"  skip (exists): {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n=== {name} | hidden_dim={hd} num_heads={nh} | split_seed={seed_value} ===")

    torch.manual_seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    torch.cuda.manual_seed_all(MODEL_INIT_SEED)

    train_df, temp_df = train_test_split(
        df_full,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df_full["style"],
        random_state=seed_value,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["style"],
        random_state=seed_value,
    )
    print(f"  split sizes train/val/test: {len(train_df)} {len(val_df)} {len(test_df)}")

    transform = clip_preprocess

    train_ds = FashionMultiModalDataset(train_df, captions_dict, style_to_idx, transform, base_dir=base_dir)
    val_ds = FashionMultiModalDataset(val_df, captions_dict, style_to_idx, transform, base_dir=base_dir)
    test_ds = FashionMultiModalDataset(test_df, captions_dict, style_to_idx, transform, base_dir=base_dir)

    if len(train_ds) == 0:
        ex_raw = str(train_df.iloc[0]["image_path"]) if len(train_df) > 0 else "<empty train_df>"
        ex_res = resolve_split_image_path(ex_raw, base_dir) if len(train_df) > 0 else ""
        raise RuntimeError(
            "Training dataset has 0 samples after filtering (missing image files or captions). "
            f"First train row image_path={ex_raw!r} resolved={ex_res!r} file_exists={os.path.isfile(ex_res) if ex_res else False}. "
            "Expected images under `<repo>/dataset/` or, when that folder is missing, `<repo>/data/raw dataset/` for paths like `dataset/<style>/...`."
        )

    loader_seed = MODEL_INIT_SEED + int(seed_value)
    g_train = torch.Generator()
    g_train.manual_seed(loader_seed)
    g_val = torch.Generator()
    g_val.manual_seed(loader_seed + 1)
    g_test = torch.Generator()
    g_test.manual_seed(loader_seed + 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        worker_init_fn=make_seed_worker(loader_seed),
        generator=g_train,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=make_seed_worker(loader_seed + 1),
        generator=g_val,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=make_seed_worker(loader_seed + 2),
        generator=g_test,
        pin_memory=True,
    )

    train_valid_df = train_ds.df.iloc[train_ds.valid_indices]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(num_classes),
        y=train_valid_df["style"].map(style_to_idx).values,
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = MultiModalFashionClassifier(
        clip_model,
        fashionbert_model,
        num_classes=num_classes,
        dropout=DROPOUT,
        fusion_hidden_dim=hd,
        fusion_num_heads=nh,
    ).to(device)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(
        n.startswith("fusion.") or n.startswith("classifier.") for n in trainable_names
    ), trainable_names

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        list(model.fusion.parameters()) + list(model.classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    fusion_head_path = os.path.join(artifacts_dir, f"seed_{seed_value}_best_fusion_head.pth")
    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_val_macro_precision = 0.0
    best_val_macro_recall = 0.0
    patience_counter = 0
    early_stopped = False
    train_losses, val_losses = [], []
    val_macro_f1s = []

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        tr_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, _, _, _, va_f1, va_p, va_r = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_macro_f1s.append(va_f1)

        if va_f1 > best_val_macro_f1:
            best_val_macro_f1 = va_f1
            best_val_macro_precision = float(va_p)
            best_val_macro_recall = float(va_r)
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "fusion": model.fusion.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "seed": int(seed_value),
                    "hidden_dim": hd,
                    "num_heads": nh,
                    "best_epoch": int(best_epoch),
                    "best_val_macro_f1": float(best_val_macro_f1),
                },
                fusion_head_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(f"  early stop epoch {epoch + 1}")
            break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}: tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_macro_f1={va_f1:.4f}")

    if os.path.isfile(fusion_head_path):
        ck = torch.load(fusion_head_path, map_location=device)
        model.fusion.load_state_dict(ck["fusion"])
        model.classifier.load_state_dict(ck["classifier"])
    model.eval()
    te_loss, te_acc, te_pred, te_lab, te_f1, te_p, te_r = validate_epoch(model, test_loader, criterion, device)

    elapsed = time.time() - t0
    results = {
        "experiment_id": f"{name}_seed_{seed_value}",
        "ablation_name": name,
        "ablation_label": ablation_cfg.get("label", name),
        "seed_value": int(seed_value),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "fusion_hidden_dim": hd,
            "fusion_num_heads": nh,
            "learning_rate": float(LEARNING_RATE),
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "weight_decay": float(WEIGHT_DECAY),
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "model_init_seed": MODEL_INIT_SEED,
            "data_split_seed": int(seed_value),
            "caption_csv": CAPTION_CSV,
            "image_preprocess": "openai_clip_preprocess_ViT-B_32",
        },
        "validation_metrics": {
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_val_macro_precision": float(best_val_macro_precision),
            "best_val_macro_recall": float(best_val_macro_recall),
            "best_epoch": int(best_epoch),
            "early_stopped": bool(early_stopped),
        },
        "test_metrics": {
            "test_macro_f1": float(te_f1),
            "test_macro_precision": float(te_p),
            "test_macro_recall": float(te_r),
            "test_accuracy": float(te_acc),
            "test_loss": float(te_loss),
        },
        "training_info": {"total_time_seconds": float(elapsed)},
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"  done test_macro_f1={te_f1:.4f} test_macro_p={te_p:.4f} test_macro_r={te_r:.4f} "
        f"test_acc={te_acc:.4f} -> {result_path}"
    )
    return results


print("run_one_ablation defined.")
"""
    )
)

cells.append(
    code(
        r"""
try:
    from IPython.display import display
except ImportError:
    display = print

base_dir = PROJECT_ROOT
df_full = df_success.reset_index(drop=True)

configs = ABLATION_CONFIGS[:1] if RUN_SMOKE_ONLY else ABLATION_CONFIGS
seeds = ABLATION_SEEDS[:1] if RUN_SMOKE_ONLY else ABLATION_SEEDS

rows = []
if not RUN_ABLATION_GRID:
    print("Set RUN_ABLATION_GRID = True (and RUN_SMOKE_ONLY as needed) then re-run this cell.")
else:
    for cfg in configs:
        for sv in seeds:
            r = run_one_ablation(cfg, sv, df_full, base_dir)
            tm = r.get("test_metrics", {})
            vm = r.get("validation_metrics", {})
            rows.append(
                {
                    "setting": cfg["label"],
                    "ablation_name": cfg["name"],
                    "hidden_dim": cfg["hidden_dim"],
                    "num_heads": cfg["num_heads"],
                    "seed": sv,
                    "best_val_macro_f1": vm.get("best_val_macro_f1"),
                    "best_val_macro_precision": vm.get("best_val_macro_precision"),
                    "best_val_macro_recall": vm.get("best_val_macro_recall"),
                    "test_macro_f1": tm.get("test_macro_f1"),
                    "test_macro_precision": tm.get("test_macro_precision"),
                    "test_macro_recall": tm.get("test_macro_recall"),
                    "test_accuracy": tm.get("test_accuracy"),
                }
            )

    ablation_df = pd.DataFrame(rows)
    display(ablation_df)

    summary = (
        ablation_df.groupby(["setting", "ablation_name", "hidden_dim", "num_heads"], sort=False)
        .agg(
            val_f1_mean=("best_val_macro_f1", "mean"),
            val_f1_std=("best_val_macro_f1", "std"),
            val_precision_mean=("best_val_macro_precision", "mean"),
            val_precision_std=("best_val_macro_precision", "std"),
            val_recall_mean=("best_val_macro_recall", "mean"),
            val_recall_std=("best_val_macro_recall", "std"),
            test_f1_mean=("test_macro_f1", "mean"),
            test_f1_std=("test_macro_f1", "std"),
            test_precision_mean=("test_macro_precision", "mean"),
            test_precision_std=("test_macro_precision", "std"),
            test_recall_mean=("test_macro_recall", "mean"),
            test_recall_std=("test_macro_recall", "std"),
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    display(summary)
    out_csv = os.path.join(EXPERIMENT_ROOT, "ablation_summary_by_setting.csv")
    summary.to_csv(out_csv, index=False)
    print("Wrote", out_csv)
"""
    )
)

cells.append(
    code(
        r"""
# --- Optional: backfill test_macro_precision / test_macro_recall from checkpoints (no training) ---
# Works even if validate_epoch still returns 5 values (older kernel): P/R computed from preds/labels.
FORCE_BACKFILL_TEST_METRICS = False

try:
    from IPython.display import display
except ImportError:
    display = print

base_dir_bf = PROJECT_ROOT
df_full_bf = df_success.reset_index(drop=True)

for cfg in ABLATION_CONFIGS:
    name = cfg["name"]
    hd, nh = int(cfg["hidden_dim"]), int(cfg["num_heads"])
    root = os.path.join(EXPERIMENT_ROOT, name)
    metrics_dir = os.path.join(root, "metrics", "experiments")
    artifacts_dir = os.path.join(root, "artifacts", "models")

    for seed_value in ABLATION_SEEDS:
        result_path = os.path.join(metrics_dir, f"seed_{seed_value}_results.json")
        fusion_head_path = os.path.join(artifacts_dir, f"seed_{seed_value}_best_fusion_head.pth")

        if not os.path.isfile(result_path):
            print(f"SKIP (no json): {result_path}")
            continue
        if not os.path.isfile(fusion_head_path):
            print(f"SKIP (no checkpoint): {fusion_head_path}")
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tm = data.get("test_metrics") or {}
        if tm.get("test_macro_precision") is not None and not FORCE_BACKFILL_TEST_METRICS:
            print(f"SKIP (already has test_macro_precision): {name} seed={seed_value}")
            continue

        torch.manual_seed(MODEL_INIT_SEED)
        np.random.seed(MODEL_INIT_SEED)
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)

        train_df, temp_df = train_test_split(
            df_full_bf,
            test_size=(VAL_RATIO + TEST_RATIO),
            stratify=df_full_bf["style"],
            random_state=seed_value,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=temp_df["style"],
            random_state=seed_value,
        )

        transform = clip_preprocess
        train_ds = FashionMultiModalDataset(train_df, captions_dict, style_to_idx, transform, base_dir=base_dir_bf)
        test_ds = FashionMultiModalDataset(test_df, captions_dict, style_to_idx, transform, base_dir=base_dir_bf)

        train_valid_df = train_ds.df.iloc[train_ds.valid_indices]
        class_weights = compute_class_weight(
            "balanced",
            classes=np.arange(num_classes),
            y=train_valid_df["style"].map(style_to_idx).values,
        )
        class_weights = torch.FloatTensor(class_weights).to(device)

        loader_seed = MODEL_INIT_SEED + int(seed_value)
        g_test = torch.Generator()
        g_test.manual_seed(loader_seed + 2)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            worker_init_fn=make_seed_worker(loader_seed + 2),
            generator=g_test,
            pin_memory=True,
        )

        model = MultiModalFashionClassifier(
            clip_model,
            fashionbert_model,
            num_classes=num_classes,
            dropout=DROPOUT,
            fusion_hidden_dim=hd,
            fusion_num_heads=nh,
        ).to(device)
        ck = torch.load(fusion_head_path, map_location=device)
        model.fusion.load_state_dict(ck["fusion"])
        model.classifier.load_state_dict(ck["classifier"])
        model.eval()

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        _out = validate_epoch(model, test_loader, criterion, device)
        if len(_out) == 7:
            te_loss, te_acc, te_pred, te_lab, te_f1, te_p, te_r = _out
        else:
            te_loss, te_acc, te_pred, te_lab, te_f1 = _out
            if te_lab:
                te_p = precision_score(te_lab, te_pred, average="macro", zero_division=0)
                te_r = recall_score(te_lab, te_pred, average="macro", zero_division=0)
            else:
                te_p = te_r = 0.0

        old_f1 = tm.get("test_macro_f1")
        if old_f1 is not None and abs(float(old_f1) - float(te_f1)) > 0.02:
            print(
                f"  WARN {name} seed={seed_value}: test_macro_f1 drift "
                f"json={float(old_f1):.4f} reeval={float(te_f1):.4f}"
            )

        data.setdefault("test_metrics", {})
        data["test_metrics"].update(
            {
                "test_macro_f1": float(te_f1),
                "test_macro_precision": float(te_p),
                "test_macro_recall": float(te_r),
                "test_accuracy": float(te_acc),
                "test_loss": float(te_loss),
            }
        )
        data["backfill_test_metrics"] = {
            "timestamp": datetime.now().isoformat(),
            "note": "eval-only backfill cell",
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(
            f"OK {name} seed={seed_value} | test_macro_f1={te_f1:.4f} "
            f"P={te_p:.4f} R={te_r:.4f} acc={te_acc:.4f}"
        )

all_rows = []
for cfg in ABLATION_CONFIGS:
    name = cfg["name"]
    hd, nh = cfg["hidden_dim"], cfg["num_heads"]
    for seed_value in ABLATION_SEEDS:
        result_path = os.path.join(EXPERIMENT_ROOT, name, "metrics", "experiments", f"seed_{seed_value}_results.json")
        if not os.path.isfile(result_path):
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            r = json.load(f)
        tm = r.get("test_metrics") or {}
        vm = r.get("validation_metrics") or {}
        all_rows.append(
            {
                "setting": cfg["label"],
                "ablation_name": name,
                "hidden_dim": hd,
                "num_heads": nh,
                "seed": seed_value,
                "best_val_macro_f1": vm.get("best_val_macro_f1"),
                "best_val_macro_precision": vm.get("best_val_macro_precision"),
                "best_val_macro_recall": vm.get("best_val_macro_recall"),
                "test_macro_f1": tm.get("test_macro_f1"),
                "test_macro_precision": tm.get("test_macro_precision"),
                "test_macro_recall": tm.get("test_macro_recall"),
                "test_accuracy": tm.get("test_accuracy"),
            }
        )

ablation_df_bf = pd.DataFrame(all_rows)
display(ablation_df_bf)

summary_bf = (
    ablation_df_bf.groupby(["setting", "ablation_name", "hidden_dim", "num_heads"], sort=False)
    .agg(
        val_f1_mean=("best_val_macro_f1", "mean"),
        val_f1_std=("best_val_macro_f1", "std"),
        val_precision_mean=("best_val_macro_precision", "mean"),
        val_precision_std=("best_val_macro_precision", "std"),
        val_recall_mean=("best_val_macro_recall", "mean"),
        val_recall_std=("best_val_macro_recall", "std"),
        test_f1_mean=("test_macro_f1", "mean"),
        test_f1_std=("test_macro_f1", "std"),
        test_precision_mean=("test_macro_precision", "mean"),
        test_precision_std=("test_macro_precision", "std"),
        test_recall_mean=("test_macro_recall", "mean"),
        test_recall_std=("test_macro_recall", "std"),
        test_acc_mean=("test_accuracy", "mean"),
        test_acc_std=("test_accuracy", "std"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)
display(summary_bf)

out_csv_bf = os.path.join(EXPERIMENT_ROOT, "ablation_summary_by_setting.csv")
summary_bf.to_csv(out_csv_bf, index=False)
print("Wrote", out_csv_bf)
"""
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

out = Path(__file__).resolve().parent.parent / "notebooks" / "robustness" / "AttentionFusion_Ablation_FrozenCLIP_LLaVA.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Wrote", out)
