#!/usr/bin/env python3
"""Generator for TextOnly_Qwen_StratifiedSplits_Robustness.ipynb — Qwen captions, legacy stratified splits."""
import json
import re
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
# Text-only robustness (Qwen captions, **legacy stratified splits**)

**Row table for `train_test_split`:** **`data/processed/LLaVA_caption_dataset_final_full.csv`** (success rows), or legacy **`caption_dataset_final_full.csv`**. Same **70% / 15% / 15%** and seeds as **`TextOnly_LLaVA_StratifiedSplits_Robustness.ipynb`**.

**Qwen file (`qwen25vl_caption_full.csv`)** is used only for **caption text** (lookup by `image_path`), not for defining which rows enter the stratified split. The two CSVs are *not* interchangeable for splits unless every `image_path` (and row set) matches the LLaVA table exactly — in practice the LLaVA table defines parity with the LLaVA notebook.

**Override:** env **`FASHION_STRATIFIED_TABLE_CSV`** → path to a table with `image_path`, `style`, `status` (same role as the LLaVA file).

If the processed LLaVA file is missing, the notebook **errors** with both default paths; fix paths or set the env var (do not silently use the Qwen CSV for splits).

Use this notebook for **fair comparison** with LLaVA text-only and with **`AttentionFusion_FinetunedCLIP_Qwen_Robustness.ipynb`** when that notebook is in **`SPLIT_MODE = "stratified"`**.

For **fixed `data/splits/seed_*`** (aligned with CSV-finetuned CLIP / image-only), use **`AttentionFusion_FinetunedCLIP_Qwen_Robustness.ipynb`** with **`SPLIT_MODE = "csv"`**.

## Outputs
`experiments/textonly_qwen_stratified_splits_robustness/`

Set **`RUN_ALL_SEEDS = True`** for all seeds in `seeds_list.txt`, or **`False`** + **`SMOKE_SEED`** for one run.

After pulling or regenerating this notebook, use **Kernel → Restart** (then Run All). Otherwise Jupyter can still run an **old cached** first cell (e.g. the removed CAPTION_CSV split fallback).

## Notes
- **`seeds_list.txt`:** seeds are read only from lines matching `N. Seed <id>` (the 30 protocol entries). Header numbers like `Total seeds: 30` are ignored.
- Only the **classifier** is optimized; BERT stays effectively frozen (see model `train()` override in the code cell).
"""
    )
)

cells.append(
    code(
        r"""
import os
import re
import json
import time
import warnings
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

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

CAPTION_CSV = os.path.join(PROJECT_ROOT, "data", "captions", "qwen25vl_caption_full.csv")
if not os.path.isfile(CAPTION_CSV):
    raise FileNotFoundError(f"Missing Qwen caption CSV: {CAPTION_CSV}")

VAL_RATIO = 0.15
TEST_RATIO = 0.15
_llava_t = os.path.join(PROJECT_ROOT, "data", "processed", "LLaVA_caption_dataset_final_full.csv")
_legacy_t = os.path.join(PROJECT_ROOT, "data", "processed", "caption_dataset_final_full.csv")
if os.path.isfile(_llava_t):
    _default_table = _llava_t
elif os.path.isfile(_legacy_t):
    _default_table = _legacy_t
else:
    _default_table = _llava_t
LEGACY_TABLE_CSV = os.environ.get("FASHION_STRATIFIED_TABLE_CSV", _default_table).strip()
if LEGACY_TABLE_CSV and not os.path.isabs(LEGACY_TABLE_CSV):
    LEGACY_TABLE_CSV = os.path.normpath(os.path.join(PROJECT_ROOT, LEGACY_TABLE_CSV))

if not os.path.isfile(LEGACY_TABLE_CSV):
    raise FileNotFoundError(
        "Stratified split table not found.\n"
        "  Checked (defaults):\n    "
        + _llava_t
        + "\n    "
        + _legacy_t
        + "\n  Resolved LEGACY_TABLE_CSV (after env FASHION_STRATIFIED_TABLE_CSV):\n    "
        + repr(LEGACY_TABLE_CSV)
        + "\n  Add LLaVA_caption_dataset_final_full.csv under data/processed/, or set FASHION_STRATIFIED_TABLE_CSV "
        "to a CSV with the same columns as the LLaVA stratified notebook (image_path, style, status).\n"
        "  The Qwen caption file is only used for **text**; it does not replace the LLaVA table for row membership."
    )

print("Stratified row table:", LEGACY_TABLE_CSV)
DF_FULL = pd.read_csv(LEGACY_TABLE_CSV)
if "status" in DF_FULL.columns:
    DF_FULL = DF_FULL[DF_FULL["status"].astype(str).str.lower() == "success"].copy()
DF_FULL["style"] = DF_FULL["style"].astype(str).str.strip()
all_styles = sorted(DF_FULL["style"].dropna().unique())
style_to_idx = {s: i for i, s in enumerate(all_styles)}
num_classes = len(all_styles)
print("Legacy table rows:", len(DF_FULL), "| num_classes:", num_classes)

EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "experiments", "textonly_qwen_stratified_splits_robustness")
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

RUN_ALL_SEEDS = False
SMOKE_SEED = 13
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
MODEL_INIT_SEED = 42
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
print("Device:", device)
print("CAPTION_CSV:", CAPTION_CSV)
print("EXPERIMENT_ROOT:", EXPERIMENT_ROOT)

seeds_file = os.path.join(PROJECT_ROOT, "data", "processed", "seeds_list.txt")
SEEDS = []
if os.path.isfile(seeds_file):
    with open(seeds_file) as f:
        content = f.read()
    # Prefer "N. Seed <id>" lines (data/processed/seeds_list.txt); do not scrape every digit
    # (would pull 1, 30, 42, 500, line indices, etc. and inflate the count).
    listed = re.findall(r"(?m)^\s*\d+\.\s*Seed\s+(\d+)\s*$", content)
    for num_str in listed:
        v = int(num_str)
        if 1 <= v <= 500:
            SEEDS.append(v)
    if not SEEDS:
        for num_str in re.findall(r"\d+", content):
            try:
                v = int(num_str)
                if 1 <= v <= 500:
                    SEEDS.append(v)
            except ValueError:
                pass
        SEEDS = sorted(set(SEEDS))
        print(f"Warning: no 'N. Seed <id>' lines in {seeds_file}; fell back to digit scan -> {len(SEEDS)} values")
    else:
        SEEDS = sorted(set(SEEDS))
    print(f"Loaded {len(SEEDS)} seeds from {seeds_file}")
else:
    print("Warning: seeds_list.txt missing — SEEDS empty")

with open(os.path.join(METRICS_DIR, "seeds_list.txt"), "w") as f:
    f.write("Seeds from data/processed/seeds_list.txt (stratified protocol)\n")
    for s in SEEDS:
        f.write(f"{s}\n")


def resolve_split_image_path(row_image_path, base_dir):
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


def load_caption_dict(csv_path: str, base_dir: str):
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()
    if "caption" not in df.columns or "image_path" not in df.columns:
        raise ValueError(f"Expected image_path and caption columns; got {list(df.columns)}")

    def register(d, raw_path, text):
        t = str(text).strip()
        if not t:
            return
        p = str(raw_path)
        d[p] = t
        d[os.path.normpath(p)] = t
        if base_dir and not os.path.isabs(p):
            d[os.path.normpath(os.path.join(base_dir, p))] = t
        res = resolve_split_image_path(p, base_dir or ".")
        d[res] = t

    out = {}
    for _, row in df.iterrows():
        register(out, row["image_path"], row["caption"])
    return out


BASE_DIR = PROJECT_ROOT
captions_dict = load_caption_dict(CAPTION_CSV, BASE_DIR)
print("Caption dict entries (with path aliases):", len(captions_dict))

"""
    )
)

cells.append(md("## Load BERT"))

cells.append(
    code(
        r"""
print("Loading BERT …")
fashionbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fashionbert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
fashionbert_model.eval()
print("BERT ready.")
"""
    )
)

cells.append(md("## Dataset, model, train/val"))

cells.append(
    code(
        r"""
class FashionTextOnlyCSVDataset(Dataset):
    def _caption_lookup(self, raw):
        r = str(raw)
        keys = (r, os.path.normpath(r), resolve_split_image_path(r, self.base_dir or "."))
        for k in keys:
            if k in self.captions_dict:
                return self.captions_dict[k]
        return None

    def __init__(self, df, captions_dict, style_to_idx, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.captions_dict = captions_dict
        self.style_to_idx = style_to_idx
        self.base_dir = base_dir
        self.valid_indices = []
        missing = 0
        for i in range(len(self.df)):
            raw = str(self.df.iloc[i]["image_path"])
            if self._caption_lookup(raw) is None:
                missing += 1
                continue
            self.valid_indices.append(i)
        print(f"  TextOnly split: {len(self.valid_indices)} / {len(self.df)} rows with caption | missing: {missing}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        j = self.valid_indices[idx]
        row = self.df.iloc[j]
        raw = str(row["image_path"])
        cap = self._caption_lookup(raw)
        if cap is None:
            raise RuntimeError("missing caption after filter")
        sty = str(row["style"]).strip()
        return {"caption": cap, "label": self.style_to_idx[sty], "style": sty, "image_path": raw}


class TextOnlyFashionClassifier(nn.Module):
    def __init__(self, bert_model, tokenizer, num_classes, dropout=0.5, dev=None):
        super().__init__()
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.dev = dev or torch.device("cpu")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, captions):
        with torch.no_grad():
            tok = self.tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.dev)
            h = self.bert(**tok).last_hidden_state[:, 0, :]
        return self.classifier(h)


def train_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    tot_loss = 0.0
    correct = 0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        y = batch["label"].to(dev)
        optimizer.zero_grad()
        logits = model(batch["caption"])
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        pred = logits.argmax(1)
        n += y.size(0)
        correct += (pred == y).sum().item()
    return tot_loss / max(len(loader), 1), correct / max(n, 1)


def validate_epoch(model, loader, criterion, dev):
    model.eval()
    tot_loss = 0.0
    correct = 0
    n = 0
    preds, labs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            y = batch["label"].to(dev)
            logits = model(batch["caption"])
            loss = criterion(logits, y)
            tot_loss += loss.item()
            pr = logits.argmax(1)
            n += y.size(0)
            correct += (pr == y).sum().item()
            preds.extend(pr.cpu().numpy().tolist())
            labs.extend(y.cpu().numpy().tolist())
    f1m = f1_score(labs, preds, average="macro", zero_division=0) if preds else 0.0
    acc = correct / max(n, 1)
    return tot_loss / max(len(loader), 1), acc, preds, labs, f1m


def evaluate_test(model, loader, criterion, dev, idx_to_style, n_cls):
    model.eval()
    tot_loss = 0.0
    correct = 0
    n = 0
    preds, labs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test", leave=False):
            y = batch["label"].to(dev)
            logits = model(batch["caption"])
            loss = criterion(logits, y)
            tot_loss += loss.item()
            pr = logits.argmax(1)
            n += y.size(0)
            correct += (pr == y).sum().item()
            preds.extend(pr.cpu().numpy().tolist())
            labs.extend(y.cpu().numpy().tolist())
    macro_f1 = f1_score(labs, preds, average="macro", zero_division=0)
    macro_p = precision_score(labs, preds, average="macro", zero_division=0)
    macro_r = recall_score(labs, preds, average="macro", zero_division=0)
    acc = correct / max(n, 1)
    labels_all = list(range(n_cls))
    per_f1 = f1_score(labs, preds, labels=labels_all, average=None, zero_division=0)
    per_class_f1 = {idx_to_style[i]: float(per_f1[i]) for i in range(n_cls)}
    return tot_loss / max(len(loader), 1), acc, macro_f1, macro_p, macro_r, preds, labs, per_class_f1


print("Classes defined.")
"""
    )
)

cells.append(md("## Runner (stratified splits)"))

cells.append(
    code(
        r"""
idx_to_style = {v: k for k, v in style_to_idx.items()}


def run_one_seed(seed_value, seed_idx):
    print(f"\n{'='*60}\nSeed {seed_idx}/{len(SEEDS)}: {seed_value}\n{'='*60}")
    out_json = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
    if os.path.isfile(out_json):
        print("  Skip (exists):", out_json)
        with open(out_json) as f:
            return json.load(f)

    train_df, temp_df = train_test_split(
        DF_FULL,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=DF_FULL["style"],
        random_state=seed_value,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["style"],
        random_state=seed_value,
    )
    print("  Split sizes:", len(train_df), len(val_df), len(test_df))

    train_ds = FashionTextOnlyCSVDataset(train_df, captions_dict, style_to_idx, base_dir=BASE_DIR)
    val_ds = FashionTextOnlyCSVDataset(val_df, captions_dict, style_to_idx, base_dir=BASE_DIR)
    test_ds = FashionTextOnlyCSVDataset(test_df, captions_dict, style_to_idx, base_dir=BASE_DIR)

    if len(train_ds) == 0:
        raise RuntimeError("Empty train set — check Qwen caption keys vs legacy image_path")

    g = torch.Generator()
    g.manual_seed(MODEL_INIT_SEED + int(seed_value))
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_labels = train_ds.df.iloc[train_ds.valid_indices]["style"].map(style_to_idx).values
    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)
    cw_t = torch.FloatTensor(cw).to(device)

    torch.manual_seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)

    model = TextOnlyFashionClassifier(fashionbert_model, fashionbert_tokenizer, num_classes, DROPOUT, device).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f1 = -1.0
    best_ep = 0
    pat = 0
    train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []

    for epoch in range(MAX_EPOCHS):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, _, _, vf1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(float(ta))
        val_accs.append(float(va))
        val_f1s.append(float(vf1))
        if vf1 > best_f1:
            best_f1 = vf1
            best_ep = epoch + 1
            pat = 0
            torch.save(model.classifier.state_dict(), os.path.join(ARTIFACTS_DIR, "models", f"seed_{seed_value}_best_classifier.pth"))
        else:
            pat += 1
        if pat >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stop epoch {epoch+1}")
            break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1}: tr_loss={tl:.4f} val_macro_f1={vf1:.4f}")

    ck = os.path.join(ARTIFACTS_DIR, "models", f"seed_{seed_value}_best_classifier.pth")
    if os.path.isfile(ck):
        model.classifier.load_state_dict(torch.load(ck, map_location=device))
    model.eval()

    te_loss, te_acc, te_f1, te_p, te_r, te_pred, te_lab, te_per = evaluate_test(
        model, test_loader, criterion, device, idx_to_style, num_classes
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_losses, label="train")
    ax[0].plot(val_losses, label="val")
    ax[0].legend()
    ax[0].set_title("Loss")
    ax[1].plot(val_f1s, label="val macro F1")
    ax[1].legend()
    ax[1].set_title("Val macro F1")
    fig.suptitle(f"Text-only Qwen (stratified splits) | seed {seed_value}")
    plt.tight_layout()
    lc_path = os.path.join(ARTIFACTS_DIR, "learning_curves", f"seed_{seed_value}_learning_curves.png")
    plt.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "experiment": "textonly_qwen_stratified",
        "caption_source": "qwen",
        "caption_csv": CAPTION_CSV,
        "legacy_split_table": LEGACY_TABLE_CSV,
        "data_split_seed": int(seed_value),
        "split_protocol": "stratified_train_test_split_70_15_15",
        "test_metrics": {
            "test_macro_f1": float(te_f1),
            "test_accuracy": float(te_acc),
            "test_loss": float(te_loss),
            "test_macro_precision": float(te_p),
            "test_macro_recall": float(te_r),
            "per_class_f1": te_per,
            "test_predictions": [int(x) for x in te_pred],
            "test_labels": [int(x) for x in te_lab],
        },
        "validation_metrics": {"best_val_macro_f1": float(best_f1), "best_epoch": int(best_ep)},
        "configuration": {
            "learning_rate": float(LEARNING_RATE),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "dropout": DROPOUT,
            "weight_decay": float(WEIGHT_DECAY),
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Done seed {seed_value} | test macro F1={te_f1:.4f} | acc={te_acc:.4f}")
    return results


if RUN_ALL_SEEDS:
    for i, sv in enumerate(SEEDS, 1):
        run_one_seed(sv, i)
else:
    run_one_seed(SMOKE_SEED, 1)

print("Finished.")
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

out = Path(__file__).resolve().parent.parent / "notebooks" / "robustness" / "TextOnly_Qwen_StratifiedSplits_Robustness.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote", out)
