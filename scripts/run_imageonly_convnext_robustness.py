"""Run ConvNeXt-only 30-seed robustness (same logic as ImageOnly_ConvNext.ipynb)."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

LEARNING_RATE = 5e-5
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
MODEL_INIT_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CONVNEXT_MODEL_NAME = "convnext_base"


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    repo = here.parent
    if (repo / "notebooks" / "robustness").is_dir():
        return repo
    raise FileNotFoundError("Could not locate fashion-multimodal-fusion repo root.")


def find_data_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "FusionStyle" / "FashionStyle14_v1",
        repo_root / "FusionStyle" / "FashionStyle14_v1",
    ]
    for data_dir in candidates:
        if (data_dir / "dataset").is_dir() and (data_dir / "complete_dataset.csv").is_file():
            return data_dir.resolve()
    raise FileNotFoundError("Could not locate FusionStyle/FashionStyle14_v1.")


def load_seeds(seeds_file: Path) -> List[int]:
    content = seeds_file.read_text(encoding="utf-8")
    matches = re.findall(r"Seed\s+(\d+)", content, flags=re.IGNORECASE)
    seeds = sorted({int(s) for s in matches if 1 <= int(s) <= 500})
    return seeds[:30]


class FashionImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, style_to_idx: Dict[str, int], transform):
        self.frame = frame.reset_index(drop=True)
        self.style_to_idx = style_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.frame.iloc[idx]
        try:
            img = Image.open(row["abs_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return {"pixel_values": self.transform(img), "label": self.style_to_idx[row["style"]]}


class ConvNeXtImageClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(CONVNEXT_MODEL_NAME, pretrained=True, num_classes=0)
        in_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def main() -> None:
    repo_root = find_repo_root()
    os.chdir(repo_root)
    data_dir = find_data_dir(repo_root)
    results_root = repo_root / "results" / "imageonly_convnext"
    results_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    batch_size = BATCH_SIZE
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb < 8:
            batch_size = 8
        elif mem_gb < 12:
            batch_size = 16

    random.seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    torch.manual_seed(MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)

    seeds = load_seeds(data_dir / "seeds_list.txt")
    print(f"REPO_ROOT: {repo_root}")
    print(f"DATA_DIR: {data_dir}")
    print(f"RESULTS_ROOT: {results_root}")
    print(f"Device: {device}, batch_size: {batch_size}, seeds: {len(seeds)}")

    lines = (data_dir / "complete_dataset.csv").read_text(encoding="utf-8").splitlines()
    rel = [ln.strip().replace("\\", "/") for ln in lines if ln.strip()]
    df_full = pd.DataFrame({"rel_path": rel})
    df_full["style"] = df_full["rel_path"].str.split("/").str[1]
    df_full["abs_path"] = df_full["rel_path"].apply(
        lambda r: str((data_dir / r.replace("/", os.sep)).resolve())
    )
    df_full = df_full[df_full["abs_path"].map(os.path.isfile)].reset_index(drop=True)
    classes = sorted(df_full["style"].unique().tolist())
    style_to_idx = {s: i for i, s in enumerate(classes)}
    num_classes = len(classes)

    transform = timm.data.create_transform(
        **timm.data.resolve_data_config({}, model=CONVNEXT_MODEL_NAME, verbose=False),
        is_training=False,
    )

    summary_rows: List[Dict[str, Any]] = []

    for seed_idx, seed_value in enumerate(seeds, start=1):
        seed_dir = results_root / f"seed_{seed_value}"
        done_marker = seed_dir / "test_metrics.json"
        if done_marker.is_file():
            print(f"[ConvNeXt] Seed {seed_value} ({seed_idx}/{len(seeds)}): skip (already done)")
            with open(done_marker, encoding="utf-8") as f:
                summary_rows.append({"seed": seed_value, **json.load(f)})
            continue

        print("=" * 70)
        print(f"[ConvNeXt] Experiment {seed_idx}/{len(seeds)} | data split seed = {seed_value}")
        print("=" * 70)

        train_df, temp_df = train_test_split(
            df_full, test_size=(VAL_RATIO + TEST_RATIO), stratify=df_full["style"], random_state=seed_value
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=temp_df["style"],
            random_state=seed_value,
        )

        def loaders(frame_train, frame_val, frame_test):
            return (
                DataLoader(FashionImageDataset(frame_train, style_to_idx, transform), batch_size=batch_size, shuffle=True),
                DataLoader(FashionImageDataset(frame_val, style_to_idx, transform), batch_size=batch_size, shuffle=False),
                DataLoader(FashionImageDataset(frame_test, style_to_idx, transform), batch_size=batch_size, shuffle=False),
            )

        train_loader, val_loader, test_loader = loaders(train_df, val_df, test_df)

        random.seed(MODEL_INIT_SEED)
        np.random.seed(MODEL_INIT_SEED)
        torch.manual_seed(MODEL_INIT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(MODEL_INIT_SEED)

        model = ConvNeXtImageClassifier(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "train_acc": [], "val_acc": []}
        best_f1, best_state, patience = -1.0, None, EARLY_STOPPING_PATIENCE

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            tr_loss = tr_correct = tr_total = 0.0
            for batch in train_loader:
                x = batch["pixel_values"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                tr_loss += float(loss.item()) * y.size(0)
                tr_correct += int((logits.argmax(1) == y).sum().item())
                tr_total += int(y.size(0))
            tr_loss /= max(tr_total, 1)
            tr_acc = tr_correct / max(tr_total, 1)

            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["pixel_values"].to(device)
                    y = batch["label"].to(device)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    val_loss += float(loss.item()) * y.size(0)
                    val_preds.extend(logits.argmax(1).cpu().tolist())
                    val_labels.extend(y.cpu().tolist())
            val_loss /= len(val_loader.dataset)
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(val_labels, val_preds)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_f1)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)
            print(f"  Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} macroF1 {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience = EARLY_STOPPING_PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    print("  Early stopping triggered.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), seed_dir / "best_model.pt")
        with open(seed_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[1].plot(epochs, history["val_macro_f1"], label="val macro F1")
        fig.savefig(seed_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["pixel_values"].to(device)
                y = batch["label"].to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                test_preds.extend(logits.argmax(1).cpu().tolist())
                test_labels.extend(y.cpu().tolist())

        test_metrics = {
            "accuracy": float(accuracy_score(test_labels, test_preds)),
            "macro_precision": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
        }
        with open(seed_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
        per_class_p = precision_score(test_labels, test_preds, average=None, zero_division=0)
        per_class_r = recall_score(test_labels, test_preds, average=None, zero_division=0)
        pd.DataFrame(
            {
                "class": classes,
                "acc": per_class_r,
                "precision": per_class_p,
                "recall": per_class_r,
                "f1": per_class_f1,
            }
        ).to_csv(seed_dir / "per_class.csv", index=False, encoding="utf-8")

        print(f"[ConvNeXt | seed {seed_value}] Test metrics: {test_metrics}")
        summary_rows.append({"seed": seed_value, **test_metrics})

        del model, train_loader, val_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "all_seeds_summary.csv", index=False, encoding="utf-8")
    print(f"Saved summary: {results_root / 'all_seeds_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
