#!/usr/bin/env python3
"""
One-seed pilot for partial CLIP fine-tuning on the image-only fashion task.

This script is meant for hyperparameter tuning before launching a full
30-seed robustness run. It follows the repo's CSV-backed split workflow:

    data/splits/seed_<seed>/{train,val,test}.csv

Outputs are written under:

    experiments/imageonly_clip_finetuned_pilot/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "imageonly_clip_finetuned_pilot"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "caption_dataset_final_full.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-seed partial CLIP fine-tuning pilot for image-only fashion classification."
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed id to load from data/splits/seed_<seed>/")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5, help="Learning rate for classifier head")
    parser.add_argument("--clip-lr", type=float, default=1e-5, help="Learning rate for unfrozen CLIP visual layers")
    parser.add_argument(
        "--unfreeze-visual-blocks",
        type=int,
        default=2,
        help="How many final CLIP visual transformer blocks to unfreeze",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--debug-samples", type=int, default=5, help="How many sample paths to sanity-check")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run all data/model/path sanity checks, then exit before training",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root used for relative image path resolution",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_path: str, base_dir: Path) -> str:
    image_path = str(image_path)

    if not os.path.isabs(image_path):
        rel_path = image_path.replace("\\", "/")
        dataset_root = base_dir / "dataset"
        if rel_path.startswith("dataset/") and not dataset_root.is_dir():
            image_path = str(base_dir / "data" / "raw dataset" / rel_path[len("dataset/") :])
        else:
            image_path = str(base_dir / image_path)

    if "%" in image_path:
        parts = image_path.split("/")
        image_path = "/".join(unquote(part) if "%" in part else part for part in parts)

    return os.path.normpath(image_path)


class FashionImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, style_to_idx: dict[str, int], transform=None, base_dir: Path | None = None):
        self.df = df.reset_index(drop=True)
        self.style_to_idx = style_to_idx
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.valid_indices: list[int] = []
        self.missing_examples: list[tuple[str, str]] = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            resolved_path = resolve_image_path(row["image_path"], self.base_dir or PROJECT_ROOT)
            if os.path.exists(resolved_path):
                self.valid_indices.append(idx)
            elif len(self.missing_examples) < 5:
                self.missing_examples.append((str(row["image_path"]), resolved_path))

        print(
            f"Dataset initialized with {len(self.valid_indices)} valid samples "
            f"(out of {len(self.df)})"
        )
        if self.missing_examples:
            print("  Missing image examples:")
            for raw_path, resolved_path in self.missing_examples:
                print(f"    raw={raw_path}")
                print(f"    resolved={resolved_path}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, object]:
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        image_path = resolve_image_path(row["image_path"], self.base_dir or PROJECT_ROOT)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        style = row["style"]
        label = self.style_to_idx[style]

        return {
            "image": image,
            "label": label,
            "style": style,
            "image_path": image_path,
        }


class ImageOnlyFashionClassifier(nn.Module):
    def __init__(self, clip_model, num_classes: int, dropout: float = 0.5, visual_dim: int = 512):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual_features = self.clip_model.encode_image(images).float()
        return self.classifier(visual_features)


def configure_partial_visual_finetuning(model: ImageOnlyFashionClassifier, unfreeze_visual_blocks: int) -> dict[str, object]:
    for param in model.clip_model.parameters():
        param.requires_grad = False

    visual = model.clip_model.visual
    trainable_names: list[str] = []

    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = list(visual.transformer.resblocks)
        if not blocks:
            raise RuntimeError("CLIP visual transformer has no resblocks to unfreeze.")

        unfreeze_visual_blocks = max(0, min(unfreeze_visual_blocks, len(blocks)))
        for block_idx in range(len(blocks) - unfreeze_visual_blocks, len(blocks)):
            if block_idx < 0:
                continue
            for name, param in blocks[block_idx].named_parameters():
                param.requires_grad = True
                trainable_names.append(f"clip_model.visual.transformer.resblocks.{block_idx}.{name}")

    if hasattr(visual, "ln_post"):
        for name, param in visual.ln_post.named_parameters():
            param.requires_grad = True
            trainable_names.append(f"clip_model.visual.ln_post.{name}")

    if hasattr(visual, "proj") and isinstance(visual.proj, torch.nn.Parameter):
        visual.proj.requires_grad = True
        trainable_names.append("clip_model.visual.proj")

    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
        trainable_names.append(f"classifier.{name}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(trainable_params),
        "total_parameter_count": int(total_params),
        "unfrozen_visual_blocks": int(unfreeze_visual_blocks),
    }


def build_optimizer(model: ImageOnlyFashionClassifier, clip_lr: float, head_lr: float, weight_decay: float):
    clip_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("clip_model.visual"):
            clip_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if clip_params:
        param_groups.append({"params": clip_params, "lr": clip_lr, "name": "clip"})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})

    if not param_groups:
        raise RuntimeError("No trainable parameters were found.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(data_loader), 100.0 * correct / total


def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return total_loss / len(data_loader), 100.0 * correct / total, all_predictions, all_labels, macro_f1


def evaluate_with_per_class_metrics(model, data_loader, criterion, device, all_styles):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(data_loader)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)

    per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)

    per_class_accuracy = []
    labels_np = np.array(all_labels)
    predictions_np = np.array(all_predictions)
    for class_idx in range(len(all_styles)):
        class_mask = labels_np == class_idx
        class_acc = float(np.sum((predictions_np == class_idx) & class_mask) / np.sum(class_mask)) if np.sum(class_mask) > 0 else 0.0
        per_class_accuracy.append(class_acc)

    per_class_f1_dict = {all_styles[i]: float(per_class_f1[i]) for i in range(len(all_styles))}
    per_class_precision_dict = {all_styles[i]: float(per_class_precision[i]) for i in range(len(all_styles))}
    per_class_recall_dict = {all_styles[i]: float(per_class_recall[i]) for i in range(len(all_styles))}
    per_class_accuracy_dict = {all_styles[i]: float(per_class_accuracy[i]) for i in range(len(all_styles))}

    return (
        avg_loss,
        accuracy,
        all_predictions,
        all_labels,
        macro_f1,
        macro_precision,
        macro_recall,
        per_class_f1_dict,
        per_class_precision_dict,
        per_class_recall_dict,
        per_class_accuracy_dict,
    )


def load_split_csvs(splits_root: Path, seed: int):
    split_dir = splits_root / f"seed_{seed}"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    return split_dir, train_df, val_df, test_df


def load_style_metadata(dataset_csv: Path):
    df = pd.read_csv(dataset_csv)
    df = df[df["status"] == "success"].copy()
    df["style"] = df["style"].str.strip()
    all_styles = sorted(df["style"].unique())
    style_to_idx = {style: idx for idx, style in enumerate(all_styles)}
    return all_styles, style_to_idx, len(all_styles)


def debug_path_resolution(df: pd.DataFrame, base_dir: Path, sample_count: int) -> None:
    print("\n=== Path resolution sanity check ===")
    print("BASE_DIR:", base_dir)
    print("Top-level dataset/ exists:", (base_dir / "dataset").is_dir())
    print("data/raw dataset/ exists:", (base_dir / "data" / "raw dataset").is_dir())

    checked = 0
    found = 0
    for _, row in df.head(sample_count).iterrows():
        raw_path = row["image_path"]
        resolved_path = resolve_image_path(raw_path, base_dir)
        exists = os.path.isfile(resolved_path)
        found += int(exists)
        checked += 1
        print(f"  exists={exists} raw={raw_path}")
        print(f"         resolved={resolved_path}")

    print(f"Resolved files found: {found}/{checked}")
    if found != checked:
        raise RuntimeError("Some pilot image paths did not resolve correctly. Fix path mapping before training.")


def save_learning_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    val_macro_f1s,
    clip_lrs,
    head_lrs,
    best_epoch,
    output_path: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_losses, label="Train Loss", color="blue", linewidth=2)
    axes[0, 0].plot(val_losses, label="Val Loss", color="red", linewidth=2)
    axes[0, 0].axvline(x=best_epoch - 1, color="green", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label="Train Acc", color="blue", linewidth=2)
    axes[0, 1].plot(val_accs, label="Val Acc", color="red", linewidth=2)
    axes[0, 1].axvline(x=best_epoch - 1, color="green", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(val_macro_f1s, label="Val Macro F1", color="green", linewidth=2)
    axes[1, 0].axvline(x=best_epoch - 1, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("Validation Macro F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(head_lrs, label="Head LR", color="purple", linewidth=2)
    if clip_lrs:
        axes[1, 1].plot(clip_lrs, label="CLIP LR", color="orange", linewidth=2)
    axes[1, 1].set_title("Learning Rates")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Image-only CLIP fine-tuning pilot: seed {seed}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Project root:", PROJECT_ROOT)

    split_dir, train_df, val_df, test_df = load_split_csvs(args.splits_root, args.seed)
    all_styles, style_to_idx, num_classes = load_style_metadata(args.dataset_csv)

    print(f"Loaded seed {args.seed} CSVs from {split_dir}")
    print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Classes: {num_classes}")

    debug_path_resolution(train_df, args.base_dir, args.debug_samples)

    output_root = args.output_root
    metrics_dir = output_root / "metrics"
    artifacts_dir = output_root / "artifacts"
    models_dir = artifacts_dir / "models"
    curves_dir = artifacts_dir / "learning_curves"
    experiments_dir = metrics_dir / "experiments"
    for directory in [metrics_dir, artifacts_dir, models_dir, curves_dir, experiments_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(args.model_init_seed)
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model = clip_model.float()

    model = ImageOnlyFashionClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        dropout=args.dropout,
        visual_dim=int(getattr(clip_model.visual, "output_dim", 512)),
    ).to(device)

    trainable_info = configure_partial_visual_finetuning(model, args.unfreeze_visual_blocks)
    print("\n=== Trainable parameter summary ===")
    print("Unfrozen visual blocks:", trainable_info["unfrozen_visual_blocks"])
    print("Trainable parameters:", trainable_info["trainable_parameter_count"])
    print("Total parameters:", trainable_info["total_parameter_count"])
    for name in trainable_info["trainable_parameter_names"][:20]:
        print("  ", name)
    if len(trainable_info["trainable_parameter_names"]) > 20:
        print("  ...")

    train_dataset = FashionImageOnlyDataset(train_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)
    val_dataset = FashionImageOnlyDataset(val_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)
    test_dataset = FashionImageOnlyDataset(test_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.setup_only:
        print("\nSetup-only mode complete. No training was run.")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        return

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(style_to_idx.values())),
        y=train_df["style"].map(style_to_idx).values,
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.clip_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_macro_f1s = []
    clip_lrs = []
    head_lrs = []
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    model_path = models_dir / f"seed_{args.seed}_best_model.pth"
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"STARTING IMAGE-ONLY CLIP FINE-TUNE PILOT (seed {args.seed})")
    print(f"{'=' * 70}")

    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, val_macro_f1 = validate_epoch(model, val_loader, criterion, device)

        current_clip_lr = 0.0
        current_head_lr = 0.0
        for group in optimizer.param_groups:
            if group.get("name") == "clip":
                current_clip_lr = float(group["lr"])
            else:
                current_head_lr = float(group["lr"])
        clip_lrs.append(current_clip_lr)
        head_lrs.append(current_head_lr)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_macro_f1s.append(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch + 1}/{args.max_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} clip_lr={current_clip_lr:.2e} head_lr={current_head_lr:.2e}"
        )

        scheduler.step()

        if patience_counter >= args.patience:
            early_stopped = True
            print(f"Early stopping at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time
    model.load_state_dict(torch.load(model_path, map_location=device))

    (
        val_loss,
        val_acc,
        _,
        _,
        val_macro_f1,
        val_macro_precision,
        val_macro_recall,
        val_per_class_f1,
        val_per_class_precision,
        val_per_class_recall,
        val_per_class_accuracy,
    ) = evaluate_with_per_class_metrics(model, val_loader, criterion, device, all_styles)

    (
        test_loss,
        test_acc,
        _,
        _,
        test_macro_f1,
        test_macro_precision,
        test_macro_recall,
        test_per_class_f1,
        test_per_class_precision,
        test_per_class_recall,
        test_per_class_accuracy,
    ) = evaluate_with_per_class_metrics(model, test_loader, criterion, device, all_styles)

    overfitting_detected = False
    if len(val_losses) > best_epoch:
        overfitting_detected = min(val_losses[best_epoch:]) > best_val_loss * 1.05

    train_val_gap = train_losses[best_epoch - 1] - best_val_loss if best_epoch > 0 else 0.0
    curve_path = curves_dir / f"seed_{args.seed}_learning_curves.png"
    save_learning_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        val_macro_f1s,
        clip_lrs,
        head_lrs,
        best_epoch,
        curve_path,
        args.seed,
    )

    results = {
        "experiment_id": f"seed_{args.seed}",
        "seed_value": args.seed,
        "seed_index": 1,
        "timestamp": datetime.now().isoformat(),
        "model_type": "image_only_clip_finetuned_partial",
        "configuration": {
            "clip_model": args.clip_model,
            "clip_lr": args.clip_lr,
            "head_lr": args.head_lr,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "model_init_seed": args.model_init_seed,
            "data_split_seed": args.seed,
            "split_source": "csv",
            "unfreeze_visual_blocks": args.unfreeze_visual_blocks,
        },
        "training_info": {
            "total_epochs": len(train_losses),
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "total_time_minutes": float(total_time / 60.0),
            "trainable_parameter_count": trainable_info["trainable_parameter_count"],
        },
        "validation_metrics": {
            "best_val_macro_f1": float(val_macro_f1),
            "best_val_accuracy": float(val_acc),
            "best_val_macro_precision": float(val_macro_precision),
            "best_val_macro_recall": float(val_macro_recall),
            "best_val_loss": float(best_val_loss),
            "per_class_metrics": {
                "f1": val_per_class_f1,
                "precision": val_per_class_precision,
                "recall": val_per_class_recall,
                "accuracy": val_per_class_accuracy,
            },
        },
        "test_metrics": {
            "test_macro_f1": float(test_macro_f1),
            "test_accuracy": float(test_acc),
            "test_macro_precision": float(test_macro_precision),
            "test_macro_recall": float(test_macro_recall),
            "test_loss": float(test_loss),
            "per_class_metrics": {
                "f1": test_per_class_f1,
                "precision": test_per_class_precision,
                "recall": test_per_class_recall,
                "accuracy": test_per_class_accuracy,
            },
        },
        "overfitting_analysis": {
            "overfitting_detected": overfitting_detected,
            "best_val_loss": float(best_val_loss),
            "train_val_gap": float(train_val_gap),
        },
        "training_curves": {
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "train_accs": [float(x) for x in train_accs],
            "val_accs": [float(x) for x in val_accs],
            "val_macro_f1s": [float(x) for x in val_macro_f1s],
            "clip_learning_rates": [float(x) for x in clip_lrs],
            "head_learning_rates": [float(x) for x in head_lrs],
        },
        "data_split_info": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        },
        "trainable_parameters": trainable_info,
    }

    result_json = experiments_dir / f"seed_{args.seed}_results.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total_seeds": 1,
        "successful_experiments": 1,
        "failed_experiments": 0,
        "completed_seeds": [args.seed],
        "experiment_root": str(output_root),
    }
    with open(metrics_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(metrics_dir / "trainable_parameters.json", "w", encoding="utf-8") as f:
        json.dump(trainable_info, f, indent=2)

    print(f"\nSaved result JSON to {result_json}")
    print(f"Saved model checkpoint to {model_path}")
    print(f"Saved learning curves to {curve_path}")
    print(f"Best val macro F1: {best_val_macro_f1:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
