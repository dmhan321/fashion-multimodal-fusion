#!/usr/bin/env python3
"""Recompute test macro F1 / precision / recall / accuracy from saved fusion checkpoints (no training).

Updates each ``experiments/attention_fusion_ablation_frozen_clip_llava/<config>/metrics/experiments/seed_*.json``
``test_metrics`` block and rewrites ``ablation_summary_by_setting.csv``.

Usage (from repo root, GPU recommended):
    python3 scripts/backfill_attention_ablation_test_metrics.py
    python3 scripts/backfill_attention_ablation_test_metrics.py --force   # overwrite existing test_macro_precision
"""
from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

ABLATION_CONFIGS = [
    {"name": "proj256_h4", "label": "Smaller projection", "hidden_dim": 256, "num_heads": 4},
    {"name": "default_512_h8", "label": "Default", "hidden_dim": 512, "num_heads": 8},
    {"name": "proj768_h8", "label": "Larger projection", "hidden_dim": 768, "num_heads": 8},
    {"name": "heads4_512", "label": "Fewer heads", "hidden_dim": 512, "num_heads": 4},
    {"name": "head1_512", "label": "Single-head", "hidden_dim": 512, "num_heads": 1},
]
ABLATION_SEEDS = [13, 14, 16, 17, 45]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 32
DROPOUT = 0.5
MODEL_INIT_SEED = 42


def resolve_split_image_path(row_image_path: str, base_dir: str) -> str:
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
            has_caption = raw_key in captions_dict or nk in captions_dict or image_path in captions_dict
            if has_file and has_caption:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        raw_key = str(row["image_path"])
        image_path = resolve_split_image_path(raw_key, self.base_dir or ".")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.captions_dict.get(
            raw_key,
            self.captions_dict.get(os.path.normpath(raw_key), self.captions_dict.get(image_path, "")),
        )
        label = self.style_to_idx[row["style"]]
        return {"image": image, "caption": caption, "label": label}


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
        tokenizer,
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
        self._tokenizer = tokenizer
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
            inputs = self._tokenizer(
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


def validate_epoch(model, loader, criterion, dev):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_pred, all_lab = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test_eval", leave=False):
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
    return total_loss / max(len(loader), 1), acc, macro_f1, macro_p, macro_r


def make_seed_worker(base_seed):
    def seed_worker(worker_id):
        ws = base_seed + worker_id
        np.random.seed(ws)
        random.seed(ws)
        torch.manual_seed(ws)

    return seed_worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--force", action="store_true", help="Recompute even if test_macro_precision exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    project_root = args.root.resolve()
    os.chdir(project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PROJECT_ROOT:", project_root)
    print("device:", device)

    caption_csv = project_root / "data" / "processed" / "LLaVA_caption_dataset_final_full.csv"
    df = pd.read_csv(caption_csv)
    if "status" in df.columns:
        df_success = df[df["status"] == "success"].copy()
    else:
        df_success = df.copy()
    df_full = df_success.reset_index(drop=True)
    all_styles = sorted(df_full["style"].dropna().astype(str).unique())
    style_to_idx = {s: i for i, s in enumerate(all_styles)}
    num_classes = len(all_styles)

    captions_dict: dict[str, str] = {}
    for _, row in df_success.iterrows():
        raw = str(row["image_path"])
        cap = str(row["caption"])
        keys = {raw, os.path.normpath(raw)}
        if not os.path.isabs(raw):
            keys.add(os.path.normpath(os.path.join(str(project_root), raw)))
        keys.add(resolve_split_image_path(raw, str(project_root)))
        for k in keys:
            captions_dict[k] = cap

    import clip

    print("Loading CLIP + BERT...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    exp_root = project_root / "experiments" / "attention_fusion_ablation_frozen_clip_llava"
    base_dir = str(project_root)

    for cfg in ABLATION_CONFIGS:
        hd, nh = int(cfg["hidden_dim"]), int(cfg["num_heads"])
        for seed_value in ABLATION_SEEDS:
            name = cfg["name"]
            result_path = exp_root / name / "metrics" / "experiments" / f"seed_{seed_value}_results.json"
            ckpt_path = exp_root / name / "artifacts" / "models" / f"seed_{seed_value}_best_fusion_head.pth"
            if not result_path.is_file():
                print(f"SKIP (no json): {result_path}")
                continue
            if not ckpt_path.is_file():
                print(f"SKIP (no ckpt): {ckpt_path}")
                continue

            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
            tm = data.get("test_metrics") or {}
            if tm.get("test_macro_precision") is not None and not args.force:
                print(f"SKIP (has test_macro_precision): {result_path.name} in {name}")
                continue

            torch.manual_seed(MODEL_INIT_SEED)
            np.random.seed(MODEL_INIT_SEED)
            if torch.cuda.is_available():
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

            transform = clip_preprocess
            train_ds = FashionMultiModalDataset(train_df, captions_dict, style_to_idx, transform, base_dir=base_dir)
            test_ds = FashionMultiModalDataset(test_df, captions_dict, style_to_idx, transform, base_dir=base_dir)
            train_valid_df = train_ds.df.iloc[train_ds.valid_indices]
            class_weights = compute_class_weight(
                "balanced",
                classes=np.arange(num_classes),
                y=train_valid_df["style"].map(style_to_idx).values,
            )
            class_weights_t = torch.FloatTensor(class_weights).to(device)

            loader_seed = MODEL_INIT_SEED + int(seed_value)
            g_test = torch.Generator()
            g_test.manual_seed(loader_seed + 2)
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                worker_init_fn=make_seed_worker(loader_seed + 2),
                generator=g_test,
                pin_memory=torch.cuda.is_available(),
            )

            model = MultiModalFashionClassifier(
                clip_model,
                bert,
                tokenizer,
                num_classes=num_classes,
                dropout=DROPOUT,
                fusion_hidden_dim=hd,
                fusion_num_heads=nh,
            ).to(device)
            ck = torch.load(ckpt_path, map_location=device)
            model.fusion.load_state_dict(ck["fusion"])
            model.classifier.load_state_dict(ck["classifier"])
            model.eval()

            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
            te_loss, te_acc, te_f1, te_p, te_r = validate_epoch(model, test_loader, criterion, device)

            old_f1 = tm.get("test_macro_f1")
            if old_f1 is not None and abs(float(old_f1) - float(te_f1)) > 0.02:
                print(
                    f"  WARN {name} seed {seed_value}: test_macro_f1 drift old={old_f1:.4f} new={te_f1:.4f} "
                    "(splits/preprocess should match training notebook)"
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
                "source": "scripts/backfill_attention_ablation_test_metrics.py",
                "timestamp": datetime.now().isoformat(),
                "device": str(device),
            }

            if not args.dry_run:
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"OK {name} seed={seed_value} test_f1={te_f1:.4f} P={te_p:.4f} R={te_r:.4f} acc={te_acc:.4f}")
            else:
                print(f"DRY {name} seed={seed_value} test_f1={te_f1:.4f} P={te_p:.4f} R={te_r:.4f} acc={te_acc:.4f}")

    # Rebuild full summary from all JSON on disk (includes skipped rows with existing keys)
    all_rows: list[dict] = []
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        hd, nh = cfg["hidden_dim"], cfg["num_heads"]
        for seed_value in ABLATION_SEEDS:
            result_path = exp_root / name / "metrics" / "experiments" / f"seed_{seed_value}_results.json"
            if not result_path.is_file():
                continue
            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
            tm = data.get("test_metrics") or {}
            vm = data.get("validation_metrics") or {}
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

    ablation_df = pd.DataFrame(all_rows)
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
    out_csv = exp_root / "ablation_summary_by_setting.csv"
    if not args.dry_run:
        summary.to_csv(out_csv, index=False)
        print("\nWrote", out_csv)
    print("\n=== Summary (test metrics mean ± std) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
