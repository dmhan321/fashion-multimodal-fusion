#!/usr/bin/env python3
"""
30-seed robustness runner for partial CLIP fine-tuning on the image-only fashion task.

This runner uses the same CSV-backed split protocol as the local fusion workflows:

    data/splits/seed_<seed>/{train,val,test}.csv

Outputs are written under:

    experiments/imageonly_clip_finetuned_robustness/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.imageonly_clip_finetune_one_seed import (  # noqa: E402
    FashionImageOnlyDataset,
    ImageOnlyFashionClassifier,
    build_optimizer,
    configure_partial_visual_finetuning,
    debug_path_resolution,
    evaluate_with_per_class_metrics,
    load_split_csvs,
    load_style_metadata,
    save_learning_curves,
    set_seed,
    train_epoch,
    validate_epoch,
)


DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "imageonly_clip_finetuned_robustness"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "caption_dataset_final_full.csv"
DEFAULT_SEEDS_FILE = PROJECT_ROOT / "experiments" / "phase3_robustness" / "metrics" / "seeds_list.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="30-seed robustness runner for partial CLIP fine-tuning.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--clip-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-visual-blocks", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument("--setup-only", action="store_true", help="Sanity-check the first seed then exit")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--only-seeds",
        type=str,
        default="",
        help="Comma-separated subset of seeds to run instead of the full seeds file",
    )
    return parser.parse_args()


def load_phase3_seeds(seeds_file: Path) -> list[int]:
    text = seeds_file.read_text(encoding="utf-8")
    seeds = [int(match) for match in re.findall(r"Seed\s+(\d+)", text)]
    if not seeds:
        raise RuntimeError(f"No seeds found in {seeds_file}")
    return seeds


def calculate_stats(values: list[float], name: str) -> dict[str, float | int | str]:
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    median_val = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    cv = float((std_val / mean_val * 100.0) if mean_val != 0 else 0.0)

    if len(arr) > 1:
        ci = stats.t.interval(0.95, len(arr) - 1, loc=mean_val, scale=stats.sem(arr))
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    else:
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "metric": name,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "q25": q25,
        "q75": q75,
        "cv_percent": cv,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n": int(len(arr)),
    }


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    metrics_dir = output_root / "metrics"
    artifacts_dir = output_root / "artifacts"
    paths = {
        "root": output_root,
        "metrics": metrics_dir,
        "artifacts": artifacts_dir,
        "experiments": metrics_dir / "experiments",
        "models": artifacts_dir / "models",
        "learning_curves": artifacts_dir / "learning_curves",
    }
    for path in paths.values():
        if path != output_root:
            path.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    return paths


def run_single_seed(
    seed_value: int,
    seed_idx: int,
    total_seeds: int,
    args: argparse.Namespace,
    device: torch.device,
    all_styles: list[str],
    style_to_idx: dict[str, int],
    num_classes: int,
    output_paths: dict[str, Path],
):
    result_file = output_paths["experiments"] / f"seed_{seed_value}_results.json"
    if result_file.exists():
        print(f"Seed {seed_value}: result already exists, loading and skipping.")
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)

    split_dir, train_df, val_df, test_df = load_split_csvs(args.splits_root, seed_value)

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

    print(f"\n{'=' * 70}")
    print(f"Experiment {seed_idx}/{total_seeds}: Seed {seed_value}")
    print(f"{'=' * 70}")
    print(f"Loaded CSV splits from {split_dir}")
    print(
        f"Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)} | "
        f"trainable params: {trainable_info['trainable_parameter_count']}"
    )

    train_dataset = FashionImageOnlyDataset(train_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)
    val_dataset = FashionImageOnlyDataset(val_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)
    test_dataset = FashionImageOnlyDataset(test_df, style_to_idx, clip_preprocess, base_dir=args.base_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
    model_path = output_paths["models"] / f"seed_{seed_value}_best_model.pth"
    start_time = time.time()

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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{args.max_epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Val Macro F1={val_macro_f1:.4f}, CLIP LR={current_clip_lr:.2e}, Head LR={current_head_lr:.2e}"
            )

        scheduler.step()

        if patience_counter >= args.patience:
            early_stopped = True
            print(f"  Early stopping at epoch {epoch + 1}")
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
    curve_path = output_paths["learning_curves"] / f"seed_{seed_value}_learning_curves.png"
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
        seed_value,
    )

    results = {
        "experiment_id": f"seed_{seed_value}",
        "seed_value": seed_value,
        "seed_index": seed_idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
            "data_split_seed": seed_value,
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

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"  Completed: Best Val Macro F1={results['validation_metrics']['best_val_macro_f1']:.4f}, "
        f"Test Macro F1={results['test_metrics']['test_macro_f1']:.4f}"
    )
    return results


def write_seeds_list(metrics_dir: Path, seeds: list[int]) -> None:
    path = metrics_dir / "seeds_list.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Fine-tuned image-only robustness seeds\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total seeds: {len(seeds)}\n\n")
        for idx, seed in enumerate(seeds, 1):
            f.write(f"{idx:2d}. Seed {seed}\n")


def save_summary_outputs(all_results: list[dict], all_styles: list[str], metrics_dir: Path) -> None:
    if not all_results:
        return

    summary_rows = []
    for result in all_results:
        summary_rows.append(
            {
                "Seed": result["seed_value"],
                "Best_Epoch": result["training_info"]["best_epoch"],
                "Early_Stopped": result["training_info"]["early_stopped"],
                "Best_Val_Macro_F1": result["validation_metrics"]["best_val_macro_f1"],
                "Best_Val_Accuracy": result["validation_metrics"]["best_val_accuracy"],
                "Test_Macro_F1": result["test_metrics"]["test_macro_f1"],
                "Test_Accuracy": result["test_metrics"]["test_accuracy"],
                "Test_Macro_Precision": result["test_metrics"]["test_macro_precision"],
                "Test_Macro_Recall": result["test_metrics"]["test_macro_recall"],
                "Overfitting": result["overfitting_analysis"]["overfitting_detected"],
                "Training_Time_Min": result["training_info"]["total_time_minutes"],
            }
        )
    df_summary = pd.DataFrame(summary_rows).sort_values("Seed")
    df_summary.to_csv(metrics_dir / "summary_table.csv", index=False)

    overall_stats = [
        calculate_stats(df_summary["Test_Macro_F1"].tolist(), "Test Macro F1"),
        calculate_stats(df_summary["Test_Accuracy"].tolist(), "Test Accuracy"),
        calculate_stats(df_summary["Test_Macro_Precision"].tolist(), "Test Macro Precision"),
        calculate_stats(df_summary["Test_Macro_Recall"].tolist(), "Test Macro Recall"),
        calculate_stats(df_summary["Best_Val_Macro_F1"].tolist(), "Best Val Macro F1"),
        calculate_stats(df_summary["Best_Val_Accuracy"].tolist(), "Best Val Accuracy"),
        calculate_stats(df_summary["Training_Time_Min"].tolist(), "Training Time (min)"),
    ]
    with open(metrics_dir / "overall_metrics_statistics.json", "w", encoding="utf-8") as f:
        json.dump({"overall_metrics": overall_stats}, f, indent=2)
    pd.DataFrame(overall_stats).to_csv(metrics_dir / "overall_metrics_summary.csv", index=False)

    per_class_rows = []
    per_class_stats = {}
    for style in all_styles:
        test_f1s = []
        test_precisions = []
        test_recalls = []
        test_accuracies = []
        for result in all_results:
            test_pc = result["test_metrics"].get("per_class_metrics", {})
            if test_pc and style in test_pc.get("f1", {}):
                test_f1s.append(test_pc["f1"][style])
                test_precisions.append(test_pc["precision"][style])
                test_recalls.append(test_pc["recall"][style])
                test_accuracies.append(test_pc["accuracy"][style])
        if not test_f1s:
            continue

        style_stats = {
            "f1": calculate_stats(test_f1s, f"{style} - Test F1"),
            "precision": calculate_stats(test_precisions, f"{style} - Test Precision"),
            "recall": calculate_stats(test_recalls, f"{style} - Test Recall"),
            "accuracy": calculate_stats(test_accuracies, f"{style} - Test Accuracy"),
        }
        per_class_stats[style] = style_stats
        per_class_rows.append(
            {
                "Style": style,
                "Test_F1_Mean": style_stats["f1"]["mean"],
                "Test_F1_Std": style_stats["f1"]["std"],
                "Test_Precision_Mean": style_stats["precision"]["mean"],
                "Test_Precision_Std": style_stats["precision"]["std"],
                "Test_Recall_Mean": style_stats["recall"]["mean"],
                "Test_Recall_Std": style_stats["recall"]["std"],
                "Test_Accuracy_Mean": style_stats["accuracy"]["mean"],
                "Test_Accuracy_Std": style_stats["accuracy"]["std"],
            }
        )

    with open(metrics_dir / "per_class_metrics_statistics.json", "w", encoding="utf-8") as f:
        json.dump(per_class_stats, f, indent=2)
    pd.DataFrame(per_class_rows).to_csv(metrics_dir / "per_class_metrics_summary.csv", index=False)

    high_level = {
        "test_macro_f1": next(item for item in overall_stats if item["metric"] == "Test Macro F1"),
        "test_accuracy": next(item for item in overall_stats if item["metric"] == "Test Accuracy"),
        "n_seeds": len(all_results),
    }
    with open(metrics_dir / "statistical_analysis.json", "w", encoding="utf-8") as f:
        json.dump(high_level, f, indent=2)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.only_seeds.strip():
        seeds = [int(chunk.strip()) for chunk in args.only_seeds.split(",") if chunk.strip()]
    else:
        seeds = load_phase3_seeds(args.seeds_file)

    output_paths = ensure_output_dirs(args.output_root)
    write_seeds_list(output_paths["metrics"], seeds)
    all_styles, style_to_idx, num_classes = load_style_metadata(args.dataset_csv)

    if args.setup_only:
        _, train_df, _, _ = load_split_csvs(args.splits_root, seeds[0])
        debug_path_resolution(train_df, args.base_dir, args.debug_samples)
        print("Setup-only mode complete. No robustness training was run.")
        return

    print(f"Running {len(seeds)} seeds -> {args.output_root}")
    print(f"CLIP LR={args.clip_lr:.2e} | Head LR={args.head_lr:.2e} | Unfreeze blocks={args.unfreeze_visual_blocks}")

    all_results = []
    failed_seeds = []

    for seed_idx, seed_value in enumerate(seeds, 1):
        try:
            result = run_single_seed(
                seed_value=seed_value,
                seed_idx=seed_idx,
                total_seeds=len(seeds),
                args=args,
                device=device,
                all_styles=all_styles,
                style_to_idx=style_to_idx,
                num_classes=num_classes,
                output_paths=output_paths,
            )
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"Seed {seed_value} failed: {exc}")
            failed_seeds.append({"seed": seed_value, "error": str(exc)})

    summary = {
        "total_seeds": len(seeds),
        "successful_experiments": len(all_results),
        "failed_experiments": len(failed_seeds),
        "failed_seeds": failed_seeds,
        "completed_seeds": [result["seed_value"] for result in all_results],
        "experiment_root": str(args.output_root),
    }
    with open(output_paths["metrics"] / "experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_summary_outputs(all_results, all_styles, output_paths["metrics"])
    print(f"Completed {len(all_results)}/{len(seeds)} seeds.")
    print(f"Experiments summary written to {output_paths['metrics'] / 'experiments_summary.json'}")


if __name__ == "__main__":
    main()
