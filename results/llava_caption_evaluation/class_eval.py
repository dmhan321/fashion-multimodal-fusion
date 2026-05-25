#!/usr/bin/env python3
"""
Aggregate caption-evaluation metrics by fashion style (14 classes).

Reads per-sample / per-seed outputs from the main caption evaluation runs
(sections 1–5 in llava_caption_evaluation.ipynb) and writes summaries under
./class_eval/

Run standalone (no notebook required):
    python class_eval.py
    python class_eval.py --results-root /path/to/caption_evaluation
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_ROOT / "class_eval"
SPLIT_TAG = "test"  # main notebook metrics are on the test split only

PER_SAMPLE_SPECS: List[Tuple[str, Dict[str, str]]] = [
    ("clip_similarity", {"clip_similarity_mean": "clip_similarity"}),
    ("clip_iqa", {"clip_iqa_win_rate": "clip_iqa_win", "clip_iqa_margin_mean": "clip_iqa_margin"}),
    ("blipscore", {"blip_itm_mean": "blip_itm_score"}),
    ("gpt4o_evaluation", {"gpt4o_semantic_mean": "semantic", "gpt4o_style_mean": "style_fit"}),
]

SEED_LEVEL_SPECS: List[Tuple[str, List[str]]] = [
    ("retrieval_recall", [f"i2t_recall@{k}" for k in (1, 5, 10)] + [f"t2i_recall@{k}" for k in (1, 5, 10)]),
    ("confusion_matrix", ["accuracy", "macro_f1"]),
    ("random_caption_sanity", ["acc_correct_caption", "acc_random_caption", "accuracy_drop"]),
    ("class_randomized_sanity", ["acc_correct_caption", "acc_wrong_style_caption", "accuracy_drop"]),
]


def load_seeds(seeds_file: Path) -> List[int]:
    if not seeds_file.is_file():
        return []
    content = seeds_file.read_text(encoding="utf-8")
    matches = re.findall(r"Seed\s+(\d+)", content, flags=re.IGNORECASE)
    return sorted({int(s) for s in matches if 1 <= int(s) <= 500})


def style_from_merge_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("/").str[1]


def aggregate_per_style_per_seed(
    results_root: Path, seeds: List[int]
) -> pd.DataFrame:
    rows: List[dict] = []

    for metric_group, col_map in PER_SAMPLE_SPECS:
        metric_dir = results_root / metric_group
        for seed in seeds:
            seed_csv = metric_dir / f"seed_{seed}_per_sample.csv"
            if not seed_csv.is_file():
                continue
            df = pd.read_csv(seed_csv)
            if "style" not in df.columns and "merge_key" in df.columns:
                df["style"] = style_from_merge_key(df["merge_key"])
            if "style" not in df.columns:
                continue
            for out_metric, src_col in col_map.items():
                if src_col not in df.columns:
                    continue
                for style, grp in df.groupby("style"):
                    rows.append(
                        {
                            "seed": seed,
                            "split": SPLIT_TAG,
                            "metric_group": metric_group,
                            "metric": out_metric,
                            "style": style,
                            "value": float(grp[src_col].mean()),
                            "n_samples": int(len(grp)),
                        }
                    )

    conf_dir = results_root / "confusion_matrix"
    for seed in seeds:
        pred_csv = conf_dir / f"seed_{seed}_predictions.csv"
        if not pred_csv.is_file():
            continue
        df = pd.read_csv(pred_csv)
        if "style" not in df.columns or "pred_style" not in df.columns:
            continue
        for style, grp in df.groupby("style"):
            acc = float((grp["style"] == grp["pred_style"]).mean())
            rows.append(
                {
                    "seed": seed,
                    "split": SPLIT_TAG,
                    "metric_group": "confusion_matrix",
                    "metric": "per_style_accuracy",
                    "style": style,
                    "value": acc,
                    "n_samples": int(len(grp)),
                }
            )

    for metric_group, value_cols in SEED_LEVEL_SPECS:
        summary_path = results_root / metric_group / "all_seeds_summary.csv"
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            seed = int(row["seed"])
            for col in value_cols:
                if col not in row.index:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "split": SPLIT_TAG,
                        "metric_group": metric_group,
                        "metric": col,
                        "style": "__all__",
                        "value": float(row[col]),
                        "n_samples": int(row.get("n_test", row.get("n", np.nan)))
                        if pd.notna(row.get("n_test", row.get("n", np.nan)))
                        else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def mean_std_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return per_seed_df
    agg = (
        per_seed_df.groupby(["split", "metric_group", "metric", "style"], as_index=False)["value"]
        .agg(mean="mean", std=lambda x: float(x.std(ddof=0)), n_seeds="count")
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption evaluation by fashion style class.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Folder containing clip_similarity/, blipscore/, etc.",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="seeds_list.txt (default: FashionStyle14_v1 next to robustness/)",
    )
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    fusionstyle = results_root.parent.parent
    seeds_file = args.seeds_file or (fusionstyle / "FashionStyle14_v1" / "seeds_list.txt")
    seeds = load_seeds(seeds_file)
    if not seeds:
        found = sorted(
            int(p.stem.split("_")[1])
            for p in (results_root / "clip_similarity").glob("seed_*_per_sample.csv")
        )
        seeds = found
        print(f"No seeds_list.txt; inferred {len(seeds)} seeds from per-sample files.")

    out_dir = results_root / "class_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = aggregate_per_style_per_seed(results_root, seeds)
    if per_seed.empty:
        raise SystemExit(
            f"No metric outputs found under {results_root}. "
            "Run llava_caption_evaluation.ipynb sections 1–5 first."
        )

    per_seed_path = out_dir / "by_style_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary = mean_std_table(per_seed)
    summary_path = out_dir / "by_style_aggregation_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    pivot = summary.pivot_table(
        index=["style", "metric_group", "metric"],
        columns="split",
        values="mean",
        aggfunc="first",
    )
    pivot_path = out_dir / "by_style_mean_pivot.csv"
    pivot.reset_index().to_csv(pivot_path, index=False)

    print(f"Wrote {per_seed_path} ({len(per_seed)} rows)")
    print(f"Wrote {summary_path} ({len(summary)} rows)")
    print(f"Wrote {pivot_path}")
    print("\nSample (first 10 style-level aggregates):")
    print(summary[summary["style"] != "__all__"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
