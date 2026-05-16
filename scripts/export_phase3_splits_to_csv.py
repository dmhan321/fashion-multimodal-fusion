#!/usr/bin/env python3
"""
Export train/val/test CSVs using the same protocol as Phase3_Robustness_Experiments.ipynb:

  - Load caption_dataset_final_full.csv, keep status == 'success', strip style.
  - df_full row order = CSV read order (no extra sort on df_full).
  - Stratified train_test_split with TRAIN_RATIO=0.7, VAL=0.15, TEST=0.15,
    same two-step split and random_state=seed per experiment.

Writes data/splits/seed_<seed>/{train,val,test}.csv (all columns from df_full).

Examples:
  python scripts/export_phase3_splits_to_csv.py
  python scripts/export_phase3_splits_to_csv.py --all-from-seeds-list
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

COLAB_FIVE_SEEDS = {13, 17, 53, 309, 347}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_seeds_from_phase3_list(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    seeds: list[int] = []
    for m in re.finditer(r"Seed\s+(\d+)\s*$", text, re.MULTILINE):
        seeds.append(int(m.group(1)))
    if len(seeds) < 30:
        raise RuntimeError(f"Expected ~30 seeds in {path}, found {len(seeds)}")
    return seeds


def load_df_full(root: Path) -> pd.DataFrame:
    csv_path = root / "data" / "processed" / "caption_dataset_final_full.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df_success = df[df["status"] == "success"].copy()
    df_success["style"] = df_success["style"].str.strip()
    return df_success.copy()


def split_one(df_full: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df_full,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df_full["style"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["style"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def main() -> int:
    root = project_root()
    default_seeds_file = root / "experiments" / "phase3_robustness" / "metrics" / "seeds_list.txt"

    p = argparse.ArgumentParser(description="Export Phase-3-style stratified splits to CSV per seed.")
    p.add_argument(
        "--seeds-file",
        type=Path,
        default=default_seeds_file,
        help="Path to seeds_list.txt (default: experiments/phase3_robustness/metrics/seeds_list.txt)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "splits",
        help="Root directory for seed_<id>/ folders",
    )
    p.add_argument(
        "--all-from-seeds-list",
        action="store_true",
        help="Export every seed in seeds_list.txt (30). Default excludes the five Colab concat seeds.",
    )
    p.add_argument(
        "--exclude-seeds",
        type=str,
        default="",
        help="Comma-separated extra seeds to skip (e.g. 13,17).",
    )
    p.add_argument(
        "--only-seeds",
        type=str,
        default="",
        help="If set, export only these comma-separated seeds (overrides --remaining-only / --all).",
    )
    args = p.parse_args()

    if args.only_seeds.strip():
        seeds = [int(x.strip()) for x in args.only_seeds.split(",") if x.strip()]
    elif args.all_from_seeds_list:
        seeds = load_seeds_from_phase3_list(args.seeds_file)
    else:
        # Default (and --remaining-only): 30 minus the five Colab concat seeds
        all_seeds = load_seeds_from_phase3_list(args.seeds_file)
        seeds = [s for s in all_seeds if s not in COLAB_FIVE_SEEDS]

    extra_skip = set()
    if args.exclude_seeds.strip():
        extra_skip = {int(x.strip()) for x in args.exclude_seeds.split(",") if x.strip()}
    seeds = [s for s in seeds if s not in extra_skip]

    df_full = load_df_full(root)
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        train_df, val_df, test_df = split_one(df_full, seed)
        d = out_root / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(d / "train.csv", index=False)
        val_df.to_csv(d / "val.csv", index=False)
        test_df.to_csv(d / "test.csv", index=False)
        print(
            f"seed_{seed}: train={len(train_df)} val={len(val_df)} test={len(test_df)} -> {d}"
        )

    print(f"Done. {len(seeds)} seed folder(s) under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
