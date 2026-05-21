#!/usr/bin/env python3
"""
Build a Qwen-caption CSV with the same columns as LLaVA_caption_dataset_final_full.csv:

    image_path, style, caption, status

- Row order and (image_path, style) come from the LLaVA CSV (drop-in alignment).
- caption / status come from the Qwen run CSV (column `caption` = Qwen text).
- Rows with no Qwen row get caption="" and status="missing_qwen".

Usage:
    python3 scripts/export_qwen_captions_llava_schema.py
    python3 scripts/export_qwen_captions_llava_schema.py --out data/captions/custom.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--llava",
        type=Path,
        default=root / "data/processed/LLaVA_caption_dataset_final_full.csv",
        help="Master CSV (defines row order and image_path, style).",
    )
    p.add_argument(
        "--qwen",
        type=Path,
        default=root / "data/captions/qwen25vl_caption_full.csv",
        help="Qwen CSV with columns image_path, style, caption, status (uses `caption` and `status`).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "data/captions/qwen25vl_prompt_a_v2_llava_schema.csv",
        help="Output CSV path.",
    )
    args = p.parse_args()

    llava = pd.read_csv(args.llava)
    qwen = pd.read_csv(args.qwen)

    required_l = {"image_path", "style", "caption", "status"}
    if not required_l.issubset(set(llava.columns)):
        raise ValueError(f"LLaVA CSV missing columns; have {list(llava.columns)}")
    for col in ("image_path", "caption", "status"):
        if col not in qwen.columns:
            raise ValueError(f"Qwen CSV missing {col!r}; have {list(qwen.columns)}")

    master = llava[["image_path", "style"]].copy()
    q_idx = qwen.set_index("image_path", drop=False)
    # If duplicate image_path in Qwen (shouldn't), last wins
    if q_idx.index.duplicated().any():
        q_idx = q_idx[~q_idx.index.duplicated(keep="last")]

    master["caption"] = master["image_path"].map(q_idx["caption"])
    master["status"] = master["image_path"].map(q_idx["status"])

    missing = master["caption"].isna() | (master["caption"].astype(str).str.strip() == "")
    master.loc[missing, "caption"] = ""
    master.loc[missing, "status"] = "missing_qwen"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.out, index=False)

    n_miss = int(missing.sum())
    print(f"Wrote {args.out} ({len(master)} rows, {n_miss} missing_qwen, {len(master) - n_miss} with Qwen caption)")


if __name__ == "__main__":
    main()
