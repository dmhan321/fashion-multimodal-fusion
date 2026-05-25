#!/usr/bin/env python3
"""
Caption evaluation across train / validation / test splits, averaged over seeds.

Self-contained: loads FashionStyle14 + LLaVA captions, runs sample-level metrics
on each split for each seed in seeds_list.txt, then aggregates mean +/- std.

Run standalone:
    python splits_eval.py
    python splits_eval.py --max-samples-per-split 200
    python splits_eval.py --overwrite
"""
from __future__ import annotations

import argparse
import os
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BlipForImageTextRetrieval, BlipProcessor, CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
FUSIONSTYLE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "splits_eval"

VAL_RATIO = 0.15
TEST_RATIO = 0.15
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BLIP_ITM_MODEL_ID = "Salesforce/blip-itm-base-coco"
CLIP_IQA_NEGATIVES = 5
RETRIEVAL_K = [1, 5, 10]

SPLITS = ("train", "val", "test")


def load_seeds(seeds_file: Path) -> List[int]:
    content = seeds_file.read_text(encoding="utf-8")
    matches = re.findall(r"Seed\s+(\d+)", content, flags=re.IGNORECASE)
    seeds = sorted({int(s) for s in matches if 1 <= int(s) <= 500})
    if len(seeds) != 30:
        print(f"Warning: expected 30 seeds, found {len(seeds)}")
    return seeds


def normalize_rel_path(path_str: str) -> str:
    return str(path_str).strip().replace("\\", "/")


def canonical_merge_key(raw: str) -> str:
    s = normalize_rel_path(raw).lstrip("./")
    low = s.lower()
    if low.startswith("fashionstyle14_v1/"):
        s = s[len("fashionstyle14_v1/") :].lstrip("/")
        low = s.lower()
    marker = "dataset/"
    ix = low.find(marker)
    if ix >= 0:
        return normalize_rel_path(s[ix:])
    if low.startswith("/dataset/"):
        return normalize_rel_path(s.lstrip("/"))
    return s


def load_complete_dataset(csv_path: Path, image_root: Path) -> pd.DataFrame:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    rel = [ln.strip() for ln in lines if ln.strip()]
    df = pd.DataFrame({"rel_path": rel})
    df["rel_path"] = df["rel_path"].map(normalize_rel_path)
    df["merge_key"] = df["rel_path"].map(canonical_merge_key)
    df["style"] = df["merge_key"].str.split("/").str[1]
    df["abs_path"] = df["rel_path"].apply(lambda r: str(image_root / r.replace("/", os.sep)))
    return df[df["abs_path"].map(os.path.isfile)].reset_index(drop=True)


def load_captions_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str).str.lower() == "success"]
    path_col = next((c for c in work.columns if c.lower().strip() in {"image_path", "rel_path", "path"}), None)
    cap_col = next((c for c in work.columns if c.lower().strip() in {"caption", "text", "description"}), None)
    if path_col is None or cap_col is None:
        raise ValueError(f"Caption CSV columns invalid: {list(work.columns)}")
    out = work[[path_col, cap_col]].rename(columns={path_col: "raw_image_path", cap_col: "caption"})
    out["merge_key"] = out["raw_image_path"].map(canonical_merge_key)
    out["caption"] = out["caption"].fillna("").astype(str).str.strip()
    return out[out["caption"] != ""].drop_duplicates(subset=["merge_key"], keep="last").reset_index(drop=True)


def split_by_seed(df: pd.DataFrame, seed_value: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["style"],
        random_state=seed_value,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["style"],
        random_state=seed_value,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def subsample_df(frame: pd.DataFrame, max_n: Optional[int]) -> pd.DataFrame:
    if max_n is None or len(frame) <= max_n:
        return frame
    return frame.sample(n=max_n, random_state=42).reset_index(drop=True)


@torch.no_grad()
def load_pil_batch(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


@torch.no_grad()
def clip_cosine_similarity_batch(clip_model, clip_processor, paths, captions, device, batch_size):
    scores = []
    clip_model.eval()
    for i in range(0, len(paths), batch_size):
        bp, bc = paths[i : i + batch_size], captions[i : i + batch_size]
        images = load_pil_batch(bp)
        inputs = clip_processor(text=bc, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1).detach().cpu().numpy()
        scores.extend(sim.tolist())
    return np.array(scores, dtype=np.float32)


@torch.no_grad()
def blip_itm_scores_batch(blip_model, blip_processor, paths, captions, device, batch_size):
    scores = []
    blip_model.eval()
    for i in range(0, len(paths), batch_size):
        bp, bc = paths[i : i + batch_size], captions[i : i + batch_size]
        images = load_pil_batch(bp)
        inputs = blip_processor(images=images, text=bc, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = blip_model(**inputs, use_itm_head=True)
        logits = out.itm_score
        batch_scores = logits if logits.ndim == 1 else torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(batch_scores.detach().cpu().numpy().reshape(-1).tolist())
    return np.array(scores, dtype=np.float32)


def recall_at_k(sim_matrix: np.ndarray, k: int) -> float:
    n = sim_matrix.shape[0]
    hits = 0
    for i in range(n):
        order = np.argsort(-sim_matrix[i])
        if i in order[:k]:
            hits += 1
    return hits / max(n, 1)


@torch.no_grad()
def clip_encode_batch(clip_model, clip_processor, paths, captions, device, batch_size):
    img_feats, txt_feats = [], []
    for i in range(0, len(paths), batch_size):
        bp, bc = paths[i : i + batch_size], captions[i : i + batch_size]
        images = load_pil_batch(bp)
        inputs = clip_processor(text=bc, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip_model(**inputs)
        img_feats.append(out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True))
        txt_feats.append(out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True))
    return torch.cat(img_feats, dim=0), torch.cat(txt_feats, dim=0)


def try_load_existing_test(metric_dir: Path, seed: int) -> Optional[pd.DataFrame]:
    """Reuse section-4 test outputs when available."""
    p = metric_dir / f"seed_{seed}_per_sample.csv"
    if p.is_file():
        return pd.read_csv(p)
    return None


def run_clip_similarity(
    clip_model, clip_processor, split_df, device, batch_size, seed, split, overwrite, reuse_test_dir
) -> dict:
    out_csv = OUTPUT_DIR / "clip_similarity" / f"seed_{seed}_{split}_per_sample.csv"
    if split == "test" and not overwrite:
        existing = try_load_existing_test(reuse_test_dir, seed)
        if existing is not None and "clip_similarity" in existing.columns:
            existing.to_csv(out_csv, index=False)
            return {"clip_similarity_mean": float(existing["clip_similarity"].mean()), "n": len(existing)}
    if out_csv.is_file() and not overwrite:
        per = pd.read_csv(out_csv)
        return {"clip_similarity_mean": float(per["clip_similarity"].mean()), "n": len(per)}
    sims = clip_cosine_similarity_batch(
        clip_model,
        clip_processor,
        split_df["abs_path"].tolist(),
        split_df["caption"].tolist(),
        device,
        batch_size,
    )
    cols = [c for c in ("merge_key", "style", "rel_path") if c in split_df.columns]
    per = split_df[cols].copy()
    per["clip_similarity"] = sims
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_csv, index=False)
    return {"clip_similarity_mean": float(np.mean(sims)), "n": len(per)}


def run_clip_iqa(clip_model, clip_processor, split_df, device, batch_size, seed, split, overwrite) -> dict:
    out_csv = OUTPUT_DIR / "clip_iqa" / f"seed_{seed}_{split}_per_sample.csv"
    if out_csv.is_file() and not overwrite:
        per = pd.read_csv(out_csv)
        return {
            "clip_iqa_win_rate": float(per["clip_iqa_win"].mean()),
            "clip_iqa_margin_mean": float(per["clip_iqa_margin"].mean()),
            "n": len(per),
        }
    n = len(split_df)
    rng = np.random.default_rng(seed)
    paths, all_caps = split_df["abs_path"].tolist(), split_df["caption"].tolist()
    pos_sims = clip_cosine_similarity_batch(clip_model, clip_processor, paths, all_caps, device, batch_size)
    margins, wins = [], []
    for i in range(n):
        neg_idx = rng.choice([j for j in range(n) if j != i], size=min(CLIP_IQA_NEGATIVES, max(1, n - 1)), replace=False)
        neg_caps = [all_caps[j] for j in neg_idx]
        neg_sims = clip_cosine_similarity_batch(
            clip_model, clip_processor, [paths[i]] * len(neg_caps), neg_caps, device, batch_size
        )
        margins.append(float(pos_sims[i] - neg_sims.mean()))
        wins.append(1.0 if pos_sims[i] > neg_sims.max() else 0.0)
    per = split_df[["merge_key", "style"]].copy()
    per["clip_iqa_margin"] = margins
    per["clip_iqa_win"] = wins
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_csv, index=False)
    return {"clip_iqa_win_rate": float(np.mean(wins)), "clip_iqa_margin_mean": float(np.mean(margins)), "n": n}


def run_blip(blip_model, blip_processor, split_df, device, batch_size, seed, split, overwrite, reuse_test_dir) -> dict:
    out_csv = OUTPUT_DIR / "blipscore" / f"seed_{seed}_{split}_per_sample.csv"
    if split == "test" and not overwrite:
        existing = try_load_existing_test(reuse_test_dir, seed)
        if existing is not None and "blip_itm_score" in existing.columns:
            existing.to_csv(out_csv, index=False)
            return {"blip_itm_mean": float(existing["blip_itm_score"].mean()), "n": len(existing)}
    if out_csv.is_file() and not overwrite:
        per = pd.read_csv(out_csv)
        return {"blip_itm_mean": float(per["blip_itm_score"].mean()), "n": len(per)}
    sc = blip_itm_scores_batch(
        blip_model, blip_processor, split_df["abs_path"].tolist(), split_df["caption"].tolist(), device, batch_size
    )
    per = split_df[["merge_key", "style"]].copy()
    per["blip_itm_score"] = sc
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_csv, index=False)
    return {"blip_itm_mean": float(np.mean(sc)), "n": len(per)}


def run_retrieval(clip_model, clip_processor, split_df, device, batch_size, seed, split, overwrite) -> dict:
    out_json = OUTPUT_DIR / "retrieval_recall" / f"seed_{seed}_{split}.json"
    if out_json.is_file() and not overwrite:
        import json

        return json.loads(out_json.read_text(encoding="utf-8"))
    paths, caps = split_df["abs_path"].tolist(), split_df["caption"].tolist()
    img_e, txt_e = clip_encode_batch(clip_model, clip_processor, paths, caps, device, batch_size)
    sim = (img_e @ txt_e.T).cpu().numpy()
    row: Dict[str, float] = {"n": len(split_df)}
    for k in RETRIEVAL_K:
        row[f"i2t_recall@{k}"] = recall_at_k(sim, k)
        row[f"t2i_recall@{k}"] = recall_at_k(sim.T, k)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_json.write_text(json.dumps(row), encoding="utf-8")
    return row


def run_confusion(clip_model, clip_processor, split_df, style_to_idx, classes, device, batch_size, seed, split, overwrite) -> dict:
    out_csv = OUTPUT_DIR / "confusion_matrix" / f"seed_{seed}_{split}_predictions.csv"
    if out_csv.is_file() and not overwrite:
        per = pd.read_csv(out_csv)
        y_true = per["style"].map(style_to_idx).to_numpy()
        y_pred = per["pred_style"].map(style_to_idx).to_numpy()
    else:
        style_prompts = [f"a fashion outfit in the {s} style" for s in classes]
        text_inputs = clip_processor(text=style_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        preds = []
        paths, caps = split_df["abs_path"].tolist(), split_df["caption"].tolist()
        for i in range(0, len(paths), batch_size):
            bp, bc = paths[i : i + batch_size], caps[i : i + batch_size]
            images = load_pil_batch(bp)
            img_inputs = clip_processor(images=images, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            img_emb = clip_model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            cap_inputs = clip_processor(text=bc, return_tensors="pt", padding=True, truncation=True)
            cap_inputs = {k: v.to(device) for k, v in cap_inputs.items()}
            cap_emb = clip_model.get_text_features(**cap_inputs)
            cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
            fused = (img_emb + cap_emb) / (img_emb + cap_emb).norm(dim=-1, keepdim=True)
            preds.extend((fused @ text_emb.T).argmax(dim=-1).cpu().tolist())
        idx_to_style = {i: s for s, i in style_to_idx.items()}
        per = split_df[["merge_key", "style"]].copy()
        per["pred_style"] = [idx_to_style[i] for i in preds]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        per.to_csv(out_csv, index=False)
        y_true = per["style"].map(style_to_idx).to_numpy()
        y_pred = np.array(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": len(per),
    }


def aggregate_summaries(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, metric), grp in long_df.groupby(["split", "metric"]):
        vals = grp["value"].astype(float)
        rows.append(
            {
                "split": split,
                "metric": metric,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "n_seeds": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption eval across train/val/test splits.")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-seeds", type=int, default=None, help="Use first N seeds only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval (faster)")
    parser.add_argument("--skip-confusion", action="store_true", help="Skip zero-shot confusion")
    args = parser.parse_args()

    data_dir = FUSIONSTYLE_DIR / "FashionStyle14_v1"
    complete_csv = data_dir / "complete_dataset.csv"
    caption_csv = data_dir / "caption" / "fashion_captions_llava_success.csv"
    seeds_file = data_dir / "seeds_list.txt"
    image_root = data_dir
    reuse_test_dir = SCRIPT_DIR  # parent metric outputs from main notebook

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        args.batch_size = min(args.batch_size, 16)

    seeds = load_seeds(seeds_file)
    if args.num_seeds:
        seeds = seeds[: args.num_seeds]

    df_paths = load_complete_dataset(complete_csv, image_root)
    cap_df = load_captions_long(caption_csv)
    df_full = df_paths.merge(cap_df[["merge_key", "caption"]], on="merge_key", how="inner").reset_index(drop=True)
    classes = sorted(df_full["style"].unique().tolist())
    style_to_idx = {s: i for i, s in enumerate(classes)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | samples: {len(df_full)} | seeds: {len(seeds)}")

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    blip_model = BlipForImageTextRetrieval.from_pretrained(BLIP_ITM_MODEL_ID).to(device)
    blip_processor = BlipProcessor.from_pretrained(BLIP_ITM_MODEL_ID)

    long_rows: List[dict] = []
    for si, seed in enumerate(seeds, start=1):
        train_df, val_df, test_df = split_by_seed(df_full, seed)
        split_frames = {
            "train": subsample_df(train_df, args.max_samples_per_split),
            "val": subsample_df(val_df, args.max_samples_per_split),
            "test": subsample_df(test_df, args.max_samples_per_split),
        }
        for split, split_df in split_frames.items():
            print(f"[seed {seed} ({si}/{len(seeds)})] split={split} n={len(split_df)}")
            r = run_clip_similarity(
                clip_model, clip_processor, split_df, device, args.batch_size, seed, split, args.overwrite, reuse_test_dir
            )
            long_rows.append({"seed": seed, "split": split, "metric": "clip_similarity_mean", "value": r["clip_similarity_mean"]})
            r = run_clip_iqa(clip_model, clip_processor, split_df, device, args.batch_size, seed, split, args.overwrite)
            long_rows.append({"seed": seed, "split": split, "metric": "clip_iqa_win_rate", "value": r["clip_iqa_win_rate"]})
            long_rows.append({"seed": seed, "split": split, "metric": "clip_iqa_margin_mean", "value": r["clip_iqa_margin_mean"]})
            r = run_blip(blip_model, blip_processor, split_df, device, args.batch_size, seed, split, args.overwrite, reuse_test_dir)
            long_rows.append({"seed": seed, "split": split, "metric": "blip_itm_mean", "value": r["blip_itm_mean"]})
            if not args.skip_retrieval:
                r = run_retrieval(clip_model, clip_processor, split_df, device, args.batch_size, seed, split, args.overwrite)
                for k in RETRIEVAL_K:
                    long_rows.append({"seed": seed, "split": split, "metric": f"i2t_recall@{k}", "value": r[f"i2t_recall@{k}"]})
                    long_rows.append({"seed": seed, "split": split, "metric": f"t2i_recall@{k}", "value": r[f"t2i_recall@{k}"]})
            if not args.skip_confusion:
                r = run_confusion(
                    clip_model, clip_processor, split_df, style_to_idx, classes, device, args.batch_size, seed, split, args.overwrite
                )
                long_rows.append({"seed": seed, "split": split, "metric": "accuracy", "value": r["accuracy"]})
                long_rows.append({"seed": seed, "split": split, "metric": "macro_f1", "value": r["macro_f1"]})

    long_df = pd.DataFrame(long_rows)
    per_seed_path = OUTPUT_DIR / "all_seeds_by_split_summary.csv"
    long_df.to_csv(per_seed_path, index=False)
    agg = aggregate_summaries(long_df)
    agg_path = OUTPUT_DIR / "aggregation_mean_std_by_split.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\nWrote {per_seed_path}")
    print(f"Wrote {agg_path}\n")
    print(agg.sort_values(["metric", "split"]).to_string(index=False))


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
