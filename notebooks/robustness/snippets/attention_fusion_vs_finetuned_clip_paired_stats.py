# Paste as a new notebook cell at the end of AttentionFusion_FinetunedCLIP_Qwen_Robustness.ipynb
# Requires: pandas, numpy, scipy (paired t, Wilcoxon, bootstrap block)
#
# Compares attention fusion vs fine-tuned CLIP **per split seed** from saved metrics JSON,
# then runs paired one-sided tests (H1: fusion > CLIP): paired t, Wilcoxon signed-rank,
# classical 95% CI on mean paired difference, and bootstrap percentile CI + bootstrap one-sided p
# (same pattern as TextOnly_Qwen_StratifiedSplits_Robustness.ipynb).

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:

    def display(obj):
        print(obj)


# --- resolve PROJECT_ROOT (same pattern as the training cell) ---
_walk = os.path.abspath(os.getcwd())
for _ in range(10):
    if os.path.isdir(os.path.join(_walk, "experiments")) and os.path.isdir(os.path.join(_walk, "data")):
        PROJECT_ROOT = _walk
        break
    _walk = os.path.dirname(_walk)
else:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

try:
    EXPERIMENT_ROOT  # type: ignore[name-defined]
except NameError:
    EXPERIMENT_ROOT = os.path.join(
        PROJECT_ROOT, "experiments", "attention_fusion_finetuned_clip_qwen_v2"
    )

CLIP_METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "imageonly_clip_finetuned_robustness",
    "metrics",
    "experiments",
)
FUSION_METRICS_DIR = os.path.join(EXPERIMENT_ROOT, "metrics", "experiments")


def _load_seed_metrics(metrics_dir: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    d = Path(metrics_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Metrics dir not found: {metrics_dir}")
    for p in d.glob("seed_*_results.json"):
        m = re.match(r"seed_(\d+)_results\.json$", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        with open(p, "r", encoding="utf-8") as f:
            out[seed] = json.load(f)
    return out


def _extract_metrics(doc: dict, prefix: str) -> dict:
    vm = doc.get("validation_metrics") or {}
    tm = doc.get("test_metrics") or {}
    ti = doc.get("training_info") or {}
    best_epoch = vm.get("best_epoch")
    if best_epoch is None:
        best_epoch = ti.get("best_epoch")
    return {
        f"{prefix}_best_val_macro_f1": vm.get("best_val_macro_f1"),
        f"{prefix}_best_epoch": best_epoch,
        f"{prefix}_test_macro_f1": tm.get("test_macro_f1"),
        f"{prefix}_test_accuracy": tm.get("test_accuracy"),
    }


clip_by_seed = _load_seed_metrics(CLIP_METRICS_DIR)
fusion_by_seed = _load_seed_metrics(FUSION_METRICS_DIR)

common = sorted(set(clip_by_seed) & set(fusion_by_seed))
missing_clip = sorted(set(fusion_by_seed) - set(clip_by_seed))
missing_fusion = sorted(set(clip_by_seed) - set(fusion_by_seed))

if missing_clip:
    print(
        "Seeds with fusion JSON but no CLIP JSON:",
        missing_clip[:20],
        f"... ({len(missing_clip)} total)" if len(missing_clip) > 20 else "",
    )
if missing_fusion:
    print(
        "Seeds with CLIP JSON but no fusion JSON:",
        missing_fusion[:20],
        f"... ({len(missing_fusion)} total)" if len(missing_fusion) > 20 else "",
    )

rows = []
for s in common:
    rows.append({"seed": s, **_extract_metrics(clip_by_seed[s], "clip"), **_extract_metrics(fusion_by_seed[s], "fusion")})

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No overlapping seeds between CLIP and fusion metrics.")

df["delta_val_macro_f1"] = df["fusion_best_val_macro_f1"] - df["clip_best_val_macro_f1"]
df["delta_test_macro_f1"] = df["fusion_test_macro_f1"] - df["clip_test_macro_f1"]

cols = [
    "seed",
    "clip_best_val_macro_f1",
    "fusion_best_val_macro_f1",
    "delta_val_macro_f1",
    "clip_test_macro_f1",
    "fusion_test_macro_f1",
    "delta_test_macro_f1",
    "clip_best_epoch",
    "fusion_best_epoch",
    "clip_test_accuracy",
    "fusion_test_accuracy",
]
df = df[[c for c in cols if c in df.columns]].sort_values("seed").reset_index(drop=True)

print(f"Paired seeds: n={len(df)}")
print("CLIP metrics:", CLIP_METRICS_DIR)
print("Fusion metrics:", FUSION_METRICS_DIR)
display(df)

# --- summary table ---
summary = pd.DataFrame(
    [
        {
            "metric": "best_val_macro_f1",
            "clip_mean": df["clip_best_val_macro_f1"].mean(),
            "fusion_mean": df["fusion_best_val_macro_f1"].mean(),
            "mean_delta_fusion_minus_clip": df["delta_val_macro_f1"].mean(),
            "fusion_wins": int((df["delta_val_macro_f1"] > 0).sum()),
            "ties_abs_le_1e6": int((df["delta_val_macro_f1"].abs() <= 1e-6).sum()),
        },
        {
            "metric": "test_macro_f1",
            "clip_mean": df["clip_test_macro_f1"].mean(),
            "fusion_mean": df["fusion_test_macro_f1"].mean(),
            "mean_delta_fusion_minus_clip": df["delta_test_macro_f1"].mean(),
            "fusion_wins": int((df["delta_test_macro_f1"] > 0).sum()),
            "ties_abs_le_1e6": int((df["delta_test_macro_f1"].abs() <= 1e-6).sum()),
        },
    ]
)
print("\nSummary (paired by seed)")
display(summary)


# --- Paired tests (same pattern as TextOnly_Qwen_StratifiedSplits_Robustness): t, Wilcoxon, bootstrap ---
from scipy import stats

ALPHA = 0.05
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def bootstrap_paired_mean_diff(d: np.ndarray, n_boot: int = N_BOOT, rng: np.random.Generator = RNG):
    """Resample seed indices with replacement; mean of d on each replicate."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("Need at least 2 pairs for bootstrap.")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
    # One-sided p (fusion > CLIP): small if mass of bootstrap means is above 0
    p_one_sided = (1.0 + np.sum(boot_means <= 0.0)) / (n_boot + 1.0)
    return float(ci_lo), float(ci_hi), float(p_one_sided), boot_means


def paired_test_rows(metric_label: str, fusion: np.ndarray, clip: np.ndarray, alpha: float = ALPHA):
    """Paired t (greater), Wilcoxon on differences (greater), parametric 95% CI on mean(d), bootstrap CI + boot p."""
    fusion = np.asarray(fusion, dtype=float)
    clip = np.asarray(clip, dtype=float)
    d = fusion - clip
    n = len(d)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    cohen_dz = mean_d / std_d if std_d > 1e-12 else np.nan

    try:
        t_res = stats.ttest_rel(fusion, clip, alternative="greater")
        t_stat = float(t_res.statistic)
        t_p_one = float(t_res.pvalue)
    except TypeError:
        t_res = stats.ttest_rel(fusion, clip)
        t_stat = float(t_res.statistic)
        p_two = float(t_res.pvalue)
        t_p_one = p_two / 2.0 if mean_d > 0 else 1.0 - p_two / 2.0

    if np.allclose(d, 0.0, atol=1e-12):
        w_stat = np.nan
        w_p_one = 1.0
    else:
        try:
            w_res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox", method="approx")
        except (TypeError, ValueError):
            try:
                w_res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox", mode="auto")
            except (TypeError, ValueError):
                w_res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
        w_stat = float(w_res.statistic) if w_res.statistic is not None else np.nan
        w_p_one = float(w_res.pvalue)

    if n > 1:
        se = stats.sem(d)
        h = se * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
        ci_lo, ci_hi = mean_d - h, mean_d + h
    else:
        ci_lo = ci_hi = mean_d

    wins = int(np.sum(d > 0))
    ties = int(np.sum(np.isclose(d, 0.0, atol=1e-12)))
    losses = int(np.sum(d < 0))

    boot_ci_lo, boot_ci_hi, boot_p_one, _ = bootstrap_paired_mean_diff(d, n_boot=N_BOOT, rng=RNG)
    boot_ci_entirely_gt_0 = bool(boot_ci_lo > 0.0)

    return {
        "metric": metric_label,
        "n_pairs": n,
        "mean_diff_fusion_minus_clip": mean_d,
        "std_diff": std_d,
        "cohen_dz_paired": cohen_dz,
        "ci95_mean_diff_lo": ci_lo,
        "ci95_mean_diff_hi": ci_hi,
        "t_statistic": t_stat,
        "t_p_one_sided_fusion_gt_clip": t_p_one,
        "reject_t_H0_at_alpha": bool(t_p_one < alpha),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_one_sided_fusion_gt_clip": w_p_one,
        "reject_wilcoxon_H0_at_alpha": bool(w_p_one < alpha),
        "wins_fusion": wins,
        "ties": ties,
        "losses_fusion": losses,
        "boot_ci95_mean_diff_lo": boot_ci_lo,
        "boot_ci95_mean_diff_hi": boot_ci_hi,
        "boot_ci_entirely_gt_0": boot_ci_entirely_gt_0,
        "boot_p_one_sided_fusion_gt_clip": boot_p_one,
        "reject_boot_H0_at_alpha": bool(boot_p_one < alpha),
    }


df_paired_tests = pd.DataFrame(
    [
        paired_test_rows("test_macro_f1", df["fusion_test_macro_f1"].values, df["clip_test_macro_f1"].values),
        paired_test_rows(
            "best_val_macro_f1",
            df["fusion_best_val_macro_f1"].values,
            df["clip_best_val_macro_f1"].values,
        ),
    ]
)

print(
    f"Paired tests: one-sided fusion > CLIP | alpha = {ALPHA} | "
    f"bootstrap mean-delta: {N_BOOT:,} resamples, RNG seed=42"
)
print(
    "Parametric CI = classical t interval on mean paired difference; "
    "Boot CI = percentile interval on bootstrap mean(d); "
    "boot p = (1 + #{boot mean <= 0}) / (n_boot+1)."
)
print("Seeds are paired by split id; interpret independence cautiously.")
display(df_paired_tests)
