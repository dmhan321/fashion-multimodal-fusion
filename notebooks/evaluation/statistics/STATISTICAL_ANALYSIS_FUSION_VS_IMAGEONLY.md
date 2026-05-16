# Statistical analysis: cross-attention fusion vs image-only CLIP

This document summarizes outputs from the three notebooks in this folder for **one** 30-seed pairwise comparison: **test macro-F1** and **test accuracy** from `experiments/*/metrics/experiments/seed_*_results.json`, with **d = attention-based fusion − image-only** (phase3_robustness vs imageonly_robustness). Accuracy is converted from percentage to 0–1 when loading from JSON.

**Hypothesis (one-sided):** fusion **>** image-only on the mean or distribution of paired differences. **α = 0.05.**

## Protocol clarification (seed counts across model families)

| Model / fusion | Matched seeds used in robustness sweeps |
|----------------|----------------------------------------|
| Text-only, image-only (frozen CLIP), **attention-based fusion** (phase3) | **30** (same seed list) |
| **Gated fusion** and **concatenation / MLP fusion** | **5** only |

So the **Wilcoxon / paired *t* / bootstrap notebooks here** apply only to the **30-seed** image-only vs attention-fusion comparison. **They do not** automatically apply to gated or concat fusion, because those variants were not evaluated on all 30 seeds. For the paper: either **report fusion-strategy comparisons on the 5 shared seeds** (with smaller-*n* statistics and wider uncertainty) or **extend gated and concat to all 30 seeds** if you want the same paired-test machinery everywhere.

---

## 1. Notebook 1 — Wilcoxon signed-rank

**Notebook:** `wilcoxon_signed_rank_fusion_vs_imageonly.ipynb`  
**Procedure:** `scipy.stats.wilcoxon(d, alternative="greater", zero_method="wilcox")` on paired differences *d*.

### Test macro-F1

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8311 |
| image-only mean | 0.8145 |
| mean_diff | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| Wilcoxon statistic | 457.0 |
| *p*-value (one-sided) | ≈ 2.33 × 10⁻⁸ |
| reject *H*₀ (fusion > image-only) | True |

### Test accuracy (0–1 scale)

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8312 |
| image-only mean | 0.8153 |
| mean_diff | 0.01593 |
| stdev_diff | 0.01002 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01711 |
| Wilcoxon statistic | 458.0 |
| *p*-value (one-sided) | ≈ 1.77 × 10⁻⁸ |
| reject *H*₀ (fusion > image-only) | True |

### Interpretation

The **Wilcoxon signed-rank** test is a **nonparametric paired** procedure: it uses the **ranks of the absolute differences**, with signs, and tests whether positive shifts dominate. It does **not** assume normality of *d*.

For both **macro-F1** and **accuracy**, *p* ≈ **10⁻⁸** with **alternative = "greater"**, so fusion is **stochastically larger** than image-only on paired test performance across seeds. **Median** gains (~0.017 on the 0–1 scale) and **27 wins vs 3 losses** are the same for both metrics. **Paired Cohen’s *d*** ≈ 1.6 indicates a **large** standardized shift.

---

## 2. Notebook 2 — Paired *t*-test on differences

**Notebook:** `paired_ttest_fusion_vs_imageonly.ipynb`  
**Procedure:** `scipy.stats.ttest_1samp(d, popmean=0.0, alternative="greater")` — equivalent to a **one-sided paired *t*-test** that the **mean** of *d* is greater than zero.

### Test macro-F1

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8311 |
| image-only mean | 0.8145 |
| mean_diff | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| *t* statistic | 8.71 |
| *p*-value (one-sided) | ≈ 6.79 × 10⁻¹⁰ |
| df | 29 |
| reject *H*₀ (fusion > image-only) | True |

### Test accuracy (0–1 scale)

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8312 |
| image-only mean | 0.8153 |
| mean_diff | 0.01593 |
| stdev_diff | 0.01002 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01668 |
| *t* statistic | 8.71 |
| *p*-value (one-sided) | ≈ 6.90 × 10⁻¹⁰ |
| df | 29 |
| reject *H*₀ (fusion > image-only) | True |

### Interpretation

The **paired *t*-test** targets the **population mean** of the seed-wise improvements *d*. For both **macro-F1** and **accuracy**, one-sided *p* ≈ **10⁻¹⁰** at α = 0.05, so you **reject** *H*₀ that the mean improvement is ≤ 0. The data support a **positive average** gain for fusion over image-only on the same splits for **both metrics** (~+1.6–1.7 on the 0–1 scale). **Paired Cohen’s *d*** ≈ 1.6 for both; win/loss pattern is identical (27 wins, 3 losses).

---

## 3. Notebook 3 — Paired bootstrap (mean of *d*)

**Notebook:** `paired_bootstrap_fusion_vs_imageonly.ipynb`  
**Procedure:** *B* = 10,000 paired bootstrap resamples of the 30 differences (resample seed indices with replacement each replicate); `rng_seed` = 42. For each replicate, mean *d* is computed. Percentiles give intervals; a simple one-sided bootstrap *p*-value counts bootstrap means ≤ 0.

### Test macro-F1

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8311 |
| image-only mean | 0.8145 |
| mean_diff (observed) | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| *B* / rng_seed | 10,000 / 42 |
| 95% CI for mean *d* | [0.01282, 0.02023] |
| one-sided 95% lower bound (5th percentile) | 0.01349 |
| bootstrap *p* (one-sided, mean ≤ 0) | ≈ 1.0 × 10⁻⁴ |
| reject *H*₀ (mean ≤ 0) | True |
| one-sided lower bound > 0 | True |

### Test accuracy (0–1 scale)

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| fusion mean | 0.8312 |
| image-only mean | 0.8153 |
| mean_diff (observed) | 0.01593 |
| stdev_diff | 0.01002 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01711 |
| *B* / rng_seed | 10,000 / 42 |
| 95% CI for mean *d* | [0.01228, 0.01940] |
| one-sided 95% lower bound (5th percentile) | 0.01293 |
| bootstrap *p* (one-sided, mean ≤ 0) | ≈ 1.0 × 10⁻⁴ |
| reject *H*₀ (mean ≤ 0) | True |
| one-sided lower bound > 0 | True |

### Interpretation

The **bootstrap** gives uncertainty for the **mean** improvement **without** a strong parametric assumption on *d*. For **both metrics**, the **95% percentile interval** for the mean difference lies **entirely above zero** (macro-F1 roughly **0.013–0.020**; accuracy roughly **0.012–0.019** on the 0–1 scale). The **5th percentile** of bootstrap means is still **> 0** for both (~**0.0135** for F1, ~**0.0129** for accuracy).

The printed **bootstrap *p*** uses \((1 + \#\{\text{boot means} \le 0\})/(B+1)\); here *p* ≈ **10⁻⁴** for both metrics (still **< 0.05**). Re-running with larger *B* would refine the Monte Carlo tail; the **CI** and **lower bound** are the main stability story.

---

## 4. Overall analysis (three notebooks together)

### Convergence of conclusions

All three analyses operate on the **same 30 paired differences** *d* = fusion − image-only, computed separately for **test macro-F1** and **test accuracy**. They answer **related but not identical** questions:

| Notebook | Target | Role |
|----------|--------|------|
| Wilcoxon | Signed-rank / dominance of positive *d* | **Nonparametric** paired evidence; robust to odd tails |
| *t*-test | **Mean**(*d*) > 0 | **Parametric** paired evidence on the average gain |
| Bootstrap | **Mean**(*d*) and its sampling uncertainty | **Distribution-free** framing for the mean and CI |

All three support the **same directional conclusion** at **α = 0.05** for **both macro-F1 and accuracy**: fusion **improves** over image-only under this experimental design. The **parametric *t*-test** and **nonparametric Wilcoxon** both yield **very small *p*-values** on both metrics; the **bootstrap** places the **mean improvement** clearly **above zero** with **95% intervals** that exclude zero for both.

### Effect size and consistency

- **Mean** improvements: **~0.0166** (macro-F1) and **~0.0159** (accuracy) on the 0–1 scale (~**1.6–1.7** percentage points).
- **Median** improvements: **~0.017** for both metrics.
- **Paired Cohen’s *d*** ≈ **1.6** is **large** by common benchmarks, indicating the gain is large relative to seed-to-seed noise.
- **27 / 30** seeds favor fusion on **both** metrics; **3** favor image-only; **0** ties — improvement is **consistent** but not universal (worth one sentence in a limitation: fusion does not win on every split).

### What you can claim to a reviewer

You can say you used **matched random seeds** and reported **three complementary checks** on **test macro-F1 and test accuracy**: (1) **Wilcoxon signed-rank** (paired, nonparametric, one-sided), (2) **paired *t*-test** on differences (parametric, one-sided mean), and (3) **paired bootstrap** for a **95% CI** and one-sided evidence on the **mean** difference. Together they justify that **cross-attention fusion improves both metrics vs image-only CLIP** under your protocol, beyond a single lucky split.

### Caveats (short)

- Conclusions are about **this dataset, metric, and 30-seed protocol**, not all fashion domains.
- *t*-test assumes **roughly normal** *d*; Wilcoxon and bootstrap relax different parts of that story.
- Bootstrap *p* and percentiles include **Monte Carlo error**; *B* = 10,000 is adequate for reporting; increase *B* if a referee asks for tighter tail estimates.

---

## Reproducibility

Re-run the three notebooks after pulling the repo. Notebook 3 depends on **`rng_seed = 42`** and **`B = 10_000`** for repeatable bootstrap numbers.
