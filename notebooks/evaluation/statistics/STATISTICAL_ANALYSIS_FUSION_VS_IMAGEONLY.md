# Statistical analysis: cross-attention fusion vs image-only CLIP

This document summarizes outputs from the three notebooks in this folder for **one** 30-seed pairwise comparison: **test macro-F1** from `experiments/*/metrics/experiments/seed_*_results.json`, with **d = attention-based fusion − image-only** (phase3_robustness vs imageonly_robustness).

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

### Reported quantities

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| mean_diff | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| Wilcoxon statistic | 457.0 |
| *p*-value (one-sided) | ≈ 2.33 × 10⁻⁸ |
| reject *H*₀ (fusion > image-only) | True |

### Interpretation

The **Wilcoxon signed-rank** test is a **nonparametric paired** procedure: it uses the **ranks of the absolute differences**, with signs, and tests whether positive shifts dominate. It does **not** assume normality of *d*.

With *p* ≈ 2.3 × 10⁻⁸ and **alternative = "greater"**, there is very strong evidence that fusion is **stochastically larger** than image-only on paired test macro-F1 across seeds (beyond “mean only”). The **median** gain (~0.017 on a 0–1 F1 scale) and **27 wins vs 3 losses** align with that conclusion. **Paired Cohen’s *d*** ≈ 1.6 indicates a **large** standardized shift in the differences.

---

## 2. Notebook 2 — Paired *t*-test on differences

**Notebook:** `paired_ttest_fusion_vs_imageonly.ipynb`  
**Procedure:** `scipy.stats.ttest_1samp(d, popmean=0.0, alternative="greater")` — equivalent to a **one-sided paired *t*-test** that the **mean** of *d* is greater than zero.

### Reported quantities

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| mean_diff | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| *t* statistic | 8.71 |
| *p*-value (one-sided) | ≈ 6.79 × 10⁻¹⁰ |
| df | 29 |
| reject *H*₀ (fusion > image-only) | True |

### Interpretation

The **paired *t*-test** targets the **population mean** of the seed-wise improvements *d*. Under approximate **normality of the differences** (often reasonable for *n* = 30 if *d* is not extremely skewed), the *t*-statistic is calibrated; here **median** and **mean** are close, which supports that.

The **one-sided *p*** is on the order of **10⁻⁹**, so at α = 0.05 you **reject** *H*₀ that the mean improvement is ≤ 0. In words: the data support a **positive average** test macro-F1 gain for fusion over image-only on the same splits. **Paired Cohen’s *d*** again summarizes a **large** mean gain relative to cross-seed variability.

---

## 3. Notebook 3 — Paired bootstrap (mean of *d*)

**Notebook:** `paired_bootstrap_fusion_vs_imageonly.ipynb`  
**Procedure:** *B* = 10,000 paired bootstrap resamples of the 30 differences (resample seed indices with replacement each replicate); `rng_seed` = 42. For each replicate, mean *d* is computed. Percentiles give intervals; a simple one-sided bootstrap *p*-value counts bootstrap means ≤ 0.

### Reported quantities

| Quantity | Value |
|----------|--------|
| *n* | 30 |
| mean_diff (observed) | 0.01658 |
| stdev_diff | 0.01042 |
| paired_Cohen_d | 1.59 |
| wins / ties / losses | 27 / 0 / 3 |
| median_diff | 0.01716 |
| *B* / rng_seed | 10,000 / 42 |
| 95% CI for mean *d* (percentile method on bootstrap means) | [0.01282, 0.02023] |
| one-sided 95% lower bound (5th percentile of bootstrap means) | 0.01349 |
| bootstrap *p* (one-sided, mean ≤ 0) | ≈ 1.0 × 10⁻⁴ |
| reject *H*₀ (mean ≤ 0) | True |
| one-sided lower bound > 0 | True |

### Interpretation

The **bootstrap** gives uncertainty for the **mean** improvement **without** a strong parametric assumption on *d*. The **95% percentile interval** for the mean difference lies **entirely above zero**, so the mean gain is estimated roughly between **0.013** and **0.020** on the 0–1 F1 scale. The **5th percentile** of bootstrap means (**~0.0135**) is still **> 0**, which supports a **one-sided** statement that the mean improvement is positive at a conventional bootstrap level.

The printed **bootstrap *p*** uses a simple \((1 + \#\{\text{boot means} \le 0\})/(B+1)\) rule; it is **conservative** when almost no bootstrap means fall at or below zero (here *p* ≈ 10⁻⁴, still **< 0.05**). Re-running with larger *B* would refine the Monte Carlo tail; the **CI** and **lower bound** are the main stability story.

---

## 4. Overall analysis (three notebooks together)

### Convergence of conclusions

All three analyses operate on the **same 30 paired differences** *d* = fusion − image-only (test macro-F1). They answer **related but not identical** questions:

| Notebook | Target | Role |
|----------|--------|------|
| Wilcoxon | Signed-rank / dominance of positive *d* | **Nonparametric** paired evidence; robust to odd tails |
| *t*-test | **Mean**(*d*) > 0 | **Parametric** paired evidence on the average gain |
| Bootstrap | **Mean**(*d*) and its sampling uncertainty | **Distribution-free** framing for the mean and CI |

All three support the **same directional conclusion** at **α = 0.05**: fusion **improves** over image-only on this metric and experimental design. The **parametric *t*-test** and **nonparametric Wilcoxon** both yield **very small *p*-values**; the **bootstrap** places the **mean improvement** clearly **above zero** with a **95% interval** that excludes zero.

### Effect size and consistency

- **Mean** and **median** improvements are both **~0.017** on the 0–1 scale (**~1.7** macro-F1 points if reported as percentage points).
- **Paired Cohen’s *d*** ≈ **1.6** is **large** by common benchmarks, indicating the gain is large relative to seed-to-seed noise.
- **27 / 30** seeds favor fusion; **3** favor image-only; **0** ties — improvement is **consistent** but not universal (worth one sentence in a limitation: fusion does not win on every split).

### What you can claim to a reviewer

You can say you used **matched random seeds** and reported **three complementary checks**: (1) **Wilcoxon signed-rank** (paired, nonparametric, one-sided), (2) **paired *t*-test** on differences (parametric, one-sided mean), and (3) **paired bootstrap** for a **95% CI** and one-sided evidence on the **mean** difference. Together they justify that **cross-attention fusion improves test macro-F1 vs image-only CLIP** under your protocol, beyond a single lucky split.

### Caveats (short)

- Conclusions are about **this dataset, metric, and 30-seed protocol**, not all fashion domains.
- *t*-test assumes **roughly normal** *d*; Wilcoxon and bootstrap relax different parts of that story.
- Bootstrap *p* and percentiles include **Monte Carlo error**; *B* = 10,000 is adequate for reporting; increase *B* if a referee asks for tighter tail estimates.

---

## Reproducibility

Re-run the three notebooks after pulling the repo. Notebook 3 depends on **`rng_seed = 42`** and **`B = 10_000`** for repeatable bootstrap numbers.
