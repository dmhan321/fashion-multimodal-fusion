# Fine-Tuned Image-Only Follow-Up

Use this note after the 30-seed fine-tuned image-only run has completed under:

`experiments/imageonly_clip_finetuned_robustness/`

## Existing paired-statistics notebooks

These notebooks currently compare:
- `phase3_robustness`
- `imageonly_robustness`

Files:
- `notebooks/evaluation/statistics/paired_ttest_fusion_vs_imageonly.ipynb`
- `notebooks/evaluation/statistics/paired_bootstrap_fusion_vs_imageonly.ipynb`
- `notebooks/evaluation/statistics/wilcoxon_signed_rank_fusion_vs_imageonly.ipynb`

## Required update once results exist

Change the hardcoded image-only experiment name from:

`imageonly_robustness`

to:

`imageonly_clip_finetuned_robustness`

Only do this after the new experiment has produced:
- `metrics/experiments/seed_<seed>_results.json` for the full 30 seeds
- `metrics/experiments_summary.json`
- `metrics/summary_table.csv`

## Seed-matching check

Before updating the paired notebooks, confirm the fine-tuned run has the same seed set as:

`experiments/phase3_robustness/metrics/experiments/seed_*_results.json`

The paired tests assume matched seeds and compare:

`test_metrics.test_macro_f1`

## Reporting

If you also want fine-tuned image-only included in project summaries, update any downstream report scripts or notebooks that still hardcode:

`imageonly_robustness`

to either:
- replace it with `imageonly_clip_finetuned_robustness`, or
- add the fine-tuned variant as an additional experiment alongside the frozen baseline.
