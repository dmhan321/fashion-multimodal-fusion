# Fashion multimodal fusion

Multi-modal fashion style classification on FashionStyle14: caption generation, hyperparameter search, robustness across all experimented models, and evaluation.

## Repository layout

| Path | Purpose |
|------|---------|
| `notebooks/preprocessing/` | Dataset EDA, stratified splits, class weights |
| `notebooks/captioning/` | LLaVA and Qwen caption generation |
| `notebooks/training/` | Overfitting fixes, hyperparameter tuning, full-dataset confirmation, ResNet baseline |
| `notebooks/robustness/` | Multi-seed robustness runs for fusion, image-only, text-only, and multimodal baselines |
| `notebooks/evaluation/` | Caption quality, per-class metrics, explainability, error analysis, statistical tests |
| `notebooks/robustness/snippets/` | Shared Python snippets for robustness notebooks |
| `caption_data/` | Versioned caption CSV snapshots (`LLaVA_caption_dataset.csv`, `Qwen_caption_dataset.csv`) |
| `experiments/<name>/metrics/` | Lightweight JSON/CSV results (tracked in git) |
| `results/` | Evaluation outputs (caption eval, explainability, image-only summaries) |

## Research overview

This research implements the experiments for **Multimodal StyleFusion: Cross-Attention Learning for Fashion Style Recognition**. The work addresses abstract fashion style recognition on the FashionStyle14 dataset by combining CLIP visual embeddings, LLM-generated captions encoded with vanilla BERT, and a cross-attention fusion module.

### Problem statement

Fashion style recognition differs from garment category classification. Categories such as girlish, street, or conservative reflect abstract aesthetic concepts defined by combinations of visual attributes, textures, color composition, and semantic context rather than isolated object cues. Models trained on visual features alone often struggle to capture these higher-level stylistic semantics and may confuse overlapping style categories. Existing fashion recognition systems therefore remain limited when style perception requires both visual evidence and semantic interpretation.

### Goal

The goal of this project is to evaluate whether LLM-generated textual descriptions provide complementary information for abstract fashion style recognition when fused with strong visual representations. Specifically, we aim to:

- Build a reproducible multimodal pipeline for 14-class fashion style classification on FashionStyle14.
- Compare single-modality baselines (text-only, image-only) against multimodal fusion under controlled training conditions.
- Isolate the contribution of the fusion mechanism by comparing concatenation, gated fusion, and cross-attention fusion under a shared encoder backbone.
- Determine when textual cues help most, particularly under frozen versus fine-tuned visual encoders.

### Experimental strategies

Experiments follow a staged pipeline mirrored in `notebooks/training/` and `notebooks/robustness/`:

1. **Data preparation.** FashionStyle14 images are resized to 224 x 224, normalized with ImageNet statistics, and split 70/15/15 with stratification. Class weights address minor imbalance. LLaVA (primary) and Qwen2.5-VL (extended analysis) generate fashion-oriented captions for each image.
2. **Model selection and hyperparameter tuning.** Regularization settings (early stopping, dropout, weight decay) are searched on a 50% subset, followed by learning-rate and batch-size tuning. The final configuration is confirmed on the full dataset before robustness evaluation.
3. **Baseline comparison.** We compare a reproduced ResNet-50 baseline, text-only BERT, and image-only models (CLIP, ViT, Swin, ConvNeXt), plus optional fine-tuned CLIP variants.
4. **Multimodal fusion comparison.** Three fusion strategies are evaluated with encoders, splits, and optimization held fixed: concatenation with MLP, gated fusion, and cross-attention fusion (StyleFusion).
5. **Robustness evaluation.** Main classification experiments are repeated across 30 random stratified split seeds with fixed model initialization. An attention-fusion architecture ablation (projection dimension and number of heads) is run on five matched seeds under frozen encoders.
6. **Caption and error analysis.** Caption-image alignment is assessed independently of classification. Additional notebooks examine duplicates, per-class behavior, explainability, and statistical significance.

### Evaluation

**Classification.** Style prediction is evaluated with macro precision, macro recall, macro F1, and accuracy. Macro averaging treats all 14 style classes equally and is the primary metric for model comparison.

**Caption quality.** Generated captions are evaluated with CLIP similarity, CLIP-IQA, and BLIP image-text matching scores to assess semantic alignment between images and descriptions.

**Statistical testing.** Paired comparisons use the same random seeds across models. We report one-sided paired t-tests, Wilcoxon signed-rank tests, and paired bootstrap tests (10,000 resamples) where appropriate, defining paired differences as fusion performance minus the corresponding image-only baseline.

**Explainability and diagnostics.** Grad-CAM, attention rollout, error analysis, and SHA-256 duplicate screening support interpretation and data-quality checks.

### Results

On FashionStyle14 with LLaVA captions and frozen CLIP ViT-B/32:

| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| Reproduced ResNet-50 | — | 0.74 |
| Text-only (BERT + LLaVA captions) | 0.28 | 0.32 |
| Image-only ViT | 0.59 | 0.59 |
| Image-only ConvNeXt | 0.77 | 0.77 |
| Image-only Swin | 0.79 | 0.78 |
| Image-only frozen CLIP | 0.82 | 0.82 |
| Concatenation fusion | 0.81 | — |
| Gated fusion | 0.81 | — |
| **Cross-attention fusion (StyleFusion)** | **0.83** | **0.83** |

Key findings:

- Text-only models perform poorly, confirming that captions alone are insufficient for fine-grained style discrimination.
- Frozen CLIP provides the strongest single-modality baseline, outperforming conventional vision backbones.
- Cross-attention fusion achieves the best overall performance among frozen-encoder models, outperforming simpler fusion strategies that do not explicitly model cross-modal interactions.
- Paired statistical tests show significant gains over ViT, ConvNeXt, and Swin, but not over frozen CLIP image-only (p = 0.899, two-sided paired t-test), indicating that multimodal benefit is modest when the visual encoder is already semantically aligned.
- Under fine-tuned CLIP with Qwen captions, image-only and all fusion variants converge near 0.85 macro F1, suggesting that fusion adds limited value once the visual encoder is adapted to the target task.
- Caption sanity checks and architecture ablation further show that correct image-caption pairing matters and that projection dimension has a stronger effect than attention-head count in the two-token fusion design.

### Contributions

This work makes the following contributions to multimodal fashion style recognition:

1. **StyleFusion framework.** A multimodal pipeline that integrates CLIP visual embeddings with semantic representations from LLM-generated fashion captions (LLaVA and Qwen), encoded with vanilla BERT and fused through a cross-attention module for 14-class style classification.
2. **Controlled fusion comparison.** A systematic evaluation of concatenation, gated fusion, and cross-attention fusion under a shared encoder backbone, data splits, and training recipe, isolating the effect of the fusion mechanism.
3. **Empirical analysis across modalities and encoders.** Evidence that cross-attention fusion improves over conventional image-only backbones and simpler fusion methods under frozen CLIP (macro F1 0.83), together with analysis of when textual cues add limited benefit after visual encoder fine-tuning.
4. **Reproducible benchmark.** Multi-seed robustness evaluation, caption-quality assessment, statistical testing, and ablation studies released as notebooks and experiment outputs in this repository.

### Authors

Department of Applied Data Science, San José State University, San Jose, CA, United States

**Researchers**

| Name | Email |
|------|-------|
| Dongmei Han | dongmei.han@sjsu.edu |
| Shao-Yu Huang | shaoyu.huang@sjsu.edu |
| Jiayi Liang | jiayi.liang@sjsu.edu |

**Faculty advisor**

Mohammad Masum — mohammad.masum@sjsu.edu