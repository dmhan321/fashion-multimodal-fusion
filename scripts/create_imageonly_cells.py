#!/usr/bin/env python3
"""
Script to generate remaining cells for ImageOnly notebook
Adapts from TextOnly notebook structure
"""
import json

# Read text-only notebook
with open('TextOnly_Robustness_Experiments.ipynb', 'r') as f:
    text_only = json.load(f)

# Read image-only notebook (partially created)
with open('ImageOnly_Robustness_Experiments.ipynb', 'r') as f:
    image_only = json.load(f)

# We need to adapt cells 11-26 from text-only to image-only
# Key changes:
# - run_textonly_experiment -> run_imageonly_experiment
# - FashionTextOnlyDataset -> FashionImageOnlyDataset  
# - TextOnlyFashionClassifier -> ImageOnlyFashionClassifier
# - captions -> images
# - fashionbert_model, fashionbert_tokenizer -> clip_model
# - model_type: "text_only" -> "image_only"
# - "Text-Only Model" -> "Image-Only Model"
# - Add transforms for images
# - Remove captions_dict parameter

# For now, let's just print what needs to be adapted
print("Need to adapt cells 11-26 from text-only to image-only")
print(f"Text-only has {len(text_only['cells'])} cells")
print(f"Image-only currently has {len(image_only['cells'])} cells")





