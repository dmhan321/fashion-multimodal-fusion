# ============================================
# MISSING PARTS FOR ImageOnly_Robustness_Experiments.ipynb
# Copy these cells into the notebook after Cell 11
# ============================================

## 6. Experiment Runner Function

def run_imageonly_experiment(seed_value, seed_idx, df_full, style_to_idx,
                              clip_model, num_classes, device, all_styles, base_dir):
    """
    Run a single image-only robustness experiment with given seed
    
    Args:
        seed_value: Random seed value for data splitting (from 1-500)
        seed_idx: Index of seed in SEEDS list (1-30)
        df_full: Full dataset DataFrame
        style_to_idx: Dictionary mapping style names to indices
        clip_model: Pre-trained CLIP model
        num_classes: Number of classes
        device: Device
        all_styles: List of style names
        base_dir: Base directory for absolute paths
    
    Returns:
        Dictionary with all results and metrics (including per-class metrics)
    """
    
    print(f"\n{'='*70}")
    print(f"Experiment {seed_idx}/{len(SEEDS)}: Seed {seed_value}")
    print(f"{'='*70}")
    
    # Check if result already exists (resume capability)
    result_file = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
    if os.path.exists(result_file):
        print(f"  ⏭️  Result already exists, skipping...")
        with open(result_file, 'r') as f:
            return json.load(f)
    
    # Set fixed seed for model initialization (same for all experiments)
    torch.manual_seed(MODEL_INIT_SEED)
    np.random.seed(MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_INIT_SEED)
    
    # Create stratified train/val/test splits with this seed
    print(f"  Creating data splits with random_state={seed_value}...")
    train_df, temp_df = train_test_split(
        df_full,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df_full['style'],
        random_state=seed_value  # Different seed for each experiment
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df['style'],
        random_state=seed_value  # Same seed
    )
    
    print(f"  Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Image transforms - use CLIP's preprocessing (clip_preprocess from Cell 6)
    # Note: clip_preprocess is loaded in Cell 6, so it should be available in the notebook
    # Using CLIP's own preprocessing ensures proper normalization
    transform = clip_preprocess
    
    # Create datasets
    train_dataset = FashionImageOnlyDataset(train_df, style_to_idx, transform, base_dir=base_dir)
    val_dataset = FashionImageOnlyDataset(val_df, style_to_idx, transform, base_dir=base_dir)
    test_dataset = FashionImageOnlyDataset(test_df, style_to_idx, transform, base_dir=base_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(list(style_to_idx.values())),
        y=train_df['style'].map(style_to_idx).values
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Initialize model (with fixed seed)
    model = ImageOnlyFashionClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        dropout=DROPOUT,
        visual_dim=512
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_macro_f1s = []
    learning_rates = []
    
    best_val_macro_f1 = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    start_time = time.time()
    
    # Training loop with early stopping
    for epoch in range(MAX_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_macro_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_macro_f1s.append(val_macro_f1)
        
        # Track best Macro F1 (for saving & early stopping)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 
                      os.path.join(ARTIFACTS_DIR, "models", f"seed_{seed_value}_best_model.pth"))
        else:
            patience_counter += 1
        
        # Track best loss (for overfitting detection)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Early stopping (based on Macro F1)
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{MAX_EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Macro F1={val_macro_f1:.4f}")
    
    total_time = time.time() - start_time
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(
        os.path.join(ARTIFACTS_DIR, "models", f"seed_{seed_value}_best_model.pth")))
    model.eval()
    
    # Re-evaluate validation set at best epoch with per-class metrics
    print(f"  Computing per-class metrics on validation set (best epoch)...")
    (val_loss, val_acc, val_preds, val_labels,
     val_macro_f1, val_macro_precision, val_macro_recall,
     val_per_class_f1, val_per_class_precision, val_per_class_recall, val_per_class_accuracy) = evaluate_with_per_class_metrics(
        model, val_loader, criterion, device, all_styles
    )
    
    # Final evaluation on test set with per-class metrics
    print(f"  Computing per-class metrics on test set...")
    (test_loss, test_acc, test_preds, test_labels,
     test_macro_f1, test_macro_precision, test_macro_recall,
     test_per_class_f1, test_per_class_precision, test_per_class_recall, test_per_class_accuracy) = evaluate_with_per_class_metrics(
        model, test_loader, criterion, device, all_styles
    )
    
    # Detect overfitting
    if len(val_losses) > best_epoch:
        val_loss_after_best = min(val_losses[best_epoch:])
        overfitting_detected = val_loss_after_best > best_val_loss * 1.05
    else:
        overfitting_detected = False
    
    # Calculate train/val gap at best epoch
    train_val_gap = train_losses[best_epoch - 1] - best_val_loss if best_epoch > 0 else 0
    
    # Create learning curves plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 0].axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(train_accs, label='Train Acc', color='blue', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Acc', color='red', linewidth=2)
    axes[0, 1].axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Macro F1 curve
    axes[1, 0].plot(val_macro_f1s, label='Val Macro F1', color='green', linewidth=2)
    axes[1, 0].axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=best_val_macro_f1, color='red', linestyle='--', alpha=0.7, 
                       label=f'Best: {best_val_macro_f1:.4f}')
    axes[1, 0].set_title('Validation Macro F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
Seed: {seed_value} (Experiment {seed_idx}/{len(SEEDS)})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Epoch: {best_epoch}
Early Stopped: {early_stopped}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Val Macro F1: {best_val_macro_f1:.4f}
Test Macro F1: {test_macro_f1:.4f}
Test Accuracy: {test_acc:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overfitting: {'Yes' if overfitting_detected else 'No'}
Training Time: {total_time/60:.1f} minutes
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Learning Curves: Seed {seed_value} (Image-Only Model)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ARTIFACTS_DIR, "learning_curves", f"seed_{seed_value}_learning_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare results dictionary (with per-class metrics included)
    results = {
        "experiment_id": f"seed_{seed_value}",
        "seed_value": seed_value,
        "seed_index": seed_idx,
        "timestamp": datetime.now().isoformat(),
        "model_type": "image_only",
        "configuration": {
            "learning_rate": float(LEARNING_RATE),
            "batch_size": BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "dropout": DROPOUT,
            "weight_decay": float(WEIGHT_DECAY),
            "max_epochs": MAX_EPOCHS,
            "model_init_seed": MODEL_INIT_SEED,
            "data_split_seed": seed_value,
            "dataset_size": "100% (full dataset)"
        },
        "training_info": {
            "total_epochs": len(train_losses),
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "total_time_minutes": float(total_time / 60)
        },
        "validation_metrics": {
            "best_val_macro_f1": float(val_macro_f1),
            "best_val_accuracy": float(val_acc),
            "best_val_macro_precision": float(val_macro_precision),
            "best_val_macro_recall": float(val_macro_recall),
            "best_val_loss": float(best_val_loss),
            "final_val_macro_f1": float(val_macro_f1s[-1]),
            "final_val_accuracy": float(val_accs[-1]),
            "final_val_loss": float(val_losses[-1]),
            "per_class_metrics": {
                "f1": val_per_class_f1,
                "precision": val_per_class_precision,
                "recall": val_per_class_recall,
                "accuracy": val_per_class_accuracy
            }
        },
        "test_metrics": {
            "test_macro_f1": float(test_macro_f1),
            "test_accuracy": float(test_acc),
            "test_macro_precision": float(test_macro_precision),
            "test_macro_recall": float(test_macro_recall),
            "test_loss": float(test_loss),
            "per_class_metrics": {
                "f1": test_per_class_f1,
                "precision": test_per_class_precision,
                "recall": test_per_class_recall,
                "accuracy": test_per_class_accuracy
            }
        },
        "overfitting_analysis": {
            "overfitting_detected": overfitting_detected,
            "best_val_loss": float(best_val_loss),
            "val_loss_after_best": float(val_losses[best_epoch:][0]) if len(val_losses) > best_epoch else float(val_losses[-1]),
            "increase_percentage": float((val_losses[best_epoch:][0] - best_val_loss) / best_val_loss * 100) if len(val_losses) > best_epoch else 0.0,
            "train_val_gap": float(train_val_gap)
        },
        "training_curves": {
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "train_accs": [float(x) for x in train_accs],
            "val_accs": [float(x) for x in val_accs],
            "val_macro_f1s": [float(x) for x in val_macro_f1s],
            "learning_rates": [float(x) for x in learning_rates]
        },
        "data_split_info": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df)
        }
    }
    
    # Save results JSON
    json_path = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✅ Completed: Best Val Macro F1={best_val_macro_f1:.4f}, "
          f"Test Macro F1={test_macro_f1:.4f}, "
          f"Overfitting={'Yes' if overfitting_detected else 'No'}")
    
    return results

print("✅ Experiment runner function defined!")


## 7. Run All Experiments

# Get base directory for absolute paths
BASE_DIR = os.path.abspath('.')

# Run all image-only robustness experiments
all_results = []
failed_seeds = []

print(f"\n{'='*70}")
print(f"STARTING IMAGE-ONLY ROBUSTNESS EXPERIMENTS")
print(f"Total experiments: {len(SEEDS)}")
print(f"Dataset: Full (100%) - {len(df_full)} images")
print(f"Estimated time: ~{len(SEEDS) * 2.0:.1f} hours")
print(f"{'='*70}")

for seed_idx, seed_value in enumerate(SEEDS, 1):
    try:
        result = run_imageonly_experiment(
            seed_value=seed_value,
            seed_idx=seed_idx,
            df_full=df_full,
            style_to_idx=style_to_idx,
            clip_model=clip_model,
            num_classes=num_classes,
            device=device,
            all_styles=all_styles,
            base_dir=BASE_DIR
        )
        
        all_results.append(result)
        
        # Progress update
        remaining = len(SEEDS) - seed_idx
        print(f"\n✅ Progress: {seed_idx}/{len(SEEDS)} completed, {remaining} remaining")
        
    except Exception as e:
        print(f"\n❌ Error in seed {seed_value}: {e}")
        failed_seeds.append((seed_value, str(e)))
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*70}")
print(f"ALL EXPERIMENTS COMPLETED!")
print(f"  Successful: {len(all_results)}/{len(SEEDS)}")
if failed_seeds:
    print(f"  Failed: {len(failed_seeds)} seeds")
    print(f"  Failed seeds: {[s[0] for s in failed_seeds]}")
print(f"{'='*70}")

# Save summary of completed experiments
summary = {
    "total_seeds": len(SEEDS),
    "successful_experiments": len(all_results),
    "failed_experiments": len(failed_seeds),
    "failed_seeds": [{"seed": s[0], "error": s[1]} for s in failed_seeds],
    "completed_seeds": [r["seed_value"] for r in all_results]
}

with open(os.path.join(METRICS_DIR, "experiments_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Experiments summary saved to: {os.path.join(METRICS_DIR, 'experiments_summary.json')}")


## 8. Load All Results and Compute Statistics

# Load all results if not already loaded
if len(all_results) == 0:
    print("Loading results from JSON files...")
    all_results = []
    for seed_value in SEEDS:
        result_file = os.path.join(METRICS_DIR, "experiments", f"seed_{seed_value}_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))
    print(f"Loaded {len(all_results)} results")

if len(all_results) == 0:
    print("⚠️  No results found! Please run experiments first.")
else:
    print(f"✅ Loaded {len(all_results)} results with per-class metrics")


## 9. Statistical Analysis: Overall Metrics

def calculate_stats(values, name):
    """Calculate comprehensive statistics for a metric"""
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    median_val = np.median(values)
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    cv = (std_val / mean_val * 100) if mean_val != 0 else 0  # Coefficient of Variation
    
    # 95% Confidence Interval
    if len(values) > 1:
        ci = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=stats.sem(values))
        ci_lower, ci_upper = ci
    else:
        ci_lower, ci_upper = mean_val, mean_val
    
    return {
        "metric": name,
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
        "median": float(median_val),
        "q25": float(q25),
        "q75": float(q75),
        "cv_percent": float(cv),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n": len(values)
    }

if len(all_results) > 0:
    # Extract overall metrics
    test_f1s = [r['test_metrics']['test_macro_f1'] for r in all_results]
    test_accs = [r['test_metrics']['test_accuracy'] for r in all_results]
    test_precisions = [r['test_metrics'].get('test_macro_precision', 0) for r in all_results]
    test_recalls = [r['test_metrics'].get('test_macro_recall', 0) for r in all_results]
    
    best_val_f1s = [r['validation_metrics']['best_val_macro_f1'] for r in all_results]
    best_val_accs = [r['validation_metrics']['best_val_accuracy'] for r in all_results]
    best_val_precisions = [r['validation_metrics'].get('best_val_macro_precision', 0) for r in all_results]
    best_val_recalls = [r['validation_metrics'].get('best_val_macro_recall', 0) for r in all_results]
    
    # Create statistics table for overall metrics
    overall_stats = [
        calculate_stats(test_f1s, "Test Macro F1"),
        calculate_stats(test_accs, "Test Accuracy"),
        calculate_stats(test_precisions, "Test Macro Precision"),
        calculate_stats(test_recalls, "Test Macro Recall"),
        calculate_stats(best_val_f1s, "Best Val Macro F1"),
        calculate_stats(best_val_accs, "Best Val Accuracy"),
        calculate_stats(best_val_precisions, "Best Val Macro Precision"),
        calculate_stats(best_val_recalls, "Best Val Macro Recall")
    ]
    
    df_overall_stats = pd.DataFrame(overall_stats)
    
    print("\n" + "="*80)
    print("OVERALL METRICS - Statistical Summary (30 runs)")
    print("="*80)
    print(df_overall_stats.to_string(index=False))
    
    # Save overall statistics
    overall_stats_path = os.path.join(METRICS_DIR, "overall_metrics_statistics.json")
    with open(overall_stats_path, 'w') as f:
        json.dump({"overall_metrics": overall_stats}, f, indent=2)
    
    df_overall_stats.to_csv(os.path.join(METRICS_DIR, "overall_metrics_summary.csv"), index=False)
    
    print(f"\n✅ Overall statistics saved to:")
    print(f"   - {overall_stats_path}")
    print(f"   - {os.path.join(METRICS_DIR, 'overall_metrics_summary.csv')}")
else:
    print("⚠️  No results available for statistical analysis")


## 10. Statistical Analysis: Per-Class Metrics

if len(all_results) > 0:
    # Extract per-class metrics for each style
    per_class_stats = {}
    
    for style in all_styles:
        # Extract F1, Precision, Recall, Accuracy for this style across all runs
        test_f1s_style = []
        test_precisions_style = []
        test_recalls_style = []
        test_accs_style = []
        
        val_f1s_style = []
        val_precisions_style = []
        val_recalls_style = []
        val_accs_style = []
        
        for result in all_results:
            # Test metrics
            test_pc = result['test_metrics'].get('per_class_metrics', {})
            if test_pc and 'f1' in test_pc and style in test_pc['f1']:
                test_f1s_style.append(test_pc['f1'][style])
                test_precisions_style.append(test_pc['precision'][style])
                test_recalls_style.append(test_pc['recall'][style])
                test_accs_style.append(test_pc.get('accuracy', {}).get(style, 0))
            
            # Validation metrics
            val_pc = result['validation_metrics'].get('per_class_metrics', {})
            if val_pc and 'f1' in val_pc and style in val_pc['f1']:
                val_f1s_style.append(val_pc['f1'][style])
                val_precisions_style.append(val_pc['precision'][style])
                val_recalls_style.append(val_pc['recall'][style])
                val_accs_style.append(val_pc.get('accuracy', {}).get(style, 0))
        
        # Calculate statistics for this style
        per_class_stats[style] = {
            'test': {
                'f1': calculate_stats(test_f1s_style, f"{style} - Test F1") if test_f1s_style else None,
                'precision': calculate_stats(test_precisions_style, f"{style} - Test Precision") if test_precisions_style else None,
                'recall': calculate_stats(test_recalls_style, f"{style} - Test Recall") if test_recalls_style else None,
                'accuracy': calculate_stats(test_accs_style, f"{style} - Test Accuracy") if test_accs_style else None
            },
            'validation': {
                'f1': calculate_stats(val_f1s_style, f"{style} - Val F1") if val_f1s_style else None,
                'precision': calculate_stats(val_precisions_style, f"{style} - Val Precision") if val_precisions_style else None,
                'recall': calculate_stats(val_recalls_style, f"{style} - Val Recall") if val_recalls_style else None,
                'accuracy': calculate_stats(val_accs_style, f"{style} - Val Accuracy") if val_accs_style else None
            }
        }
    
    # Create per-class summary tables (Test F1, Precision, Recall, Accuracy)
    per_class_data_f1 = []
    per_class_data_precision = []
    per_class_data_recall = []
    per_class_data_accuracy = []
    
    for style in all_styles:
        if per_class_stats[style]['test']['f1']:
            # F1 summary
            stats_f1 = per_class_stats[style]['test']['f1']
            per_class_data_f1.append({
                'Style': style,
                'Mean': stats_f1['mean'],
                'Std': stats_f1['std'],
                'Min': stats_f1['min'],
                'Max': stats_f1['max'],
                'CI_95_Lower': stats_f1['ci_95_lower'],
                'CI_95_Upper': stats_f1['ci_95_upper'],
                'CV_%': stats_f1['cv_percent']
            })
            
            # Precision summary
            stats_prec = per_class_stats[style]['test']['precision']
            per_class_data_precision.append({
                'Style': style,
                'Mean': stats_prec['mean'],
                'Std': stats_prec['std'],
                'Min': stats_prec['min'],
                'Max': stats_prec['max'],
                'CI_95_Lower': stats_prec['ci_95_lower'],
                'CI_95_Upper': stats_prec['ci_95_upper'],
                'CV_%': stats_prec['cv_percent']
            })
            
            # Recall summary
            stats_rec = per_class_stats[style]['test']['recall']
            per_class_data_recall.append({
                'Style': style,
                'Mean': stats_rec['mean'],
                'Std': stats_rec['std'],
                'Min': stats_rec['min'],
                'Max': stats_rec['max'],
                'CI_95_Lower': stats_rec['ci_95_lower'],
                'CI_95_Upper': stats_rec['ci_95_upper'],
                'CV_%': stats_rec['cv_percent']
            })
            
            # Accuracy summary
            stats_acc = per_class_stats[style]['test']['accuracy']
            per_class_data_accuracy.append({
                'Style': style,
                'Mean': stats_acc['mean'],
                'Std': stats_acc['std'],
                'Min': stats_acc['min'],
                'Max': stats_acc['max'],
                'CI_95_Lower': stats_acc['ci_95_lower'],
                'CI_95_Upper': stats_acc['ci_95_upper'],
                'CV_%': stats_acc['cv_percent']
            })
    
    df_per_class_f1 = pd.DataFrame(per_class_data_f1)
    df_per_class_precision = pd.DataFrame(per_class_data_precision)
    df_per_class_recall = pd.DataFrame(per_class_data_recall)
    df_per_class_accuracy = pd.DataFrame(per_class_data_accuracy)
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS - Test F1-Score (30 runs)")
    print("="*80)
    print(df_per_class_f1.to_string(index=False))
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS - Test Precision (30 runs)")
    print("="*80)
    print(df_per_class_precision.to_string(index=False))
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS - Test Recall (30 runs)")
    print("="*80)
    print(df_per_class_recall.to_string(index=False))
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS - Test Accuracy (30 runs)")
    print("="*80)
    print(df_per_class_accuracy.to_string(index=False))
    
    # Save per-class statistics
    per_class_stats_path = os.path.join(METRICS_DIR, "per_class_metrics_statistics.json")
    with open(per_class_stats_path, 'w') as f:
        json.dump(per_class_stats, f, indent=2)
    
    df_per_class_f1.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_f1_summary.csv"), index=False)
    df_per_class_precision.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_precision_summary.csv"), index=False)
    df_per_class_recall.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_recall_summary.csv"), index=False)
    df_per_class_accuracy.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_accuracy_summary.csv"), index=False)
    
    # Create comprehensive per-class report
    comprehensive_per_class = []
    for style in all_styles:
        if per_class_stats[style]['test']['f1']:
            comprehensive_per_class.append({
                'Style': style,
                'Test_F1_Mean': per_class_stats[style]['test']['f1']['mean'],
                'Test_F1_Std': per_class_stats[style]['test']['f1']['std'],
                'Test_Precision_Mean': per_class_stats[style]['test']['precision']['mean'],
                'Test_Precision_Std': per_class_stats[style]['test']['precision']['std'],
                'Test_Recall_Mean': per_class_stats[style]['test']['recall']['mean'],
                'Test_Recall_Std': per_class_stats[style]['test']['recall']['std'],
                'Test_Accuracy_Mean': per_class_stats[style]['test']['accuracy']['mean'],
                'Test_Accuracy_Std': per_class_stats[style]['test']['accuracy']['std'],
            })
    
    df_comprehensive_per_class = pd.DataFrame(comprehensive_per_class)
    df_comprehensive_per_class.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_summary.csv"), index=False)
    
    print(f"\n✅ Per-class statistics saved to:")
    print(f"   - {per_class_stats_path}")
    print(f"   - {os.path.join(METRICS_DIR, 'per_class_metrics_summary.csv')}")


## 11. Create Summary Table

if len(all_results) > 0:
    # Create summary table for all experiments
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Seed': result['seed_value'],
            'Best_Epoch': result['training_info']['best_epoch'],
            'Early_Stopped': result['training_info']['early_stopped'],
            'Best_Val_Macro_F1': result['validation_metrics']['best_val_macro_f1'],
            'Best_Val_Accuracy': result['validation_metrics']['best_val_accuracy'],
            'Test_Macro_F1': result['test_metrics']['test_macro_f1'],
            'Test_Accuracy': result['test_metrics']['test_accuracy'],
            'Test_Macro_Precision': result['test_metrics']['test_macro_precision'],
            'Test_Macro_Recall': result['test_metrics']['test_macro_recall'],
            'Overfitting': result['overfitting_analysis']['overfitting_detected'],
            'Training_Time_Min': result['training_info']['total_time_minutes']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Seed')
    
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    # Save summary table
    summary_table_path = os.path.join(METRICS_DIR, "summary_table.csv")
    df_summary.to_csv(summary_table_path, index=False)
    print(f"\n✅ Summary table saved to: {summary_table_path}")


## 12. Visualizations

if len(all_results) > 0:
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract metrics
    seeds = [r['seed_value'] for r in all_results]
    test_f1s = [r['test_metrics']['test_macro_f1'] for r in all_results]
    test_accs = [r['test_metrics']['test_accuracy'] for r in all_results]
    val_f1s = [r['validation_metrics']['best_val_macro_f1'] for r in all_results]
    val_accs = [r['validation_metrics']['best_val_accuracy'] for r in all_results]
    
    # Sort by seed
    sorted_indices = np.argsort(seeds)
    seeds_sorted = [seeds[i] for i in sorted_indices]
    test_f1s_sorted = [test_f1s[i] for i in sorted_indices]
    test_accs_sorted = [test_accs[i] for i in sorted_indices]
    val_f1s_sorted = [val_f1s[i] for i in sorted_indices]
    val_accs_sorted = [val_accs[i] for i in sorted_indices]
    
    # Test F1 across seeds
    axes[0, 0].plot(seeds_sorted, test_f1s_sorted, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=np.mean(test_f1s), color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {np.mean(test_f1s):.4f}')
    axes[0, 0].fill_between(seeds_sorted, 
                            [np.mean(test_f1s) - np.std(test_f1s)] * len(seeds_sorted),
                            [np.mean(test_f1s) + np.std(test_f1s)] * len(seeds_sorted),
                            alpha=0.2, color='red')
    axes[0, 0].set_title('Test Macro F1-Score Across Seeds', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Seed Value')
    axes[0, 0].set_ylabel('Test Macro F1')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy across seeds
    axes[0, 1].plot(seeds_sorted, test_accs_sorted, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=np.mean(test_accs), color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {np.mean(test_accs):.2f}%')
    axes[0, 1].fill_between(seeds_sorted,
                            [np.mean(test_accs) - np.std(test_accs)] * len(seeds_sorted),
                            [np.mean(test_accs) + np.std(test_accs)] * len(seeds_sorted),
                            alpha=0.2, color='red')
    axes[0, 1].set_title('Test Accuracy Across Seeds', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Seed Value')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Val vs Test F1 comparison
    axes[1, 0].scatter(val_f1s_sorted, test_f1s_sorted, alpha=0.6, s=100, color='purple')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    axes[1, 0].set_title('Validation vs Test Macro F1', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Best Val Macro F1')
    axes[1, 0].set_ylabel('Test Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of Test F1
    axes[1, 1].hist(test_f1s, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(test_f1s), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(test_f1s):.4f}')
    axes[1, 1].axvline(x=np.median(test_f1s), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(test_f1s):.4f}')
    axes[1, 1].set_title('Distribution of Test Macro F1-Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Test Macro F1')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Image-Only Model: Robustness Analysis Across 30 Seeds', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = os.path.join(ARTIFACTS_DIR, "comparison_plots", "overall_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparison plots saved to: {comparison_plot_path}")
    
    # Per-class visualization
    if len(all_results) > 0 and 'per_class_metrics' in all_results[0]['test_metrics']:
        # Create heatmap of per-class F1 scores
        per_class_f1_matrix = []
        for style in all_styles:
            style_f1s = []
            for result in all_results:
                test_pc = result['test_metrics'].get('per_class_metrics', {})
                if test_pc and 'f1' in test_pc and style in test_pc['f1']:
                    style_f1s.append(test_pc['f1'][style])
                else:
                    style_f1s.append(0.0)
            per_class_f1_matrix.append(style_f1s)
        
        per_class_f1_matrix = np.array(per_class_f1_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(per_class_f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(seeds_sorted)))
        ax.set_xticklabels(seeds_sorted, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(all_styles)))
        ax.set_yticklabels(all_styles, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1-Score', fontsize=10)
        
        ax.set_title('Per-Class F1-Score Heatmap Across 30 Seeds (Image-Only Model)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Seed Value', fontsize=11)
        ax.set_ylabel('Fashion Style', fontsize=11)
        
        plt.tight_layout()
        
        # Save per-class heatmap
        per_class_heatmap_path = os.path.join(ARTIFACTS_DIR, "per_class_visualizations", "per_class_f1_heatmap.png")
        plt.savefig(per_class_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Per-class heatmap saved to: {per_class_heatmap_path}")
        
        # Create bar plot of mean per-class F1 scores
        mean_per_class_f1 = [np.mean([r['test_metrics']['per_class_metrics']['f1'].get(style, 0) 
                                     for r in all_results if 'per_class_metrics' in r['test_metrics'] 
                                     and 'f1' in r['test_metrics']['per_class_metrics'] 
                                     and style in r['test_metrics']['per_class_metrics']['f1']]) 
                            for style in all_styles]
        std_per_class_f1 = [np.std([r['test_metrics']['per_class_metrics']['f1'].get(style, 0) 
                                   for r in all_results if 'per_class_metrics' in r['test_metrics'] 
                                   and 'f1' in r['test_metrics']['per_class_metrics'] 
                                   and style in r['test_metrics']['per_class_metrics']['f1']]) 
                           for style in all_styles]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x_pos = np.arange(len(all_styles))
        bars = ax.bar(x_pos, mean_per_class_f1, yerr=std_per_class_f1, 
                     capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Fashion Style', fontsize=11)
        ax.set_ylabel('Mean F1-Score (with std)', fontsize=11)
        ax.set_title('Mean Per-Class F1-Score Across 30 Seeds (Image-Only Model)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_styles, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(mean_per_class_f1, std_per_class_f1)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save per-class bar plot
        per_class_bar_path = os.path.join(ARTIFACTS_DIR, "per_class_visualizations", "per_class_f1_barplot.png")
        plt.savefig(per_class_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Per-class bar plot saved to: {per_class_bar_path}")

print("\n✅ All visualizations completed!")

