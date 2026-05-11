"""
Generate comprehensive results report comparing all models:
- Multi-Modal (CLIP + FashionBERT)
- Image-Only (CLIP)
- Text-Only (FashionBERT)
- ResNet50 Baseline
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent


def _p(*parts: str) -> str:
    return str(_ROOT.joinpath(*parts))

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'multimodal': '#2ecc71',  # Green
    'imageonly': '#3498db',   # Blue
    'textonly': '#e74c3c',    # Red
    'resnet50': '#95a5a6'     # Gray
}

def load_stats(filepath):
    """Load statistics from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_summary_table(filepath):
    """Load summary table CSV"""
    return pd.read_csv(filepath)

def extract_f1_scores(summary_table_path):
    """Extract F1 scores from summary table"""
    df = pd.read_csv(summary_table_path)
    return df['Test_Macro_F1'].values

def create_overall_comparison_table():
    """Create main comparison table"""
    
    # Load statistics
    multimodal_stats = load_stats(_p('experiments', 'phase3_robustness', 'metrics', 'overall_metrics_statistics.json'))
    imageonly_stats = load_stats(_p('experiments', 'imageonly_robustness', 'metrics', 'overall_metrics_statistics.json'))
    textonly_stats = load_stats(_p('experiments', 'textonly_robustness', 'metrics', 'overall_metrics_statistics.json'))
    
    # ResNet50 (single run - from notebook output)
    resnet50_f1 = 0.7396
    resnet50_acc = 73.89
    
    # Extract metrics
    def get_metric(stats_dict, metric_name):
        for item in stats_dict.get('overall_metrics', []):
            if item['metric'] == metric_name:
                return item
        return None
    
    # Multi-Modal
    mm_f1 = get_metric(multimodal_stats, 'Test Macro F1')
    mm_acc = get_metric(multimodal_stats, 'Test Accuracy')
    
    # Image-Only
    io_f1 = get_metric(imageonly_stats, 'Test Macro F1')
    io_acc = get_metric(imageonly_stats, 'Test Accuracy')
    
    # Text-Only
    to_f1 = get_metric(textonly_stats, 'Test Macro F1')
    to_acc = get_metric(textonly_stats, 'Test Accuracy')
    
    # Create comparison table
    comparison_data = {
        'Model': ['Multi-Modal', 'Image-Only', 'Text-Only', 'ResNet50 Baseline'],
        'Test Macro F1': [
            f"{mm_f1['mean']:.3f} ± {mm_f1['std']:.3f}",
            f"{io_f1['mean']:.3f} ± {io_f1['std']:.3f}",
            f"{to_f1['mean']:.3f} ± {to_f1['std']:.3f}",
            f"{resnet50_f1:.3f}"
        ],
        '95% CI (F1)': [
            f"[{mm_f1['ci_95_lower']:.3f}, {mm_f1['ci_95_upper']:.3f}]",
            f"[{io_f1['ci_95_lower']:.3f}, {io_f1['ci_95_upper']:.3f}]",
            f"[{to_f1['ci_95_lower']:.3f}, {to_f1['ci_95_upper']:.3f}]",
            "-"
        ],
        'Test Accuracy (%)': [
            f"{mm_acc['mean']:.2f} ± {mm_acc['std']:.2f}",
            f"{io_acc['mean']:.2f} ± {io_acc['std']:.2f}",
            f"{to_acc['mean']:.2f} ± {to_acc['std']:.2f}",
            f"{resnet50_acc:.2f}"
        ],
        'CV (%)': [
            f"{mm_f1['cv_percent']:.2f}",
            f"{io_f1['cv_percent']:.2f}",
            f"{to_f1['cv_percent']:.2f}",
            "-"
        ],
        'Min F1': [
            f"{mm_f1['min']:.3f}",
            f"{io_f1['min']:.3f}",
            f"{to_f1['min']:.3f}",
            "-"
        ],
        'Max F1': [
            f"{mm_f1['max']:.3f}",
            f"{io_f1['max']:.3f}",
            f"{to_f1['max']:.3f}",
            "-"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    return df, {
        'multimodal': mm_f1,
        'imageonly': io_f1,
        'textonly': to_f1,
        'resnet50': {'mean': resnet50_f1, 'std': 0}
    }

def plot_performance_comparison(stats_dict):
    """Create bar chart comparing model performance"""
    
    models = ['Multi-Modal', 'Image-Only', 'Text-Only', 'ResNet50']
    means = [
        stats_dict['multimodal']['mean'],
        stats_dict['imageonly']['mean'],
        stats_dict['textonly']['mean'],
        stats_dict['resnet50']['mean']
    ]
    stds = [
        stats_dict['multimodal']['std'],
        stats_dict['imageonly']['std'],
        stats_dict['textonly']['std'],
        0  # ResNet50 single run
    ]
    colors = [COLORS['multimodal'], COLORS['imageonly'], COLORS['textonly'], COLORS['resnet50']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test Macro F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.65, 0.90])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        height = mean + std + 0.005
        ax.text(i, height, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight best performance
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_performance_comparison.png")
    plt.close()

def plot_performance_distribution():
    """Create box plot showing distribution across 30 runs"""
    
    # Load F1 scores from summary tables
    multimodal_f1 = extract_f1_scores(_p('experiments', 'phase3_robustness', 'metrics', 'summary_table.csv'))
    imageonly_f1 = extract_f1_scores(_p('experiments', 'imageonly_robustness', 'metrics', 'summary_table.csv'))
    textonly_f1 = extract_f1_scores(_p('experiments', 'textonly_robustness', 'metrics', 'summary_table.csv'))
    resnet50_f1 = np.array([0.7396] * 30)  # Single value repeated for visualization
    
    data = [multimodal_f1, imageonly_f1, textonly_f1, resnet50_f1]
    labels = ['Multi-Modal', 'Image-Only', 'Text-Only', 'ResNet50']
    colors = [COLORS['multimodal'], COLORS['imageonly'], COLORS['textonly'], COLORS['resnet50']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                    showmeans=True, meanline=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style the plot
    ax.set_ylabel('Test Macro F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Distribution Across 30 Data Splits', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('performance_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: performance_distribution.png")
    plt.close()

def create_per_class_heatmap():
    """Create heatmap comparing per-class F1 scores"""
    
    # Load per-class metrics
    mm_per_class = pd.read_csv(_p('experiments', 'phase3_robustness', 'metrics', 'per_class_metrics_summary.csv'))
    io_per_class = pd.read_csv(_p('experiments', 'imageonly_robustness', 'metrics', 'per_class_metrics_summary.csv'))
    to_per_class = pd.read_csv(_p('experiments', 'textonly_robustness', 'metrics', 'per_class_metrics_summary.csv'))
    
    # ResNet50 per-class (from notebook output - you'll need to extract this)
    # For now, using placeholder
    resnet50_per_class = {
        'conservative': 0.67, 'dressy': 0.91, 'ethnic': 0.78, 'fairy': 0.91,
        'feminine': 0.75, 'gal': 0.71, 'girlish': 0.58, 'kireime-casual': 0.58,
        'lolita': 0.91, 'mode': 0.71, 'natural': 0.72, 'retro': 0.64, 'rock': 0.XX, 'street': 0.XX
    }
    
    # Prepare data
    styles = mm_per_class['Style'].values
    data = {
        'Style': styles,
        'Multi-Modal': mm_per_class['Test_F1_Mean'].values,
        'Image-Only': io_per_class['Test_F1_Mean'].values,
        'Text-Only': to_per_class['Test_F1_Mean'].values,
        'ResNet50': [resnet50_per_class.get(style, 0.0) for style in styles]
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('Style')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fashion Style', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('per_class_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: per_class_heatmap.png")
    plt.close()

def perform_statistical_tests():
    """Perform pairwise statistical tests"""
    
    # Load F1 scores
    multimodal_f1 = extract_f1_scores(_p('experiments', 'phase3_robustness', 'metrics', 'summary_table.csv'))
    imageonly_f1 = extract_f1_scores(_p('experiments', 'imageonly_robustness', 'metrics', 'summary_table.csv'))
    textonly_f1 = extract_f1_scores(_p('experiments', 'textonly_robustness', 'metrics', 'summary_table.csv'))
    
    # Perform paired t-tests
    comparisons = [
        ('Multi-Modal vs Image-Only', multimodal_f1, imageonly_f1),
        ('Multi-Modal vs Text-Only', multimodal_f1, textonly_f1),
        ('Image-Only vs Text-Only', imageonly_f1, textonly_f1),
    ]
    
    results = []
    for name, data1, data2 in comparisons:
        t_stat, p_value = stats.ttest_rel(data1, data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        cohens_d = mean_diff / np.sqrt((np.var(data1) + np.var(data2)) / 2)
        
        results.append({
            'Comparison': name,
            'Mean Difference': f"{mean_diff:.4f}",
            't-statistic': f"{t_stat:.4f}",
            'p-value': f"{p_value:.6f}",
            "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No",
            "Cohen's d": f"{cohens_d:.4f}",
            'Effect Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        })
    
    df = pd.DataFrame(results)
    return df

def main():
    """Main function to generate all reports"""
    
    print("=" * 70)
    print("Generating Results Report")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('results_report')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create comparison table
    print("\n1. Creating overall comparison table...")
    comparison_table, stats_dict = create_overall_comparison_table()
    comparison_table.to_csv(output_dir / 'overall_comparison_table.csv', index=False)
    print(comparison_table.to_string(index=False))
    print(f"\n✅ Saved: {output_dir / 'overall_comparison_table.csv'}")
    
    # 2. Create performance comparison plot
    print("\n2. Creating performance comparison plot...")
    plot_performance_comparison(stats_dict)
    
    # 3. Create distribution plot
    print("\n3. Creating performance distribution plot...")
    plot_performance_distribution()
    
    # 4. Create per-class heatmap
    print("\n4. Creating per-class heatmap...")
    try:
        create_per_class_heatmap()
    except Exception as e:
        print(f"⚠️  Warning: Could not create per-class heatmap: {e}")
    
    # 5. Statistical tests
    print("\n5. Performing statistical tests...")
    stats_tests = perform_statistical_tests()
    stats_tests.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print(stats_tests.to_string(index=False))
    print(f"\n✅ Saved: {output_dir / 'statistical_tests.csv'}")
    
    # 6. Generate LaTeX table (for papers)
    print("\n6. Generating LaTeX table...")
    latex_table = comparison_table.to_latex(index=False, escape=False)
    with open(output_dir / 'comparison_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"✅ Saved: {output_dir / 'comparison_table.tex'}")
    
    print("\n" + "=" * 70)
    print("Report generation complete!")
    print(f"All files saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()




