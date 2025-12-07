"""
Evaluation Analysis Script

Compares human-labeled evaluation scores with automatic evaluation scores
to calibrate and validate the automatic metrics.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# File paths
HUMAN_EVAL_FILE = "human_eval_20.json"
MODEL_EVAL_FILE = "model_evaluation_report.json"


def load_data():
    """Load human evaluation and model evaluation data."""
    print("Loading data...")
    
    # Check if files exist
    if not os.path.exists(HUMAN_EVAL_FILE):
        raise FileNotFoundError(f"Human evaluation file not found: {HUMAN_EVAL_FILE}")
    if not os.path.exists(MODEL_EVAL_FILE):
        raise FileNotFoundError(f"Model evaluation file not found: {MODEL_EVAL_FILE}")
    
    # Check file sizes
    human_size = os.path.getsize(HUMAN_EVAL_FILE)
    model_size = os.path.getsize(MODEL_EVAL_FILE)
    
    if human_size == 0:
        raise ValueError(f"Human evaluation file is empty: {HUMAN_EVAL_FILE}")
    if model_size == 0:
        raise ValueError(f"Model evaluation file is empty: {MODEL_EVAL_FILE}")
    
    print(f"  Human eval file size: {human_size} bytes")
    print(f"  Model eval file size: {model_size} bytes")
    
    # Load human evaluation data
    try:
        with open(HUMAN_EVAL_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Human evaluation file appears to be empty or contains only whitespace")
            human_data = json.loads(content)
        print(f"  [OK] Loaded human evaluation data: {len(human_data.get('items', []))} items")
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse {HUMAN_EVAL_FILE}")
        print(f"  Error: {e}")
        print(f"  First 200 characters of file:")
        with open(HUMAN_EVAL_FILE, 'r', encoding='utf-8') as f:
            print(f"  {f.read()[:200]}")
        raise
    
    # Load model evaluation data
    try:
        with open(MODEL_EVAL_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Model evaluation file appears to be empty or contains only whitespace")
            model_data = json.loads(content)
        print(f"  [OK] Loaded model evaluation data: {len(model_data.get('detailed_results', []))} results")
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse {MODEL_EVAL_FILE}")
        print(f"  Error: {e}")
        print(f"  First 200 characters of file:")
        with open(MODEL_EVAL_FILE, 'r', encoding='utf-8') as f:
            print(f"  {f.read()[:200]}")
        raise
    
    return human_data, model_data


def calculate_query_similarity(query1, query2):
    """Calculate similarity between two queries using multiple methods."""
    q1 = query1.lower().strip()
    q2 = query2.lower().strip()
    
    # Exact match
    if q1 == q2:
        return 1.0
    
    # Word-based Jaccard similarity
    words1 = set(q1.split())
    words2 = set(q2.split())
    common_words = len(words1 & words2)
    total_words = len(words1 | words2)
    jaccard = common_words / total_words if total_words > 0 else 0
    
    # Substring similarity (for partial matches)
    if q1 in q2 or q2 in q1:
        substring_bonus = 0.2
    else:
        substring_bonus = 0
    
    # Key phrase matching (important words)
    key_phrases = ['ginger', 'whiskey', 'tequila', 'gin', 'tropical', 'party', 
                   'low-alcohol', 'brunch', 'complex', 'quick', 'easy', 'winter', 
                   'summer', 'refreshing', 'sweet', 'bitter', 'spicy']
    q1_phrases = [p for p in key_phrases if p in q1]
    q2_phrases = [p for p in key_phrases if p in q2]
    phrase_match = len(set(q1_phrases) & set(q2_phrases)) / max(len(set(q1_phrases) | set(q2_phrases)), 1)
    
    # Combined score
    similarity = jaccard * 0.6 + phrase_match * 0.3 + substring_bonus
    
    return min(1.0, similarity)


def match_queries(human_data, model_data):
    """
    Match human evaluation queries with model evaluation queries.
    Returns a list of matched pairs.
    """
    print("\nMatching queries between human and model evaluations...")
    
    human_items = human_data['items']
    model_results = model_data['detailed_results']
    
    matched_pairs = []
    used_model_indices = set()  # Track which model results we've used
    
    # Try to match queries by similarity
    for human_item in human_items:
        human_query = human_item['query']
        best_match = None
        best_score = 0
        best_idx = -1
        
        # Find the best matching query in model results
        # Prefer Gemini-3-Pro (best model) if available
        candidates = []
        for idx, model_result in enumerate(model_results):
            if idx in used_model_indices:
                continue
                
            model_query = model_result['query']
            similarity = calculate_query_similarity(human_query, model_query)
            
            if similarity > 0.4:  # At least 40% similarity
                candidates.append((similarity, model_result, idx, model_result.get('model', '')))
        
        # Sort by: 1) Gemini-3-Pro preference, 2) similarity score
        if candidates:
            candidates.sort(key=lambda x: (
                x[3] != 'Gemini-3-Pro',  # False (0) for Gemini comes first
                -x[0]  # Higher similarity first
            ))
            best_score, best_match, best_idx, _ = candidates[0]
        
        if best_match and best_match['metrics']:  # Ensure metrics exist
            used_model_indices.add(best_idx)
            matched_pairs.append({
                'id': human_item['id'],
                'query': human_item['query'],
                'human_scores': {
                    'relevance': human_item['human_relevance_score'],
                    'completeness': human_item['human_completeness_score'],
                    'avoids_repetition': human_item['human_avoids_repetition_score'],
                    'overall': human_item['human_overall_score']
                },
                'model_scores': {
                    'relevance': best_match['metrics']['relevance_score'],
                    'completeness': best_match['metrics']['completeness_score'],
                    'avoids_repetition': best_match['metrics']['avoids_repetition_score'],
                    'overall': best_match['metrics']['overall_score']
                },
                'model_name': best_match['model'],
                'similarity': best_score,
                'human_notes': human_item.get('human_notes', '')
            })
            print(f"  [OK] Matched ID {human_item['id']} (similarity: {best_score:.3f})")
        else:
            print(f"  [WARNING] No match found for human query ID {human_item['id']}: {human_item['query'][:60]}...")
    
    print(f"\nMatched {len(matched_pairs)} out of {len(human_data['items'])} queries")
    return matched_pairs


def calculate_correlation_metrics(human_scores, auto_scores):
    """Calculate correlation and error metrics."""
    if len(human_scores) == 0 or len(auto_scores) == 0:
        return {}
    
    human_arr = np.array(human_scores)
    auto_arr = np.array(auto_scores)
    
    # Remove NaN values
    mask = ~(np.isnan(human_arr) | np.isnan(auto_arr))
    human_arr = human_arr[mask]
    auto_arr = auto_arr[mask]
    
    if len(human_arr) < 2:
        return {}
    
    pearson_r, pearson_p = pearsonr(human_arr, auto_arr)
    spearman_r, spearman_p = spearmanr(human_arr, auto_arr)
    mae = mean_absolute_error(human_arr, auto_arr)
    rmse = np.sqrt(mean_squared_error(human_arr, auto_arr))
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(human_arr)
    }


def analyze_metrics(matched_pairs):
    """Analyze correlation between human and automatic metrics."""
    print("\n" + "="*80)
    print("METRIC CORRELATION ANALYSIS")
    print("="*80)
    
    metrics = ['relevance', 'completeness', 'avoids_repetition', 'overall']
    results = {}
    
    for metric in metrics:
        human_scores = [pair['human_scores'][metric] for pair in matched_pairs]
        auto_scores = [pair['model_scores'][metric] for pair in matched_pairs]
        
        stats = calculate_correlation_metrics(human_scores, auto_scores)
        results[metric] = stats
        
        if stats:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Pearson correlation:  {stats['pearson_r']:.4f} (p={stats['pearson_p']:.4f})")
            print(f"  Spearman correlation: {stats['spearman_r']:.4f} (p={stats['spearman_p']:.4f})")
            print(f"  Mean Absolute Error:  {stats['mae']:.4f}")
            print(f"  Root Mean Squared Error: {stats['rmse']:.4f}")
            print(f"  Sample size: {stats['n_samples']}")
    
    return results


def create_comparison_dataframe(matched_pairs):
    """Create a DataFrame for easier analysis."""
    data = []
    for pair in matched_pairs:
        data.append({
            'id': pair['id'],
            'query': pair['query'][:60] + '...' if len(pair['query']) > 60 else pair['query'],
            'model': pair['model_name'],
            'human_relevance': pair['human_scores']['relevance'],
            'auto_relevance': pair['model_scores']['relevance'],
            'human_completeness': pair['human_scores']['completeness'],
            'auto_completeness': pair['model_scores']['completeness'],
            'human_avoids_repetition': pair['human_scores']['avoids_repetition'],
            'auto_avoids_repetition': pair['model_scores']['avoids_repetition'],
            'human_overall': pair['human_scores']['overall'],
            'auto_overall': pair['model_scores']['overall'],
            'relevance_diff': abs(pair['human_scores']['relevance'] - pair['model_scores']['relevance']),
            'completeness_diff': abs(pair['human_scores']['completeness'] - pair['model_scores']['completeness']),
            'overall_diff': abs(pair['human_scores']['overall'] - pair['model_scores']['overall']),
        })
    
    return pd.DataFrame(data)


def plot_comparisons(df, results):
    """Create visualization plots."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    metrics = ['relevance', 'completeness', 'avoids_repetition', 'overall']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        human_col = f'human_{metric}'
        auto_col = f'auto_{metric}'
        
        # Scatter plot
        ax.scatter(df[human_col], df[auto_col], alpha=0.6, s=100)
        
        # Add diagonal line (perfect correlation)
        min_val = min(df[human_col].min(), df[auto_col].min())
        max_val = max(df[human_col].max(), df[auto_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect correlation')
        
        # Add correlation info
        if metric in results and results[metric]:
            r = results[metric]['pearson_r']
            ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'Human {metric.capitalize()} Score', fontsize=12)
        ax.set_ylabel(f'Automatic {metric.capitalize()} Score', fontsize=12)
        ax.set_title(f'{metric.capitalize().replace("_", " ")}: Human vs Automatic', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: evaluation_comparison.png")
    
    # Create error distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    error_metrics = ['relevance', 'completeness', 'overall']
    for idx, metric in enumerate(error_metrics):
        ax = axes[idx]
        diff_col = f'{metric}_diff'
        
        ax.hist(df[diff_col], bins=15, alpha=0.7, color=sns.color_palette("husl")[idx])
        ax.axvline(df[diff_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[diff_col].mean():.3f}')
        ax.set_xlabel(f'Absolute Error ({metric.capitalize()})', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Error Distribution: {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: error_distribution.png")
    
    # Create detailed comparison table visualization
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"ID {int(row['id'])}",
            row['query'],
            row['model'],
            f"{row['human_overall']:.2f}",
            f"{row['auto_overall']:.2f}",
            f"{row['overall_diff']:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['ID', 'Query (truncated)', 'Model', 'Human Overall', 'Auto Overall', 'Error'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.05, 0.4, 0.15, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code cells by error
    for i in range(1, len(table_data) + 1):
        error = float(table_data[i-1][5])
        if error < 0.1:
            color = 'lightgreen'
        elif error < 0.2:
            color = 'lightyellow'
        else:
            color = 'lightcoral'
        table[(i, 5)].set_facecolor(color)
    
    plt.title('Query-by-Query Comparison: Human vs Automatic Scores', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('detailed_comparison_table.png', dpi=300, bbox_inches='tight')
    print("Saved: detailed_comparison_table.png")


def generate_summary_report(df, results, matched_pairs):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print("\n1. OVERALL METRIC PERFORMANCE:")
    print("-" * 80)
    
    metric_names = {
        'relevance': 'Relevance Score',
        'completeness': 'Completeness Score',
        'avoids_repetition': 'Avoids Repetition Score',
        'overall': 'Overall Score'
    }
    
    for metric, name in metric_names.items():
        if metric in results and results[metric]:
            r = results[metric]['pearson_r']
            mae = results[metric]['mae']
            rmse = results[metric]['rmse']
            
            # Interpretation
            if r > 0.7:
                quality = "Excellent"
            elif r > 0.5:
                quality = "Good"
            elif r > 0.3:
                quality = "Moderate"
            else:
                quality = "Poor"
            
            print(f"\n{name}:")
            print(f"  Correlation: {r:.4f} ({quality})")
            print(f"  Mean Absolute Error: {mae:.4f}")
            print(f"  Root Mean Squared Error: {rmse:.4f}")
    
    print("\n2. BEST AND WORST MATCHES:")
    print("-" * 80)
    
    # Best matches (lowest error)
    best_matches = df.nsmallest(5, 'overall_diff')
    print("\nTop 5 Best Matches (lowest error):")
    for idx, row in best_matches.iterrows():
        print(f"  ID {int(row['id'])}: Error = {row['overall_diff']:.4f}")
        print(f"    Query: {row['query']}")
        print(f"    Human: {row['human_overall']:.3f}, Auto: {row['auto_overall']:.3f}")
    
    # Worst matches (highest error)
    worst_matches = df.nlargest(5, 'overall_diff')
    print("\nTop 5 Worst Matches (highest error):")
    for idx, row in worst_matches.iterrows():
        print(f"  ID {int(row['id'])}: Error = {row['overall_diff']:.4f}")
        print(f"    Query: {row['query']}")
        print(f"    Human: {row['human_overall']:.3f}, Auto: {row['auto_overall']:.3f}")
    
    print("\n3. STATISTICAL SUMMARY:")
    print("-" * 80)
    print(f"Total matched queries: {len(df)}")
    print(f"Mean absolute error (overall): {df['overall_diff'].mean():.4f}")
    print(f"Median absolute error (overall): {df['overall_diff'].median():.4f}")
    print(f"Standard deviation of error: {df['overall_diff'].std():.4f}")
    
    # Model breakdown
    print("\n4. ERROR BY MODEL:")
    print("-" * 80)
    model_errors = df.groupby('model')['overall_diff'].agg(['mean', 'std', 'count'])
    print(model_errors)
    
    # Save detailed report to CSV
    df.to_csv('evaluation_comparison.csv', index=False, encoding='utf-8-sig')
    print("\n5. DATA EXPORT:")
    print("-" * 80)
    print("Saved detailed comparison to: evaluation_comparison.csv")


def main():
    """Main analysis function."""
    print("="*80)
    print("EVALUATION METRIC CALIBRATION ANALYSIS")
    print("="*80)
    print("\nComparing human-labeled scores with automatic evaluation scores")
    print("to validate and calibrate the automatic metrics.\n")
    
    # Load data
    human_data, model_data = load_data()
    
    # Match queries
    matched_pairs = match_queries(human_data, model_data)
    
    if len(matched_pairs) == 0:
        print("\n[ERROR] No queries matched! Cannot perform analysis.")
        return
    
    # Create DataFrame
    df = create_comparison_dataframe(matched_pairs)
    
    # Analyze metrics
    results = analyze_metrics(matched_pairs)
    
    # Generate visualizations
    try:
        plot_comparisons(df, results)
    except Exception as e:
        print(f"\n[WARNING] Error generating plots: {e}")
        print("Continuing with text analysis...")
    
    # Generate summary report
    generate_summary_report(df, results, matched_pairs)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - evaluation_comparison.png (scatter plots)")
    print("  - error_distribution.png (error histograms)")
    print("  - detailed_comparison_table.png (query-by-query table)")
    print("  - evaluation_comparison.csv (detailed data)")


if __name__ == "__main__":
    main()

