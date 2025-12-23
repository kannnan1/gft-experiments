#!/usr/bin/env python3
"""Aggregate results across multiple experiment runs with naming and LoRA/GFT comparison."""

import json
import yaml
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

def get_metadata_map(config_dir: Path) -> Dict[str, Dict]:
    """Build a mapping of exp_id -> metadata by parsing config files."""
    meta_map = {}
    for cfg_file in config_dir.rglob("*.yaml"):
        try:
            with open(cfg_file, 'r') as f:
                cfg = yaml.safe_load(f)
                if not cfg or 'experiment' not in cfg:
                    continue
                exp_id = cfg['experiment'].get('exp_id')
                if exp_id:
                    meta_map[exp_id] = {
                        'name': cfg['experiment'].get('name', exp_id),
                        'method': cfg['model'].get('method', 'unknown'),
                        'task': cfg['data'].get('adapt_task', 'unknown'),
                        'rank': cfg['model'].get('rank', 0)
                    }
        except Exception:
            continue
    return meta_map

def load_experiment_results(log_dir: Path, meta_map: Dict) -> List[Dict]:
    """Load all experiment results and enrich with metadata."""
    results = []
    
    # Find all metrics JSON files
    for metrics_file in log_dir.rglob("*_metrics.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        # Extract experiment info from path
        exp_id_raw = metrics_file.parent.name
        exp_id = exp_id_raw.split('_seed')[0] if '_seed' in exp_id_raw else exp_id_raw
        
        # Merge metrics
        all_metrics = {}
        if data:
            if isinstance(data, dict) and 'history' in data:
                history = data['history']
                config = data.get('config', {})
            else:
                history = data
                config = {}
                
            for entry in history:
                if 'metrics' in entry:
                    all_metrics.update(entry['metrics'])
            
            meta = meta_map.get(exp_id, {})
            results.append({
                'exp_id': exp_id,
                'exp_id_full': exp_id_raw,
                'exp_name': meta.get('name', exp_id),
                'method': meta.get('method', config.get('model', {}).get('method', 'unknown')),
                'task': meta.get('task', config.get('data', {}).get('adapt_task', 'unknown')),
                'metrics': all_metrics
            })
    
    return results

def aggregate_by_experiment(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results by experiment ID across seeds."""
    grouped = {}
    for result in results:
        eid = result['exp_id']
        if eid not in grouped:
            grouped[eid] = []
        grouped[eid].append(result)
    
    aggregated = []
    for eid, exp_results in grouped.items():
        key_mapping = {
            'forgetting/base_accuracy': 'base_accuracy',
            'forgetting/adaptation_accuracy': 'adaptation_accuracy',
            'final/adaptation_accuracy': 'adaptation_accuracy',
            'forgetting/retention_accuracy': 'retention_accuracy',
            'forgetting/forgetting_percent': 'forgetting_percent'
        }
        
        metrics_list = {v: [] for v in key_mapping.values()}
        for res in exp_results:
            for json_key, internal_key in key_mapping.items():
                if json_key in res['metrics']:
                    metrics_list[internal_key].append(res['metrics'][json_key])
        
        first = exp_results[0]
        row = {
            'exp_id': eid,
            'exp_name': first['exp_name'],
            'method': first['method'],
            'task': first['task'],
            'n_seeds': len(exp_results)
        }
        
        for key, values in metrics_list.items():
            if values:
                row[f'{key}_mean'] = np.mean(values)
                row[f'{key}_std'] = np.std(values)
            else:
                row[f'{key}_mean'] = 0.0
                row[f'{key}_std'] = 0.0
        
        aggregated.append(row)
    
    df = pd.DataFrame(aggregated)
    if not df.empty:
        df = df.sort_values(['task', 'method'])
    return df

def create_results_plot(df: pd.DataFrame, output_dir: Path):
    """Create summarized bar plot using experiment names."""
    if df.empty: return

    plot_data = []
    for _, row in df.iterrows():
        display_name = row['exp_name']
        for m, label in [('base_accuracy', 'Base'), ('adaptation_accuracy', 'Adapt'), ('retention_accuracy', 'Retention')]:
            plot_data.append({
                'Experiment': display_name,
                'Metric': label,
                'Accuracy (%)': row[f'{m}_mean']
            })
    
    pdf = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=pdf, x='Experiment', y='Accuracy (%)', hue='Metric', palette='viridis')
    
    plt.title('Performance Across Experiments (by Name)', fontsize=20, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    
    # Add value labels
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width()/2., h), 
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), 
                        textcoords='offset points')
    
    plt.tight_layout()
    plot_file = output_dir.parent / "plots" / "summary_by_name.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300)
    print(f"Saved summary plot to: {plot_file}")
    plt.close()

def create_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Compare LoRA vs GFT across tasks."""
    # Filter for LoRA and GFT methods
    compare_df = df[df['method'].isin(['lora', 'gft'])].copy()
    if compare_df.empty:
        print("No LoRA vs GFT data found for comparison plot.")
        return

    # Create Task labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # Left: Retention Accuray
    sns.barplot(data=compare_df, x='task', y='retention_accuracy_mean', hue='method', ax=ax1, palette='Set2')
    ax1.set_title('Retention Accuracy (Higher is Better)', fontsize=15)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Right: Forgetting %
    sns.barplot(data=compare_df, x='task', y='forgetting_percent_mean', hue='method', ax=ax2, palette='Set2')
    ax2.set_title('Forgetting % (Lower is Better)', fontsize=15)
    ax2.set_ylabel('Forgetting (%)')
    
    plt.suptitle('Direct Comparison: LoRA vs GFT', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_file = output_dir.parent / "plots" / "lora_vs_gft_comparison.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved comparison plot to: {plot_file}")
    plt.close()

def create_results_table(df: pd.DataFrame, output_path: Path):
    """Save aggregated results to table files."""
    display_rows = []
    for _, row in df.iterrows():
        display_rows.append({
            'Exp ID': row['exp_id'],
            'Exp Name': row['exp_name'],
            'Method': row['method'].upper(),
            'Task': row['task'],
            'Base Acc': f"{row['base_accuracy_mean']:.2f}±{row['base_accuracy_std']:.2f}",
            'Adapt Acc': f"{row['adaptation_accuracy_mean']:.2f}±{row['adaptation_accuracy_std']:.2f}",
            'Retention Acc': f"{row['retention_accuracy_mean']:.2f}±{row['retention_accuracy_std']:.2f}",
            'Forgetting %': f"{row['forgetting_percent_mean']:.2f}±{row['forgetting_percent_std']:.2f}"
        })
        
    table_df = pd.DataFrame(display_rows)
    table_df.to_csv(output_path / "aggregated_results.csv", index=False)
    with open(output_path / "aggregated_results.md", 'w') as f:
        f.write("# GiFT Experiments - Aggregated Results\n\n")
        f.write(table_df.to_markdown(index=False))

def main():
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / "results" / "logs"
    config_dir = base_dir / "configs"
    output_dir = base_dir / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Parsing configs for metadata...")
    meta_map = get_metadata_map(config_dir)
    
    print("Loading experiment results...")
    results = load_experiment_results(log_dir, meta_map)
    print(f"Found {len(results)} experiment runs.")
    
    if not results:
        print("No results found!")
        return
    
    print("Aggregating results...")
    aggregated_df = aggregate_by_experiment(results)
    
    print("Creating table and plots...")
    create_results_table(aggregated_df, output_dir)
    create_results_plot(aggregated_df, output_dir)
    create_comparison_plot(aggregated_df, output_dir)
    
    print("\nAggregation complete!")

if __name__ == '__main__':
    main()
