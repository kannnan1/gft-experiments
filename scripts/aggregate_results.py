#!/usr/bin/env python3
"""Aggregate results across multiple experiment runs."""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List


def load_experiment_results(log_dir: Path) -> List[Dict]:
    """Load all experiment results from log directory.
    
    Args:
        log_dir: Path to logs directory
        
    Returns:
        List of experiment result dictionaries
    """
    results = []
    
    # Find all metrics JSON files
    for metrics_file in log_dir.rglob("*_metrics.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        # Extract experiment info from path
        exp_id = metrics_file.parent.name
        
        # Get final metrics
        if data:
            final_metrics = data[-1]['metrics']
            results.append({
                'exp_id': exp_id,
                'metrics': final_metrics,
                'history': data
            })
    
    return results


def aggregate_by_experiment(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results by experiment ID across seeds.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with aggregated results
    """
    # Group by experiment ID (without seed)
    grouped = {}
    for result in results:
        exp_id = result['exp_id'].rsplit('_seed', 1)[0]
        if exp_id not in grouped:
            grouped[exp_id] = []
        grouped[exp_id].append(result)
    
    # Aggregate metrics
    aggregated = []
    for exp_id, exp_results in grouped.items():
        # Extract key metrics
        metrics_list = {
            'base_accuracy': [],
            'adaptation_accuracy': [],
            'retention_accuracy': [],
            'forgetting_percent': []
        }
        
        for result in exp_results:
            metrics = result['metrics']
            for key in metrics_list.keys():
                if f'forgetting/{key}' in metrics:
                    metrics_list[key].append(metrics[f'forgetting/{key}'])
                elif f'final/{key}' in metrics:
                    metrics_list[key].append(metrics[f'final/{key}'])
        
        # Compute mean and std
        agg_row = {'exp_id': exp_id, 'n_seeds': len(exp_results)}
        for key, values in metrics_list.items():
            if values:
                agg_row[f'{key}_mean'] = np.mean(values)
                agg_row[f'{key}_std'] = np.std(values)
        
        aggregated.append(agg_row)
    
    return pd.DataFrame(aggregated)


def create_results_table(df: pd.DataFrame, output_path: Path):
    """Create formatted results table.
    
    Args:
        df: Aggregated results DataFrame
        output_path: Path to save table
    """
    # Format for display
    table_rows = []
    for _, row in df.iterrows():
        table_row = {
            'Exp ID': row['exp_id'],
            'Base Acc': f"{row.get('base_accuracy_mean', 0):.2f}±{row.get('base_accuracy_std', 0):.2f}",
            'Adapt Acc': f"{row.get('adaptation_accuracy_mean', 0):.2f}±{row.get('adaptation_accuracy_std', 0):.2f}",
            'Retention Acc': f"{row.get('retention_accuracy_mean', 0):.2f}±{row.get('retention_accuracy_std', 0):.2f}",
            'Forgetting %': f"{row.get('forgetting_percent_mean', 0):.2f}±{row.get('forgetting_percent_std', 0):.2f}"
        }
        table_rows.append(table_row)
    
    results_df = pd.DataFrame(table_rows)
    
    # Save to CSV
    csv_path = output_path / "aggregated_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Save to markdown
    md_path = output_path / "aggregated_results.md"
    with open(md_path, 'w') as f:
        f.write("# GiFT Experiments - Aggregated Results\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"Saved markdown to: {md_path}")
    
    # Print to console
    print("\n" + "="*60)
    print("AGGREGATED RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / "results" / "logs"
    output_dir = base_dir / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_experiment_results(log_dir)
    print(f"Found {len(results)} experiment runs")
    
    if not results:
        print("No results found!")
        return
    
    print("Aggregating results...")
    aggregated_df = aggregate_by_experiment(results)
    
    print("Creating results table...")
    create_results_table(aggregated_df, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
