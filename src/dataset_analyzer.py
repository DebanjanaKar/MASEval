"""
Dataset-Level Analysis Module

This module performs aggregate analysis across all traces in the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm

from .trace_parser import parse_agent_turns, detect_roles
from .graph_builder import build_communication_graph, extract_graph_features
from .metrics import compute_all_metrics, compute_coordination_stability
from .architecture_classifier import classify_architecture


def analyze_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform dataset-level analysis on all traces.
    
    Args:
        df: DataFrame with traces (from load_mast_dataset)
        
    Returns:
        DataFrame with analysis results for each trace
    """
    print("\n" + "="*60)
    print("DATASET-LEVEL ANALYSIS")
    print("="*60)
    
    results = []
    transition_matrices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing traces"):
        task_id = row["task_id"]
        trace = row["trace"]
        
        # Skip empty traces
        if not trace or len(trace) == 0:
            continue
        
        # Parse trace
        parsed = parse_agent_turns(trace)
        
        # Build graph
        graph_data = build_communication_graph(trace)
        
        # Extract features
        graph_features = extract_graph_features(graph_data["graph"])
        
        # Detect roles
        roles = detect_roles(trace)
        
        # Classify architecture
        architecture = classify_architecture(graph_features, roles)
        
        # Compute metrics
        metrics = compute_all_metrics(graph_data)
        
        # Store transition matrix for stability analysis
        if graph_data["transition_matrix"].size > 0:
            transition_matrices.append(graph_data["transition_matrix"])
        
        # Compile results
        result = {
            "task_id": task_id,
            "architecture_type": architecture,
            "num_agents": graph_features["number_of_agents"],
            "num_edges": graph_features["number_of_edges"],
            "num_turns": len(trace),
            "graph_density": graph_features["graph_density"],
            "avg_degree": graph_features["average_degree"],
            "clustering_coefficient": graph_features["clustering_coefficient"],
            "num_cycles": graph_features["number_of_cycles"],
            "loop_index": metrics["loop_index"],
            "agent_dependency_ratio": metrics["agent_dependency_ratio"],
            "communication_entropy": metrics["communication_entropy"],
            "longest_path_length": metrics["longest_path_length"],
            "avg_path_length": metrics["avg_path_length"],
        }
        
        # Add role flags
        for role, present in roles.items():
            result[f"has_{role}"] = present
        
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute coordination stability across all traces
    if len(transition_matrices) > 1:
        css = compute_coordination_stability(transition_matrices)
        print(f"\nCoordination Stability Score (CSS): {css:.4f}")
        print("(Lower values indicate more consistent coordination patterns)")
    
    # Print summary statistics
    print_dataset_summary(results_df)
    
    return results_df


def print_dataset_summary(results_df: pd.DataFrame):
    """
    Print summary statistics for the dataset analysis.
    
    Args:
        results_df: DataFrame with analysis results
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal traces analyzed: {len(results_df)}")
    
    # Architecture distribution
    print("\n--- Architecture Distribution ---")
    arch_counts = results_df["architecture_type"].value_counts()
    for arch, count in arch_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {arch}: {count} ({percentage:.1f}%)")
    
    # Numeric metrics summary
    print("\n--- Metric Statistics ---")
    metrics_to_summarize = [
        "num_agents",
        "num_turns",
        "graph_density",
        "loop_index",
        "agent_dependency_ratio",
        "communication_entropy"
    ]
    
    for metric in metrics_to_summarize:
        if metric in results_df.columns:
            values = results_df[metric]
            print(f"\n{metric}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std:  {values.std():.4f}")
            print(f"  Min:  {values.min():.4f}")
            print(f"  Max:  {values.max():.4f}")
    
    # Role presence
    print("\n--- Role Presence ---")
    role_columns = [col for col in results_df.columns if col.startswith("has_")]
    for col in role_columns:
        role_name = col.replace("has_", "")
        count = results_df[col].sum()
        percentage = (count / len(results_df)) * 100
        print(f"  {role_name}: {count} traces ({percentage:.1f}%)")


def compute_architecture_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute detailed statistics grouped by architecture type.
    
    Args:
        results_df: DataFrame with analysis results
        
    Returns:
        Dictionary with architecture-specific statistics
    """
    stats = {}
    
    for arch_type in results_df["architecture_type"].unique():
        arch_data = results_df[results_df["architecture_type"] == arch_type]
        
        stats[arch_type] = {
            "count": len(arch_data),
            "avg_num_agents": arch_data["num_agents"].mean(),
            "avg_density": arch_data["graph_density"].mean(),
            "avg_loop_index": arch_data["loop_index"].mean(),
            "avg_dependency_ratio": arch_data["agent_dependency_ratio"].mean(),
            "avg_entropy": arch_data["communication_entropy"].mean()
        }
    
    return stats


def identify_outliers(results_df: pd.DataFrame, metric: str, threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify outlier traces based on a specific metric.
    
    Args:
        results_df: DataFrame with analysis results
        metric: Metric column name
        threshold: Number of standard deviations for outlier detection
        
    Returns:
        DataFrame with outlier traces
    """
    if metric not in results_df.columns:
        return pd.DataFrame()
    
    values = results_df[metric]
    mean = values.mean()
    std = values.std()
    
    # Identify outliers
    outliers = results_df[
        (values > mean + threshold * std) | 
        (values < mean - threshold * std)
    ]
    
    return outliers


def compare_architectures(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare metrics across different architecture types.
    
    Args:
        results_df: DataFrame with analysis results
        
    Returns:
        DataFrame with comparison statistics
    """
    metrics_to_compare = [
        "num_agents",
        "graph_density",
        "loop_index",
        "agent_dependency_ratio",
        "communication_entropy"
    ]
    
    comparison = results_df.groupby("architecture_type")[metrics_to_compare].agg([
        "mean", "std", "min", "max"
    ])
    
    return comparison


def export_results(results_df: pd.DataFrame, output_path: str = "outputs/mast_analysis_results.csv"):
    """
    Export analysis results to CSV file.
    
    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save CSV file
    """
    results_df.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")

# Made with Bob
