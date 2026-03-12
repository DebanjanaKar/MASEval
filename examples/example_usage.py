"""
Example Usage of MAS Analysis Pipeline

This script demonstrates how to use the pipeline programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import load_mast_dataset
from src.dataset_analyzer import analyze_dataset, export_results
from src.single_trace_analyzer import analyze_single_trace
from src.visualizer import visualize_dataset_analysis, visualize_single_trace


def example_dataset_analysis():
    """Example: Run dataset-level analysis."""
    print("="*70)
    print("EXAMPLE 1: Dataset-Level Analysis")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = load_mast_dataset()
    print(f"   Loaded {len(df)} traces")
    
    # Analyze dataset
    print("\n2. Analyzing dataset...")
    results_df = analyze_dataset(df)
    
    # Export results
    print("\n3. Exporting results...")
    export_results(results_df, "outputs/example_results.csv")
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    visualize_dataset_analysis(results_df, "visualizations")
    
    print("\n✓ Dataset analysis complete!")
    return results_df


def example_single_trace_analysis(task_id=0):
    """Example: Analyze a single trace."""
    print("\n" + "="*70)
    print(f"EXAMPLE 2: Single-Trace Analysis (Task ID: {task_id})")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = load_mast_dataset()
    
    # Analyze single trace
    print(f"\n2. Analyzing trace {task_id}...")
    analysis = analyze_single_trace(df, task_id)
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    visualize_single_trace(analysis, "visualizations")
    
    print(f"\n✓ Trace {task_id} analysis complete!")
    return analysis


def example_custom_analysis():
    """Example: Custom analysis using pipeline components."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Analysis")
    print("="*70)
    
    from src.trace_parser import parse_agent_turns, detect_roles
    from src.graph_builder import build_communication_graph, extract_graph_features
    from src.metrics import compute_all_metrics
    from src.architecture_classifier import classify_architecture
    
    # Load dataset
    df = load_mast_dataset()
    
    # Get a specific trace
    trace = df.iloc[0]["trace"]
    
    print("\n1. Parsing trace...")
    parsed = parse_agent_turns(trace)
    print(f"   Agents: {parsed['agents']}")
    print(f"   Transitions: {len(parsed['transitions'])}")
    
    print("\n2. Building communication graph...")
    graph_data = build_communication_graph(trace)
    print(f"   Nodes: {graph_data['graph'].number_of_nodes()}")
    print(f"   Edges: {graph_data['graph'].number_of_edges()}")
    
    print("\n3. Extracting graph features...")
    features = extract_graph_features(graph_data["graph"])
    print(f"   Density: {features['graph_density']:.3f}")
    print(f"   Avg Degree: {features['average_degree']:.3f}")
    
    print("\n4. Computing metrics...")
    metrics = compute_all_metrics(graph_data)
    print(f"   Loop Index: {metrics['loop_index']:.4f}")
    print(f"   ADR: {metrics['agent_dependency_ratio']:.4f}")
    print(f"   CE: {metrics['communication_entropy']:.4f}")
    
    print("\n5. Detecting roles and architecture...")
    roles = detect_roles(trace)
    architecture = classify_architecture(features, roles)
    print(f"   Architecture: {architecture}")
    print(f"   Active roles: {[r for r, p in roles.items() if p]}")
    
    print("\n✓ Custom analysis complete!")


def example_batch_analysis():
    """Example: Analyze multiple specific traces."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Analysis")
    print("="*70)
    
    # Load dataset
    df = load_mast_dataset()
    
    # Analyze first 5 traces
    trace_ids = range(min(5, len(df)))
    
    results = []
    for task_id in trace_ids:
        print(f"\nAnalyzing trace {task_id}...")
        analysis = analyze_single_trace(df, task_id)
        
        # Extract key metrics
        results.append({
            "task_id": task_id,
            "architecture": analysis["architecture"]["architecture_type"],
            "num_agents": len(analysis["trace_info"]["agents"]),
            "loop_index": analysis["metrics"]["loop_index"]
        })
    
    # Print summary
    print("\n" + "-"*70)
    print("BATCH ANALYSIS SUMMARY")
    print("-"*70)
    for r in results:
        print(f"Trace {r['task_id']}: {r['architecture']} "
              f"({r['num_agents']} agents, LI={r['loop_index']:.3f})")
    
    print("\n✓ Batch analysis complete!")


def example_filtering_and_comparison():
    """Example: Filter traces and compare architectures."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Filtering and Comparison")
    print("="*70)
    
    import pandas as pd
    
    # Load and analyze dataset
    df = load_mast_dataset()
    results_df = analyze_dataset(df)
    
    print("\n1. Filtering traces...")
    
    # Find traces with high loop index
    high_loop = results_df[results_df["loop_index"] > 0.3]
    print(f"   Traces with high loop index (>0.3): {len(high_loop)}")
    
    # Find collaborative architectures
    collaborative = results_df[results_df["architecture_type"] == "collaborative"]
    print(f"   Collaborative architectures: {len(collaborative)}")
    
    print("\n2. Comparing architectures...")
    
    # Compare metrics by architecture
    comparison = results_df.groupby("architecture_type").agg({
        "loop_index": "mean",
        "agent_dependency_ratio": "mean",
        "communication_entropy": "mean",
        "num_agents": "mean"
    })
    
    print("\n   Average metrics by architecture:")
    print(comparison.to_string())
    
    print("\n✓ Filtering and comparison complete!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MAS ANALYSIS PIPELINE - EXAMPLE USAGE")
    print("="*70)
    
    # Run examples
    try:
        # Example 1: Dataset analysis
        results_df = example_dataset_analysis()
        
        # Example 2: Single trace analysis
        analysis = example_single_trace_analysis(task_id=0)
        
        # Example 3: Custom analysis
        example_custom_analysis()
        
        # Example 4: Batch analysis
        example_batch_analysis()
        
        # Example 5: Filtering and comparison
        example_filtering_and_comparison()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
