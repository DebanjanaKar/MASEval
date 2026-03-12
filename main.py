"""
Main Pipeline Script for MAS Analysis

This script orchestrates the entire analysis pipeline for the MAST dataset.
It supports both dataset-level analysis and single-trace analysis modes.

Usage:
    # Dataset-level analysis
    python main.py
    
    # Single-trace analysis
    python main.py --task_id 5
    
    # Dataset analysis with custom output
    python main.py --output_dir my_results
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_mast_dataset
from src.dataset_analyzer import analyze_dataset, export_results
from src.single_trace_analyzer import analyze_single_trace, export_trace_analysis
from src.visualizer import visualize_dataset_analysis, visualize_single_trace


def setup_directories(output_dir: str = "outputs", viz_dir: str = "visualizations"):
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Visualization directory: {viz_dir}")


def run_dataset_analysis(output_dir: str = "outputs", viz_dir: str = "visualizations"):
    """
    Run complete dataset-level analysis.
    
    Args:
        output_dir: Directory for output files
        viz_dir: Directory for visualizations
    """
    print("\n" + "="*70)
    print(" "*15 + "MAS ANALYSIS PIPELINE - DATASET MODE")
    print("="*70)
    
    # Load dataset
    print("\n[1/4] Loading MAST dataset...")
    df = load_mast_dataset()
    
    # Analyze dataset
    print("\n[2/4] Analyzing all traces...")
    results_df = analyze_dataset(df)
    
    # Export results
    print("\n[3/4] Exporting results...")
    output_path = os.path.join(output_dir, "mast_analysis_results.csv")
    export_results(results_df, output_path)
    
    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    visualize_dataset_analysis(results_df, viz_dir)
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_path}")
    print(f"Visualizations saved to: {viz_dir}/")
    
    return results_df


def run_single_trace_analysis(
    task_id: int,
    output_dir: str = "outputs",
    viz_dir: str = "visualizations"
):
    """
    Run detailed analysis on a single trace.
    
    Args:
        task_id: ID of the trace to analyze
        output_dir: Directory for output files
        viz_dir: Directory for visualizations
    """
    print("\n" + "="*70)
    print(" "*10 + f"MAS ANALYSIS PIPELINE - SINGLE TRACE MODE (ID: {task_id})")
    print("="*70)
    
    # Load dataset
    print("\n[1/4] Loading MAST dataset...")
    df = load_mast_dataset()
    
    # Validate task_id
    if task_id >= len(df):
        print(f"\nError: task_id {task_id} not found. Dataset has {len(df)} traces.")
        print(f"Valid task_id range: 0 to {len(df) - 1}")
        return None
    
    # Analyze single trace
    print(f"\n[2/4] Analyzing trace {task_id}...")
    analysis = analyze_single_trace(df, task_id)
    
    # Export trace analysis
    print(f"\n[3/4] Exporting trace analysis...")
    export_trace_analysis(analysis, output_dir)
    
    # Generate visualizations
    print(f"\n[4/4] Generating visualizations...")
    visualize_single_trace(analysis, viz_dir)
    
    print("\n" + "="*70)
    print(f"TRACE {task_id} ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Visualizations saved to: {viz_dir}/")
    
    return analysis


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="MAS Analysis Pipeline for MAST Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dataset-level analysis
  python main.py
  
  # Analyze a specific trace
  python main.py --task_id 5
  
  # Custom output directories
  python main.py --output_dir results --viz_dir plots
  
  # Analyze trace with custom directories
  python main.py --task_id 10 --output_dir trace_results --viz_dir trace_plots
        """
    )
    
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Task ID for single-trace analysis (omit for dataset-level analysis)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for output files (default: outputs)"
    )
    
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="visualizations",
        help="Directory for visualizations (default: visualizations)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories(args.output_dir, args.viz_dir)
    
    try:
        if args.task_id is not None:
            # Single-trace analysis mode
            run_single_trace_analysis(args.task_id, args.output_dir, args.viz_dir)
        else:
            # Dataset-level analysis mode
            run_dataset_analysis(args.output_dir, args.viz_dir)
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
