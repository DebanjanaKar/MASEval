"""
MAS Analysis Pipeline

A research pipeline for analyzing Multi-Agent System interaction traces.
"""

from .data_loader import load_mast_dataset, get_trace_by_id
from .trace_parser import parse_agent_turns, detect_roles
from .graph_builder import build_communication_graph, extract_graph_features
from .metrics import compute_all_metrics, compute_coordination_stability
from .architecture_classifier import classify_architecture
from .dataset_analyzer import analyze_dataset, export_results
from .single_trace_analyzer import analyze_single_trace, export_trace_analysis
from .visualizer import visualize_dataset_analysis, visualize_single_trace

__version__ = "1.0.0"

__all__ = [
    "load_mast_dataset",
    "get_trace_by_id",
    "parse_agent_turns",
    "detect_roles",
    "build_communication_graph",
    "extract_graph_features",
    "compute_all_metrics",
    "compute_coordination_stability",
    "classify_architecture",
    "analyze_dataset",
    "export_results",
    "analyze_single_trace",
    "export_trace_analysis",
    "visualize_dataset_analysis",
    "visualize_single_trace",
]

# Made with Bob
