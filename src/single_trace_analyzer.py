"""
Single Trace Analysis Module

This module provides detailed analysis for individual traces.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from .trace_parser import parse_agent_turns, detect_roles, extract_agent_statistics
from .graph_builder import build_communication_graph, extract_graph_features
from .metrics import compute_all_metrics
from .architecture_classifier import (
    classify_architecture,
    analyze_architecture_properties,
    identify_bottlenecks
)


def analyze_single_trace(df: pd.DataFrame, task_id: int) -> Dict[str, Any]:
    """
    Perform detailed analysis on a single trace.
    
    Args:
        df: DataFrame with all traces
        task_id: ID of the trace to analyze
        
    Returns:
        Dictionary with comprehensive trace analysis
    """
    if task_id >= len(df):
        raise ValueError(f"task_id {task_id} not found. Dataset has {len(df)} traces.")
    
    print("\n" + "="*60)
    print(f"SINGLE TRACE ANALYSIS - Task ID: {task_id}")
    print("="*60)
    
    # Get trace
    trace = df.iloc[task_id]["trace"]
    
    if not trace or len(trace) == 0:
        print("Warning: Empty trace")
        return {}
    
    # Parse trace
    parsed = parse_agent_turns(trace)
    turns = parsed["turns"]
    agents = parsed["agents"]
    transitions = parsed["transitions"]
    
    # Build communication graph
    graph_data = build_communication_graph(trace)
    G = graph_data["graph"]
    
    # Extract graph features
    graph_features = extract_graph_features(G)
    
    # Detect roles
    roles = detect_roles(trace)
    
    # Classify architecture
    architecture = classify_architecture(graph_features, roles)
    
    # Analyze architecture properties
    arch_analysis = analyze_architecture_properties(graph_features, roles, architecture)
    
    # Compute metrics
    metrics = compute_all_metrics(graph_data)
    
    # Extract agent statistics
    agent_stats = extract_agent_statistics(turns)
    
    # Identify bottlenecks
    bottlenecks = identify_bottlenecks(graph_features)
    
    # Compile comprehensive analysis
    analysis = {
        "task_id": task_id,
        "trace_info": {
            "num_turns": len(turns),
            "num_agents": len(agents),
            "agents": agents,
            "transitions": transitions
        },
        "turns": turns,
        "graph_data": graph_data,
        "graph_features": graph_features,
        "roles": roles,
        "architecture": arch_analysis,
        "metrics": metrics,
        "agent_statistics": agent_stats,
        "bottlenecks": bottlenecks
    }
    
    # Print analysis
    print_trace_analysis(analysis)
    
    return analysis


def print_trace_analysis(analysis: Dict[str, Any]):
    """
    Print formatted trace analysis to console.
    
    Args:
        analysis: Dictionary with trace analysis results
    """
    trace_info = analysis["trace_info"]
    graph_features = analysis["graph_features"]
    roles = analysis["roles"]
    arch_analysis = analysis["architecture"]
    metrics = analysis["metrics"]
    agent_stats = analysis["agent_statistics"]
    bottlenecks = analysis["bottlenecks"]
    
    # Basic information
    print("\n--- Trace Information ---")
    print(f"Number of turns: {trace_info['num_turns']}")
    print(f"Number of agents: {trace_info['num_agents']}")
    print(f"Agents: {', '.join(trace_info['agents'])}")
    
    # Architecture
    print("\n--- Architecture Analysis ---")
    print(f"Type: {arch_analysis['architecture_type']}")
    print(f"Description: {arch_analysis['description']}")
    
    if "centralization" in arch_analysis["properties"]:
        cent = arch_analysis["properties"]["centralization"]
        print(f"Centralization (max): {cent['max']:.3f}")
        print(f"Centralization (avg): {cent['avg']:.3f}")
    
    if arch_analysis["properties"].get("most_central_agent"):
        print(f"Most central agent: {arch_analysis['properties']['most_central_agent']}")
    
    # Roles
    print("\n--- Detected Roles ---")
    active_roles = [role for role, present in roles.items() if present]
    if active_roles:
        print(f"Active roles: {', '.join(active_roles)}")
    else:
        print("No specific roles detected")
    
    # Graph features
    print("\n--- Graph Structure ---")
    print(f"Number of edges: {graph_features['number_of_edges']}")
    print(f"Graph density: {graph_features['graph_density']:.3f}")
    print(f"Average degree: {graph_features['average_degree']:.3f}")
    print(f"Clustering coefficient: {graph_features['clustering_coefficient']:.3f}")
    print(f"Number of cycles: {graph_features['number_of_cycles']}")
    
    # Coordination metrics
    print("\n--- Coordination Metrics ---")
    print(f"Loop Index (LI): {metrics['loop_index']:.4f}")
    print(f"Agent Dependency Ratio (ADR): {metrics['agent_dependency_ratio']:.4f}")
    print(f"Communication Entropy (CE): {metrics['communication_entropy']:.4f}")
    
    if metrics.get('longest_path_length', 0) > 0:
        print(f"Longest path length: {metrics['longest_path_length']}")
    if metrics.get('avg_path_length', 0) > 0:
        print(f"Average path length: {metrics['avg_path_length']:.3f}")
    
    # Agent statistics
    print("\n--- Agent Participation ---")
    if agent_stats:
        print(f"Most active agent: {agent_stats['most_active_agent']} ({agent_stats['most_active_agent_turns']} turns)")
        print("\nTurn counts per agent:")
        for agent, count in sorted(agent_stats['agent_turn_counts'].items(), key=lambda x: x[1], reverse=True):
            ratio = agent_stats['participation_ratios'][agent]
            print(f"  {agent}: {count} turns ({ratio*100:.1f}%)")
    
    # Bottlenecks
    print("\n--- Bottleneck Analysis ---")
    if bottlenecks["has_bottleneck"]:
        print(f"Bottleneck detected!")
        print(f"Bottleneck agents: {', '.join(bottlenecks['bottleneck_agents'])}")
        print(f"Bottleneck score: {bottlenecks['bottleneck_score']:.3f}")
    else:
        print("No significant bottlenecks detected")
    
    # Transition sequence (first 10)
    print("\n--- Agent Transition Sequence (first 10) ---")
    transitions = trace_info['transitions'][:10]
    for i, transition in enumerate(transitions, 1):
        print(f"  {i}. {transition}")
    if len(trace_info['transitions']) > 10:
        print(f"  ... and {len(trace_info['transitions']) - 10} more transitions")


def get_turn_by_turn_breakdown(analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a turn-by-turn breakdown DataFrame.
    
    Args:
        analysis: Dictionary with trace analysis results
        
    Returns:
        DataFrame with turn-by-turn information
    """
    turns = analysis["turns"]
    
    breakdown = []
    for turn in turns:
        breakdown.append({
            "turn": turn.get("turn", 0),
            "agent": turn.get("agent", "unknown"),
            "content_length": len(turn.get("content", "")),
            "content_preview": turn.get("content", "")[:100] + "..." if len(turn.get("content", "")) > 100 else turn.get("content", "")
        })
    
    return pd.DataFrame(breakdown)


def export_trace_analysis(analysis: Dict[str, Any], output_dir: str = "outputs"):
    """
    Export single trace analysis to files.
    
    Args:
        analysis: Dictionary with trace analysis results
        output_dir: Directory to save output files
    """
    import os
    import json
    
    task_id = analysis["task_id"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export turn-by-turn breakdown
    breakdown_df = get_turn_by_turn_breakdown(analysis)
    breakdown_path = os.path.join(output_dir, f"trace_{task_id}_breakdown.csv")
    breakdown_df.to_csv(breakdown_path, index=False)
    
    # Export summary as JSON (excluding non-serializable objects)
    summary = {
        "task_id": task_id,
        "trace_info": analysis["trace_info"],
        "roles": analysis["roles"],
        "architecture": {
            "type": analysis["architecture"]["architecture_type"],
            "description": analysis["architecture"]["description"],
            "properties": {
                k: v for k, v in analysis["architecture"]["properties"].items()
                if not isinstance(v, (dict, list)) or k in ["active_roles"]
            }
        },
        "metrics": analysis["metrics"],
        "agent_statistics": analysis["agent_statistics"],
        "bottlenecks": analysis["bottlenecks"]
    }
    
    summary_path = os.path.join(output_dir, f"trace_{task_id}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTrace analysis exported:")
    print(f"  - Breakdown: {breakdown_path}")
    print(f"  - Summary: {summary_path}")

# Made with Bob
