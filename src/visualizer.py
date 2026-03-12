"""
Visualization Module

This module creates visualizations for both dataset-level and single-trace analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def visualize_dataset_analysis(results_df: pd.DataFrame, output_dir: str = "visualizations"):
    """
    Create visualizations for dataset-level analysis.
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING DATASET VISUALIZATIONS")
    print("="*60)
    
    # 1. Architecture distribution
    plot_architecture_distribution(results_df, output_dir)
    
    # 2. Loop Index distribution
    plot_metric_distribution(results_df, "loop_index", "Loop Index Distribution", output_dir)
    
    # 3. Agent Dependency Ratio distribution
    plot_metric_distribution(results_df, "agent_dependency_ratio", "Agent Dependency Ratio Distribution", output_dir)
    
    # 4. Communication Entropy distribution
    plot_metric_distribution(results_df, "communication_entropy", "Communication Entropy Distribution", output_dir)
    
    # 5. Agent Dependency Ratio vs Architecture
    plot_metric_by_architecture(results_df, "agent_dependency_ratio", "Agent Dependency Ratio", output_dir)
    
    # 6. Graph density vs architecture
    plot_density_by_architecture(results_df, output_dir)
    
    # 7. Correlation heatmap
    plot_correlation_heatmap(results_df, output_dir)
    
    # 8. Number of agents distribution
    plot_metric_distribution(results_df, "num_agents", "Number of Agents Distribution", output_dir)
    
    print(f"\nVisualizations saved to: {output_dir}/")


def plot_architecture_distribution(results_df: pd.DataFrame, output_dir: str):
    """Plot architecture type distribution."""
    plt.figure(figsize=(12, 6))
    
    arch_counts = results_df["architecture_type"].value_counts()
    
    colors = sns.color_palette("husl", len(arch_counts))
    bars = plt.bar(range(len(arch_counts)), arch_counts.values, color=colors)
    plt.xticks(range(len(arch_counts)), arch_counts.index, rotation=45, ha='right')
    plt.ylabel("Number of Traces")
    plt.title("MAS Architecture Distribution", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "architecture_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Architecture distribution plot saved")


def plot_metric_distribution(results_df: pd.DataFrame, metric: str, title: str, output_dir: str):
    """Plot distribution of a metric."""
    if metric not in results_df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    data = results_df[metric].dropna()
    
    # Histogram with KDE
    plt.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add KDE if enough data points
    if len(data) > 5:
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        plt.plot(x_range, kde(x_range) * len(data) * (data.max() - data.min()) / 30, 
                'r-', linewidth=2, label='KDE')
        plt.legend()
    
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel("Frequency")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    plt.legend()
    
    plt.tight_layout()
    filename = f"{metric}_distribution.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {title} plot saved")


def plot_metric_by_architecture(results_df: pd.DataFrame, metric: str, metric_label: str, output_dir: str):
    """Plot metric values grouped by architecture type."""
    if metric not in results_df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Box plot
    arch_types = results_df["architecture_type"].unique()
    data_by_arch = [results_df[results_df["architecture_type"] == arch][metric].dropna() 
                    for arch in arch_types]
    
    bp = plt.boxplot(data_by_arch, labels=arch_types, patch_artist=True)
    
    # Color boxes
    colors = sns.color_palette("husl", len(arch_types))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} by Architecture Type", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{metric}_by_architecture.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {metric_label} by architecture plot saved")


def plot_density_by_architecture(results_df: pd.DataFrame, output_dir: str):
    """Plot graph density vs architecture as scatter plot."""
    plt.figure(figsize=(12, 6))
    
    arch_types = results_df["architecture_type"].unique()
    colors = sns.color_palette("husl", len(arch_types))
    
    for arch, color in zip(arch_types, colors):
        data = results_df[results_df["architecture_type"] == arch]
        plt.scatter(data["num_agents"], data["graph_density"], 
                   label=arch, alpha=0.6, s=100, color=color)
    
    plt.xlabel("Number of Agents")
    plt.ylabel("Graph Density")
    plt.title("Graph Density vs Number of Agents by Architecture", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "density_vs_agents_by_architecture.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Density vs agents plot saved")


def plot_correlation_heatmap(results_df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap of numeric metrics."""
    numeric_cols = [
        "num_agents", "num_edges", "graph_density", "avg_degree",
        "clustering_coefficient", "num_cycles", "loop_index",
        "agent_dependency_ratio", "communication_entropy"
    ]
    
    # Filter to existing columns
    numeric_cols = [col for col in numeric_cols if col in results_df.columns]
    
    if len(numeric_cols) < 2:
        return
    
    plt.figure(figsize=(12, 10))
    
    corr_matrix = results_df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title("Correlation Heatmap of Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Correlation heatmap saved")


def visualize_single_trace(analysis: Dict[str, Any], output_dir: str = "visualizations"):
    """
    Create visualizations for single trace analysis.
    
    Args:
        analysis: Dictionary with trace analysis results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    task_id = analysis["task_id"]
    
    print("\n" + "="*60)
    print(f"GENERATING TRACE {task_id} VISUALIZATIONS")
    print("="*60)
    
    # 1. Communication graph
    plot_communication_graph(analysis, output_dir)
    
    # 2. Adjacency matrix heatmap
    plot_adjacency_matrix(analysis, output_dir)
    
    # 3. Agent timeline
    plot_agent_timeline(analysis, output_dir)
    
    # 4. Agent participation
    plot_agent_participation(analysis, output_dir)
    
    print(f"\nTrace visualizations saved to: {output_dir}/")


def plot_communication_graph(analysis: Dict[str, Any], output_dir: str):
    """Plot the communication graph with NetworkX."""
    G = analysis["graph_data"]["graph"]
    task_id = analysis["task_id"]
    
    if G.number_of_nodes() == 0:
        print("  ⚠ Empty graph, skipping visualization")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    node_sizes = [G.degree(node) * 500 + 500 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.9, 
                          edgecolors='black', linewidths=2)
    
    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    # Normalize weights for visualization
    edge_widths = [3 * (w / max_weight) for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', 
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{G[u][v].get('weight', 1)}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(f"Communication Graph - Trace {task_id}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    filename = f"trace_{task_id}_communication_graph.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Communication graph saved")


def plot_adjacency_matrix(analysis: Dict[str, Any], output_dir: str):
    """Plot adjacency matrix as heatmap."""
    adjacency_matrix = analysis["graph_data"]["adjacency_matrix"]
    agents = analysis["graph_data"]["agents"]
    task_id = analysis["task_id"]
    
    if adjacency_matrix.size == 0:
        print("  ⚠ Empty adjacency matrix, skipping visualization")
        return
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(adjacency_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=agents, yticklabels=agents,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title(f"Adjacency Matrix - Trace {task_id}", fontsize=14, fontweight='bold')
    plt.xlabel("To Agent")
    plt.ylabel("From Agent")
    plt.tight_layout()
    
    filename = f"trace_{task_id}_adjacency_matrix.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Adjacency matrix saved")


def plot_agent_timeline(analysis: Dict[str, Any], output_dir: str):
    """Plot agent activity timeline."""
    turns = analysis["turns"]
    task_id = analysis["task_id"]
    
    if not turns:
        return
    
    # Extract data
    turn_numbers = [t.get("turn", 0) for t in turns]
    agents = [t.get("agent", "unknown") for t in turns]
    
    # Create agent to number mapping
    unique_agents = sorted(set(agents))
    agent_to_num = {agent: i for i, agent in enumerate(unique_agents)}
    agent_nums = [agent_to_num[agent] for agent in agents]
    
    plt.figure(figsize=(14, 6))
    
    # Plot timeline
    colors = sns.color_palette("husl", len(unique_agents))
    for i, agent in enumerate(unique_agents):
        agent_turns = [t for t, a in zip(turn_numbers, agents) if a == agent]
        agent_y = [i] * len(agent_turns)
        plt.scatter(agent_turns, agent_y, s=100, color=colors[i], 
                   label=agent, alpha=0.7, edgecolors='black', linewidths=1)
    
    plt.yticks(range(len(unique_agents)), unique_agents)
    plt.xlabel("Turn Number")
    plt.ylabel("Agent")
    plt.title(f"Agent Activity Timeline - Trace {task_id}", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filename = f"trace_{task_id}_agent_timeline.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Agent timeline saved")


def plot_agent_participation(analysis: Dict[str, Any], output_dir: str):
    """Plot agent participation statistics."""
    agent_stats = analysis["agent_statistics"]
    task_id = analysis["task_id"]
    
    if not agent_stats or "agent_turn_counts" not in agent_stats:
        return
    
    turn_counts = agent_stats["agent_turn_counts"]
    
    plt.figure(figsize=(10, 6))
    
    agents = list(turn_counts.keys())
    counts = list(turn_counts.values())
    
    colors = sns.color_palette("husl", len(agents))
    bars = plt.bar(agents, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    plt.xlabel("Agent")
    plt.ylabel("Number of Turns")
    plt.title(f"Agent Participation - Trace {task_id}", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    filename = f"trace_{task_id}_agent_participation.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Agent participation plot saved")

# Made with Bob
