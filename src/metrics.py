"""
Coordination Metrics Module

This module implements graph-based coordination metrics for MAS analysis.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any
from scipy.stats import entropy


def compute_loop_index(G: nx.DiGraph, edge_list: List[tuple]) -> float:
    """
    Compute Loop Index (LI) - measures interaction loops.
    
    LI = number_of_cycles / number_of_edges
    
    Args:
        G: NetworkX DiGraph
        edge_list: List of edges
        
    Returns:
        Loop index value (0 to infinity, higher means more loops)
    """
    if G.number_of_edges() == 0:
        return 0.0
    
    try:
        cycles = list(nx.simple_cycles(G))
        num_cycles = len(cycles)
    except:
        num_cycles = 0
    
    loop_index = num_cycles / G.number_of_edges()
    
    return loop_index


def compute_agent_dependency_ratio(G: nx.DiGraph) -> float:
    """
    Compute Agent Dependency Ratio (ADR).
    
    Measures reliance on a single agent.
    ADR = max_node_degree / total_edges
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Agent dependency ratio (0 to 1, higher means more centralized)
    """
    if G.number_of_edges() == 0:
        return 0.0
    
    # Get all degrees (in + out)
    degrees = dict(G.degree())
    
    if not degrees:
        return 0.0
    
    max_degree = max(degrees.values())
    total_edges = G.number_of_edges()
    
    adr = max_degree / (2 * total_edges)  # Divide by 2 since each edge contributes to 2 degrees
    
    return adr


def compute_communication_entropy(transition_matrix: np.ndarray) -> float:
    """
    Compute Communication Entropy (CE).
    
    Measures unpredictability of agent transitions.
    Uses Shannon entropy over transition probabilities.
    
    Args:
        transition_matrix: Row-normalized transition probability matrix
        
    Returns:
        Communication entropy value (higher means more unpredictable)
    """
    if transition_matrix.size == 0:
        return 0.0
    
    # Flatten the transition matrix and remove zeros
    probs = transition_matrix.flatten()
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    # Compute Shannon entropy
    ce = entropy(probs, base=2)
    
    return ce


def compute_coordination_stability(transition_matrices: List[np.ndarray]) -> float:
    """
    Compute Coordination Stability Score (CSS).
    
    Measures consistency of agent interaction patterns across traces.
    Lower variance indicates more stable coordination.
    
    Args:
        transition_matrices: List of transition matrices from different traces
        
    Returns:
        Coordination stability score (lower means more stable)
    """
    if not transition_matrices or len(transition_matrices) < 2:
        return 0.0
    
    # Find maximum dimension
    max_dim = max(mat.shape[0] for mat in transition_matrices if mat.size > 0)
    
    if max_dim == 0:
        return 0.0
    
    # Pad all matrices to same size
    padded_matrices = []
    for mat in transition_matrices:
        if mat.size == 0:
            padded = np.zeros((max_dim, max_dim))
        else:
            current_dim = mat.shape[0]
            if current_dim < max_dim:
                padded = np.zeros((max_dim, max_dim))
                padded[:current_dim, :current_dim] = mat
            else:
                padded = mat
        padded_matrices.append(padded)
    
    # Stack matrices
    stacked = np.stack(padded_matrices)
    
    # Compute variance across traces for each transition
    variances = np.var(stacked, axis=0)
    
    # Average variance as stability score
    css = np.mean(variances)
    
    return css


def compute_failure_propagation_risk(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute failure propagation indicators.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary with failure propagation metrics
    """
    metrics = {
        "longest_path_length": 0,
        "average_path_length": 0.0,
        "reachability_scores": {}
    }
    
    if G.number_of_nodes() == 0:
        return metrics
    
    # Longest path (for DAGs)
    if nx.is_directed_acyclic_graph(G):
        try:
            longest_path = nx.dag_longest_path(G)
            metrics["longest_path_length"] = len(longest_path) - 1
        except:
            pass
    
    # Average shortest path length (for weakly connected graphs)
    if nx.is_weakly_connected(G):
        try:
            G_undirected = G.to_undirected()
            metrics["average_path_length"] = nx.average_shortest_path_length(G_undirected)
        except:
            pass
    
    # Reachability from each node
    for node in G.nodes():
        reachable = len(nx.descendants(G, node))
        metrics["reachability_scores"][node] = reachable
    
    return metrics


def compute_all_metrics(graph_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all coordination metrics for a single trace.
    
    Args:
        graph_data: Dictionary from build_communication_graph
        
    Returns:
        Dictionary with all metric values
    """
    G = graph_data["graph"]
    edge_list = graph_data["edge_list"]
    transition_matrix = graph_data["transition_matrix"]
    
    metrics = {
        "loop_index": compute_loop_index(G, edge_list),
        "agent_dependency_ratio": compute_agent_dependency_ratio(G),
        "communication_entropy": compute_communication_entropy(transition_matrix)
    }
    
    # Add failure propagation metrics
    fp_metrics = compute_failure_propagation_risk(G)
    metrics["longest_path_length"] = fp_metrics["longest_path_length"]
    metrics["average_path_length"] = fp_metrics["average_path_length"]
    
    return metrics


def compute_interaction_balance(G: nx.DiGraph) -> float:
    """
    Compute interaction balance score.
    
    Measures how evenly distributed interactions are among agents.
    Uses coefficient of variation of edge weights.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Balance score (lower means more balanced)
    """
    if G.number_of_edges() == 0:
        return 0.0
    
    # Get edge weights
    weights = [data.get("weight", 1) for u, v, data in G.edges(data=True)]
    
    if not weights:
        return 0.0
    
    # Compute coefficient of variation
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    
    if mean_weight == 0:
        return 0.0
    
    cv = std_weight / mean_weight
    
    return cv


def compute_reciprocity_score(G: nx.DiGraph) -> float:
    """
    Compute reciprocity score.
    
    Measures the proportion of bidirectional edges.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Reciprocity score (0 to 1)
    """
    if G.number_of_edges() == 0:
        return 0.0
    
    try:
        reciprocity = nx.reciprocity(G)
        return reciprocity
    except:
        return 0.0


def compute_turn_taking_regularity(turns: List[Dict[str, Any]]) -> float:
    """
    Compute turn-taking regularity.
    
    Measures how regularly agents take turns (vs. one agent dominating).
    
    Args:
        turns: List of turn dictionaries
        
    Returns:
        Regularity score (higher means more regular turn-taking)
    """
    if not turns or len(turns) < 2:
        return 0.0
    
    # Count consecutive turns by same agent
    consecutive_counts = []
    current_count = 1
    
    sorted_turns = sorted(turns, key=lambda x: x.get("turn", 0))
    
    for i in range(1, len(sorted_turns)):
        if sorted_turns[i].get("agent") == sorted_turns[i-1].get("agent"):
            current_count += 1
        else:
            consecutive_counts.append(current_count)
            current_count = 1
    consecutive_counts.append(current_count)
    
    # Lower variance in consecutive counts means more regular
    if len(consecutive_counts) == 0:
        return 0.0
    
    mean_consecutive = np.mean(consecutive_counts)
    std_consecutive = np.std(consecutive_counts)
    
    if mean_consecutive == 0:
        return 0.0
    
    # Inverse of coefficient of variation (higher is more regular)
    regularity = 1.0 / (1.0 + std_consecutive / mean_consecutive)
    
    return regularity

# Made with Bob
