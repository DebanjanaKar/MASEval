"""
Communication Graph Builder Module

This module builds directed communication graphs from agent interaction traces.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def build_communication_graph(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a directed communication graph from agent turns.
    
    Nodes represent agents, edges represent sequential interactions.
    Edge weights represent the number of times one agent follows another.
    
    Args:
        turns: List of turn dictionaries with 'agent' field
        
    Returns:
        Dictionary containing:
        - graph: NetworkX DiGraph object
        - edge_list: list of (source, target, weight) tuples
        - transition_matrix: numpy array of transition probabilities
        - adjacency_matrix: numpy array of edge weights
    """
    # Create directed graph
    G = nx.DiGraph()
    
    if not turns or len(turns) == 0:
        return {
            "graph": G,
            "edge_list": [],
            "transition_matrix": np.array([]),
            "adjacency_matrix": np.array([])
        }
    
    # Sort turns by turn number
    sorted_turns = sorted(turns, key=lambda x: x.get("turn", 0))
    
    # Extract agents
    agents = [turn.get("agent", "unknown") for turn in sorted_turns]
    unique_agents = sorted(set(agents))
    
    # Add nodes
    for agent in unique_agents:
        G.add_node(agent)
    
    # Count transitions
    edge_weights = defaultdict(int)
    
    for i in range(len(agents) - 1):
        source = agents[i]
        target = agents[i + 1]
        edge_weights[(source, target)] += 1
    
    # Add edges with weights
    for (source, target), weight in edge_weights.items():
        G.add_edge(source, target, weight=weight)
    
    # Create edge list
    edge_list = [(source, target, weight) for (source, target), weight in edge_weights.items()]
    
    # Build adjacency matrix
    n = len(unique_agents)
    adjacency_matrix = np.zeros((n, n))
    agent_to_idx = {agent: idx for idx, agent in enumerate(unique_agents)}
    
    for (source, target), weight in edge_weights.items():
        i = agent_to_idx[source]
        j = agent_to_idx[target]
        adjacency_matrix[i, j] = weight
    
    # Build transition matrix (row-normalized adjacency matrix)
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        row_sum = adjacency_matrix[i, :].sum()
        if row_sum > 0:
            transition_matrix[i, :] = adjacency_matrix[i, :] / row_sum
    
    return {
        "graph": G,
        "edge_list": edge_list,
        "transition_matrix": transition_matrix,
        "adjacency_matrix": adjacency_matrix,
        "agent_to_idx": agent_to_idx,
        "agents": unique_agents
    }


def extract_graph_features(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Extract structural features from the communication graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary with graph features
    """
    features = {}
    
    # Basic metrics
    features["number_of_agents"] = G.number_of_nodes()
    features["number_of_edges"] = G.number_of_edges()
    
    # Handle empty graph
    if G.number_of_nodes() == 0:
        features["graph_density"] = 0.0
        features["average_degree"] = 0.0
        features["degree_centrality"] = {}
        features["betweenness_centrality"] = {}
        features["clustering_coefficient"] = 0.0
        features["number_of_cycles"] = 0
        return features
    
    # Graph density
    features["graph_density"] = nx.density(G)
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    features["average_degree"] = np.mean(degrees) if degrees else 0.0
    
    # Degree centrality
    features["degree_centrality"] = nx.degree_centrality(G)
    
    # Betweenness centrality
    features["betweenness_centrality"] = nx.betweenness_centrality(G)
    
    # Clustering coefficient (for undirected version)
    G_undirected = G.to_undirected()
    features["clustering_coefficient"] = nx.average_clustering(G_undirected)
    
    # Count cycles
    try:
        cycles = list(nx.simple_cycles(G))
        features["number_of_cycles"] = len(cycles)
        features["cycles"] = cycles
    except:
        features["number_of_cycles"] = 0
        features["cycles"] = []
    
    # Additional metrics
    features["is_strongly_connected"] = nx.is_strongly_connected(G)
    features["is_weakly_connected"] = nx.is_weakly_connected(G)
    
    # Number of strongly connected components
    features["number_of_strongly_connected_components"] = nx.number_strongly_connected_components(G)
    
    # In-degree and out-degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    features["max_in_degree"] = max(in_degrees) if in_degrees else 0
    features["max_out_degree"] = max(out_degrees) if out_degrees else 0
    features["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0.0
    features["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0.0
    
    return features


def get_node_importance_ranking(G: nx.DiGraph) -> List[Tuple[str, float]]:
    """
    Rank nodes by importance using multiple centrality measures.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of (node, importance_score) tuples sorted by importance
    """
    if G.number_of_nodes() == 0:
        return []
    
    # Calculate multiple centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Combine scores (weighted average)
    importance_scores = {}
    for node in G.nodes():
        importance_scores[node] = (
            0.5 * degree_cent.get(node, 0) +
            0.5 * betweenness_cent.get(node, 0)
        )
    
    # Sort by importance
    ranked = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked


def detect_hub_nodes(G: nx.DiGraph, threshold: float = 0.5) -> List[str]:
    """
    Detect hub nodes (highly connected agents).
    
    Args:
        G: NetworkX DiGraph
        threshold: Centrality threshold for hub detection
        
    Returns:
        List of hub node names
    """
    if G.number_of_nodes() == 0:
        return []
    
    degree_cent = nx.degree_centrality(G)
    hubs = [node for node, cent in degree_cent.items() if cent >= threshold]
    
    return hubs


def find_critical_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Find critical paths in the communication graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary with path information
    """
    paths_info = {
        "longest_path_length": 0,
        "longest_path": [],
        "all_simple_paths": []
    }
    
    if G.number_of_nodes() == 0:
        return paths_info
    
    # Find longest path using DAG longest path if graph is acyclic
    if nx.is_directed_acyclic_graph(G):
        try:
            longest_path = nx.dag_longest_path(G)
            paths_info["longest_path"] = longest_path
            paths_info["longest_path_length"] = len(longest_path) - 1
        except:
            pass
    
    return paths_info


def compute_graph_diameter(G: nx.DiGraph) -> int:
    """
    Compute the diameter of the graph (longest shortest path).
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Diameter value or -1 if not connected
    """
    if G.number_of_nodes() == 0:
        return -1
    
    if not nx.is_weakly_connected(G):
        return -1
    
    try:
        # Convert to undirected for diameter calculation
        G_undirected = G.to_undirected()
        diameter = nx.diameter(G_undirected)
        return diameter
    except:
        return -1

# Made with Bob
