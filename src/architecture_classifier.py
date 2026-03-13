"""
MAS Architecture Classification Module

This module classifies Multi-Agent System architectures based on
graph topology and role detection.
"""

from typing import Dict, Any
import networkx as nx


def classify_architecture(graph_features: Dict[str, Any], roles: Dict[str, bool]) -> str:
    """
    Classify the MAS architecture using role signals and graph topology.
    
    Architecture types:
    - hierarchical: Manager present with high centrality
    - planner_executor: Planner present with worker/executor
    - collaborative: High graph density, distributed interactions
    - environment_interactive: Tool/environment nodes present
    - sequential: Linear chain of agents
    - unknown: Cannot determine architecture
    
    Args:
        graph_features: Dictionary of graph structural features
        roles: Dictionary of detected roles
        
    Returns:
        Architecture type string
    """
    # Extract key features
    num_agents = graph_features.get("number_of_agents", 0)
    density = graph_features.get("graph_density", 0.0)
    num_cycles = graph_features.get("number_of_cycles", 0)
    
    # Get centrality information
    degree_centrality = graph_features.get("degree_centrality", {})
    max_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
    
    # Classification logic
    
    # Environment-interactive: Tool or environment agents present
    if roles.get("tool", False) or roles.get("environment", False):
        return "environment_interactive"
    
    # Hierarchical: Manager present with high centralization
    elif roles.get("manager", False) and max_centrality > 0.6:
        return "hierarchical"
    
    # Planner-Executor: Planner with worker/executor
    elif roles.get("planner", False) and (roles.get("worker", False) or roles.get("assistant", False)):
        return "planner_executor"
    
    # Collaborative: High density and multiple agents
    elif density > 0.5 and num_agents >= 3:
        return "collaborative"
    
    # Sequential: Low density, few cycles, linear structure
    if density < 0.3 and num_cycles == 0 and num_agents >= 2:
        return "sequential"
    
    # Cyclic: Presence of cycles indicates iterative refinement
    elif num_cycles > 0 and density > 0.3:
        return "iterative_refinement"
    
    # Default
    return "unknown"


def get_architecture_description(architecture_type: str) -> str:
    """
    Get a human-readable description of the architecture type.
    
    Args:
        architecture_type: Architecture classification string
        
    Returns:
        Description string
    """
    descriptions = {
        "hierarchical": "Hierarchical architecture with a central manager/coordinator agent",
        "planner_executor": "Planner-Executor architecture with planning and execution agents",
        "collaborative": "Collaborative architecture with distributed, high-density interactions",
        "environment_interactive": "Environment-interactive with tool or system agents",
        "sequential": "Sequential architecture with linear agent chain",
        "iterative_refinement": "Iterative refinement with feedback loops",
        "unknown": "Unknown or mixed architecture pattern"
    }
    
    return descriptions.get(architecture_type, "Unknown architecture")


def analyze_architecture_properties(
    graph_features: Dict[str, Any],
    roles: Dict[str, bool],
    architecture_type: str
) -> Dict[str, Any]:
    """
    Analyze detailed properties of the identified architecture.
    
    Args:
        graph_features: Graph structural features
        roles: Detected roles
        architecture_type: Classified architecture type
        
    Returns:
        Dictionary with architecture analysis
    """
    analysis = {
        "architecture_type": architecture_type,
        "description": get_architecture_description(architecture_type),
        "properties": {}
    }
    
    # Extract centralization metrics
    degree_centrality = graph_features.get("degree_centrality", {})
    if degree_centrality:
        max_cent = max(degree_centrality.values())
        min_cent = min(degree_centrality.values())
        avg_cent = sum(degree_centrality.values()) / len(degree_centrality)
        
        analysis["properties"]["centralization"] = {
            "max": max_cent,
            "min": min_cent,
            "avg": avg_cent,
            "range": max_cent - min_cent
        }
    
    # Identify key agents
    if degree_centrality:
        sorted_agents = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        analysis["properties"]["most_central_agent"] = sorted_agents[0][0] if sorted_agents else None
        analysis["properties"]["least_central_agent"] = sorted_agents[-1][0] if sorted_agents else None
    
    # Role composition
    active_roles = [role for role, present in roles.items() if present]
    analysis["properties"]["active_roles"] = active_roles
    analysis["properties"]["role_count"] = len(active_roles)
    
    # Structural properties
    analysis["properties"]["has_cycles"] = graph_features.get("number_of_cycles", 0) > 0
    analysis["properties"]["is_connected"] = graph_features.get("is_weakly_connected", False)
    analysis["properties"]["density_category"] = categorize_density(graph_features.get("graph_density", 0.0))
    
    return analysis


def categorize_density(density: float) -> str:
    """
    Categorize graph density into human-readable categories.
    
    Args:
        density: Graph density value (0 to 1)
        
    Returns:
        Category string
    """
    if density < 0.2:
        return "sparse"
    elif density < 0.5:
        return "moderate"
    elif density < 0.8:
        return "dense"
    else:
        return "very_dense"


def detect_coordination_pattern(graph_features: Dict[str, Any]) -> str:
    """
    Detect the coordination pattern from graph structure.
    
    Args:
        graph_features: Graph structural features
        
    Returns:
        Coordination pattern string
    """
    num_cycles = graph_features.get("number_of_cycles", 0)
    density = graph_features.get("graph_density", 0.0)
    is_dag = graph_features.get("number_of_cycles", 0) == 0
    
    if is_dag and density < 0.3:
        return "pipeline"
    elif num_cycles > 0 and density > 0.4:
        return "feedback_loop"
    elif density > 0.6:
        return "fully_connected"
    elif density < 0.3:
        return "sparse_coordination"
    else:
        return "mixed"


def identify_bottlenecks(graph_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify potential bottleneck agents in the system.
    
    Args:
        graph_features: Graph structural features
        
    Returns:
        Dictionary with bottleneck analysis
    """
    bottlenecks = {
        "has_bottleneck": False,
        "bottleneck_agents": [],
        "bottleneck_score": 0.0
    }
    
    # High betweenness centrality indicates bottleneck
    betweenness = graph_features.get("betweenness_centrality", {})
    
    if betweenness:
        max_betweenness = max(betweenness.values())
        avg_betweenness = sum(betweenness.values()) / len(betweenness)
        
        # Agent is bottleneck if betweenness is significantly higher than average
        threshold = avg_betweenness + 0.3
        
        bottleneck_agents = [
            agent for agent, score in betweenness.items()
            if score > threshold and score > 0.3
        ]
        
        if bottleneck_agents:
            bottlenecks["has_bottleneck"] = True
            bottlenecks["bottleneck_agents"] = bottleneck_agents
            bottlenecks["bottleneck_score"] = max_betweenness
    
    return bottlenecks

# Made with Bob
