"""
Trace Parsing Module

This module handles parsing agent turns and extracting agent interaction sequences.
"""

from typing import List, Dict, Any, Tuple
import re


def parse_agent_turns(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse agent turns from a trace and extract interaction patterns.
    
    Args:
        trace: List of turn dictionaries with 'turn', 'agent', 'content' fields
        
    Returns:
        Dictionary containing:
        - turns: ordered list of turns
        - agents: list of unique agents
        - transitions: sequence of agent transitions
        - transition_pairs: list of (from_agent, to_agent) tuples
    """
    if not trace or len(trace) == 0:
        return {
            "turns": [],
            "agents": [],
            "transitions": [],
            "transition_pairs": []
        }
    
    # Extract ordered turns
    turns = sorted(trace, key=lambda x: x.get("turn", 0))
    
    # Extract unique agents
    agents = list(set(turn["agent"] for turn in turns if "agent" in turn))
    
    # Build transition sequence
    transitions = []
    transition_pairs = []
    
    for i in range(len(turns) - 1):
        current_agent = turns[i].get("agent", "unknown")
        next_agent = turns[i + 1].get("agent", "unknown")
        
        transitions.append(f"{current_agent} → {next_agent}")
        transition_pairs.append((current_agent, next_agent))
    
    return {
        "turns": turns,
        "agents": agents,
        "transitions": transitions,
        "transition_pairs": transition_pairs
    }


def detect_roles(turns: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Detect common MAS roles based on agent names.
    
    Args:
        turns: List of turn dictionaries
        
    Returns:
        Dictionary mapping role names to boolean presence indicators
    """
    # Extract all agent names
    agent_names = set()
    for turn in turns:
        if "agent" in turn:
            agent_names.add(turn["agent"].lower())
    
    # Define role keywords
    role_keywords = {
        "planner": ["planner", "plan", "planning"],
        "worker": ["worker", "executor", "execute", "action"],
        "critic": ["critic", "review", "reviewer", "evaluator"],
        "verifier": ["verifier", "verify", "validator", "validation"],
        "manager": ["manager", "coordinator", "orchestrator"],
        "tool": ["tool", "function", "api"],
        "environment": ["environment", "env", "system"],
        "user": ["user", "human"],
        "assistant": ["assistant", "agent"]
    }
    
    # Detect roles
    roles = {}
    for role, keywords in role_keywords.items():
        roles[role] = any(
            any(keyword in agent_name for keyword in keywords)
            for agent_name in agent_names
        )
    
    return roles


def extract_agent_statistics(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract statistical information about agent participation.
    
    Args:
        turns: List of turn dictionaries
        
    Returns:
        Dictionary with agent statistics
    """
    if not turns:
        return {}
    
    # Count turns per agent
    agent_turn_counts = {}
    for turn in turns:
        agent = turn.get("agent", "unknown")
        agent_turn_counts[agent] = agent_turn_counts.get(agent, 0) + 1
    
    # Calculate statistics
    total_turns = len(turns)
    unique_agents = len(agent_turn_counts)
    
    # Find most active agent
    most_active_agent = max(agent_turn_counts.items(), key=lambda x: x[1])
    
    # Calculate participation ratios
    participation_ratios = {
        agent: count / total_turns 
        for agent, count in agent_turn_counts.items()
    }
    
    return {
        "total_turns": total_turns,
        "unique_agents": unique_agents,
        "agent_turn_counts": agent_turn_counts,
        "most_active_agent": most_active_agent[0],
        "most_active_agent_turns": most_active_agent[1],
        "participation_ratios": participation_ratios
    }


def get_agent_sequence(turns: List[Dict[str, Any]]) -> List[str]:
    """
    Get the sequence of agents in order of their turns.
    
    Args:
        turns: List of turn dictionaries
        
    Returns:
        List of agent names in sequential order
    """
    sorted_turns = sorted(turns, key=lambda x: x.get("turn", 0))
    return [turn.get("agent", "unknown") for turn in sorted_turns]


def identify_conversation_patterns(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify common conversation patterns in the trace.
    
    Args:
        turns: List of turn dictionaries
        
    Returns:
        Dictionary with identified patterns
    """
    agent_sequence = get_agent_sequence(turns)
    
    patterns = {
        "has_loops": False,
        "loop_sequences": [],
        "has_back_and_forth": False,
        "back_and_forth_pairs": [],
        "longest_agent_streak": 0,
        "longest_streak_agent": None
    }
    
    # Detect loops (repeated sequences)
    for length in range(2, min(6, len(agent_sequence) // 2)):
        for i in range(len(agent_sequence) - 2 * length):
            sequence = agent_sequence[i:i + length]
            next_sequence = agent_sequence[i + length:i + 2 * length]
            if sequence == next_sequence:
                patterns["has_loops"] = True
                patterns["loop_sequences"].append(sequence)
    
    # Detect back-and-forth (A → B → A patterns)
    for i in range(len(agent_sequence) - 2):
        if agent_sequence[i] == agent_sequence[i + 2] and agent_sequence[i] != agent_sequence[i + 1]:
            patterns["has_back_and_forth"] = True
            pair = (agent_sequence[i], agent_sequence[i + 1])
            if pair not in patterns["back_and_forth_pairs"]:
                patterns["back_and_forth_pairs"].append(pair)
    
    # Find longest streak of same agent
    current_streak = 1
    max_streak = 1
    streak_agent = agent_sequence[0] if agent_sequence else None
    
    for i in range(1, len(agent_sequence)):
        if agent_sequence[i] == agent_sequence[i - 1]:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                streak_agent = agent_sequence[i]
        else:
            current_streak = 1
    
    patterns["longest_agent_streak"] = max_streak
    patterns["longest_streak_agent"] = streak_agent
    
    return patterns

# Made with Bob
