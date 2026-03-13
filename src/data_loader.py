"""
Data Loading Module for MAST Dataset Analysis

This module handles loading and initial parsing of the MAST dataset from HuggingFace.
"""

from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm


def load_mast_dataset(split: str = "train", config: str = None) -> pd.DataFrame:
    """
    Load the MAST dataset from HuggingFace.
    
    The MAST dataset has multiple configurations with different schemas.
    This function handles loading and parsing different configurations.
    
    Args:
        split: Dataset split to load (default: "train")
        config: Specific configuration to load. Options:
                - None: Load all available data files
                - "MAD_full": Full MAD dataset
                - "MAD_human": Human-labeled subset
        
    Returns:
        DataFrame containing parsed traces with columns:
        - task_id: unique identifier for each task/trace
        - trace: list of turn dictionaries
        - raw_data: original data for reference
    """
    print(f"Loading MAST dataset (split: {split})...")
    
    try:
        # Try loading with specific data files to avoid schema conflicts
        # The dataset has multiple JSON files with different schemas
        
        # First, try loading just the full dataset file
        try:
            print("Attempting to load MAD_full_dataset.json...")
            dataset = load_dataset(
                "mcemri/MAST-Data",
                data_files="MAD_full_dataset.json",
                split=split
            )
            print(f"Successfully loaded MAD_full_dataset. Total samples: {len(dataset)}")
        except Exception as e1:
            print(f"Could not load MAD_full_dataset.json: {e1}")
            
            # Try loading the human-labeled dataset instead
            try:
                print("Attempting to load MAD_human_labelled_dataset.json...")
                dataset = load_dataset(
                    "mcemri/MAST-Data",
                    data_files="MAD_human_labelled_dataset.json",
                    split=split
                )
                print(f"Successfully loaded MAD_human_labelled_dataset. Total samples: {len(dataset)}")
            except Exception as e2:
                print(f"Could not load MAD_human_labelled_dataset.json: {e2}")
                
                # Last resort: try loading without specifying data files
                # but this may cause schema conflicts
                print("Attempting to load default configuration...")
                dataset = load_dataset("mcemri/MAST-Data", split=split)
                print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        # Parse dataset into structured format
        parsed_traces = []
        
        for idx, sample in enumerate(tqdm(dataset, desc="Parsing traces")):
            trace_data = parse_trace_sample(sample, task_id=idx)
            if trace_data and trace_data.get("trace"):  # Only add non-empty traces
                parsed_traces.append(trace_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_traces)
        
        print(f"Successfully parsed {len(df)} traces")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection")
        print("2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/datasets")
        print("3. Try loading a specific file:")
        print("   load_mast_dataset(config='MAD_full')")
        raise


def parse_trace_sample(sample: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    """
    Parse a single trace sample from the MAST dataset.
    
    MAST dataset structure:
    - mas_name: Name of the multi-agent system (e.g., "ChatDev")
    - llm_name: LLM used (e.g., "GPT-4o")
    - benchmark_name: Benchmark name (e.g., "ProgramDev")
    - trace_id: Numeric trace identifier
    - trace: Dict with 'key', 'index', 'trajectory' fields
    - mast_annotation: Annotation data
    
    Args:
        sample: Raw sample from the dataset
        task_id: Unique identifier for this trace
        
    Returns:
        Dictionary containing:
        - task_id: unique identifier
        - mas_name, llm_name, benchmark_name: metadata
        - trace: list of turn dictionaries
        - agent_roles: dict mapping normalized names to full role names
        - raw_data: original sample
    """
    trace = []
    agent_roles = {}
    
    # Extract metadata
    mas_name = sample.get("mas_name", "unknown")
    llm_name = sample.get("llm_name", "unknown")
    benchmark_name = sample.get("benchmark_name", "unknown")
    trace_id = sample.get("trace_id", task_id)
    
    # Get the trace field which contains trajectory
    trace_field = sample.get("trace", {})
    
    # Handle case where trace is a dict with trajectory field
    if isinstance(trace_field, dict) and "trajectory" in trace_field:
        trajectory_text = trace_field["trajectory"]
        # Parse the trajectory log to extract agent turns
        trace = parse_trajectory_log(trajectory_text)
        # Extract agent roles from trajectory
        agent_roles = extract_agent_roles(trajectory_text)
    
    # Fallback: try other possible formats
    elif "messages" in sample:
        messages = sample["messages"]
        for turn_id, msg in enumerate(messages, start=1):
            turn = extract_turn_info(msg, turn_id)
            if turn:
                trace.append(turn)
                
    elif "conversation" in sample:
        conversation = sample["conversation"]
        for turn_id, msg in enumerate(conversation, start=1):
            turn = extract_turn_info(msg, turn_id)
            if turn:
                trace.append(turn)
    
    return {
        "task_id": trace_id,
        "mas_name": mas_name,
        "llm_name": llm_name,
        "benchmark_name": benchmark_name,
        "trace": trace,
        "agent_roles": agent_roles,
        "raw_data": sample
    }


def parse_trajectory_log(trajectory: str) -> List[Dict[str, Any]]:
    """
    Parse the trajectory log string to extract agent turns.
    
    The MAST trajectory contains:
    - Role definitions in **assistant_role_name** and **user_role_name** fields
    - Actual agent interactions in [INFO] log lines
    
    Args:
        trajectory: Raw trajectory log string
        
    Returns:
        List of turn dictionaries with 'turn', 'agent', 'content' fields, plus agent_roles dict
    """
    import re
    
    turns = []
    
    if not trajectory or len(trajectory.strip()) == 0:
        return turns
    
    # Parse INFO lines which contain actual agent interactions
    # Pattern: [timestamp INFO] Agent Name: **[content]** or **Agent<->Agent on : Phase, turn N**
    info_pattern = r'\[.*?INFO\]\s+([^:]+?):\s*\*\*(.+?)\*\*'
    
    turn_number = 0
    lines = trajectory.split('\n')
    
    for line in lines:
        match = re.search(info_pattern, line)
        if match:
            agent_name = match.group(1).strip()
            content = match.group(2).strip()
            
            # Skip system messages and empty content
            if agent_name.lower() in ['system', 'flask', 'chatdev'] or len(content) < 5:
                continue
            
            turn_number += 1
            turns.append({
                "turn": turn_number,
                "agent": normalize_agent_name(agent_name),
                "content": content[:500]  # Truncate long content
            })
    
    return turns


def extract_agent_roles(trajectory: str) -> Dict[str, str]:
    """
    Extract agent roles from the trajectory.
    
    Extracts the full role names from assistant_role_name and user_role_name fields
    and maps them to their normalized names.
    
    Args:
        trajectory: Raw trajectory log string
        
    Returns:
        Dictionary mapping normalized agent names to full role names
    """
    import re
    
    agent_roles = {}
    
    if not trajectory or len(trajectory.strip()) == 0:
        return agent_roles
    
    # Extract all assistant_role_name and user_role_name entries
    assistant_pattern = r'\*\*assistant_role_name\*\*\s*\|\s*([^\|]+?)\s*\|'
    user_pattern = r'\*\*user_role_name\*\*\s*\|\s*([^\|]+?)\s*\|'
    
    assistant_roles = re.findall(assistant_pattern, trajectory)
    user_roles = re.findall(user_pattern, trajectory)
    
    # Combine and deduplicate
    all_roles = set(assistant_roles + user_roles)
    
    # Map normalized names to full names
    for role in all_roles:
        role_clean = role.strip()
        normalized = normalize_agent_name(role_clean)
        agent_roles[normalized] = role_clean
    
    return agent_roles


def normalize_agent_name(agent_name: str) -> str:
    """
    Normalize agent/role names to consistent format.
    
    Args:
        agent_name: Raw agent name from logs
        
    Returns:
        Normalized agent name
    """
    agent_lower = agent_name.lower()
    
    # Map common role variations to standard names
    role_mappings = {
        'chief executive officer': 'CEO',
        'ceo': 'CEO',
        'chief technology officer': 'CTO',
        'cto': 'CTO',
        'chief product officer': 'CPO',
        'cpo': 'CPO',
        'programmer': 'Programmer',
        'code reviewer': 'Reviewer',
        'reviewer': 'Reviewer',
        'designer': 'Designer',
        'tester': 'Tester',
        'counselor': 'Counselor',
        'system': 'System',
    }
    
    for key, value in role_mappings.items():
        if key in agent_lower:
            return value
    
    # Return cleaned version if no mapping found
    return agent_name.replace('**', '').strip()


def extract_turn_info(message: Any, turn_id: int) -> Dict[str, Any]:
    """
    Extract turn information from a message.
    
    Args:
        message: Message object (can be dict, string, or other format)
        turn_id: Turn number in the sequence
        
    Returns:
        Dictionary with turn information or None if invalid
    """
    turn = {"turn": turn_id}
    
    # Handle dictionary format
    if isinstance(message, dict):
        # Extract agent name
        agent_name = (
            message.get("agent") or 
            message.get("agent_name") or 
            message.get("role") or 
            message.get("name") or 
            message.get("from") or
            "unknown"
        )
        
        # Extract content
        content = (
            message.get("content") or 
            message.get("message") or 
            message.get("text") or 
            str(message)
        )
        
        turn["agent"] = str(agent_name).lower().strip()
        turn["content"] = str(content)
        
        # Extract role if available
        if "role" in message:
            turn["role"] = message["role"]
            
    # Handle string format
    elif isinstance(message, str):
        # Try to parse agent from string (e.g., "Agent: message")
        if ":" in message:
            parts = message.split(":", 1)
            turn["agent"] = parts[0].strip().lower()
            turn["content"] = parts[1].strip()
        else:
            turn["agent"] = "unknown"
            turn["content"] = message
    
    else:
        # Unknown format
        turn["agent"] = "unknown"
        turn["content"] = str(message)
    
    # Only return valid turns with content
    if turn.get("content") and len(turn["content"]) > 0:
        return turn
    
    return None


def get_trace_by_id(df: pd.DataFrame, task_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve a specific trace by task_id.
    
    Args:
        df: DataFrame containing all traces
        task_id: ID of the trace to retrieve
        
    Returns:
        List of turn dictionaries for the specified trace
    """
    if task_id >= len(df):
        raise ValueError(f"task_id {task_id} not found. Dataset has {len(df)} traces.")
    
    return df.iloc[task_id]["trace"]


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for the loaded dataset.
    
    Args:
        df: DataFrame containing all traces
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_traces": len(df),
        "avg_turns_per_trace": df["trace"].apply(len).mean(),
        "min_turns": df["trace"].apply(len).min(),
        "max_turns": df["trace"].apply(len).max(),
        "total_turns": df["trace"].apply(len).sum()
    }
    
    return summary

# Made with Bob
