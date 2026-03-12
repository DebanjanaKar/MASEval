"""
Data Loading Module for MAST Dataset Analysis

This module handles loading and initial parsing of the MAST dataset from HuggingFace.
"""

from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm


def load_mast_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the MAST dataset from HuggingFace.
    
    Args:
        split: Dataset split to load (default: "train")
        
    Returns:
        DataFrame containing parsed traces with columns:
        - task_id: unique identifier for each task/trace
        - trace: list of turn dictionaries
        - raw_data: original data for reference
    """
    print(f"Loading MAST dataset (split: {split})...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("mcemri/MAST-Data", split=split)
        
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        
        # Parse dataset into structured format
        parsed_traces = []
        
        for idx, sample in enumerate(tqdm(dataset, desc="Parsing traces")):
            trace_data = parse_trace_sample(sample, task_id=idx)
            parsed_traces.append(trace_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_traces)
        
        print(f"Successfully parsed {len(df)} traces")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def parse_trace_sample(sample: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    """
    Parse a single trace sample from the dataset.
    
    Args:
        sample: Raw sample from the dataset
        task_id: Unique identifier for this trace
        
    Returns:
        Dictionary containing:
        - task_id: unique identifier
        - trace: list of turn dictionaries
        - raw_data: original sample
    """
    trace = []
    
    # The MAST dataset structure may vary, so we handle multiple possible formats
    # Common fields: messages, conversation, trajectory, etc.
    
    if "messages" in sample:
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
                
    elif "trajectory" in sample:
        trajectory = sample["trajectory"]
        for turn_id, msg in enumerate(trajectory, start=1):
            turn = extract_turn_info(msg, turn_id)
            if turn:
                trace.append(turn)
    
    # Fallback: try to parse the entire sample as a single conversation
    else:
        # Look for any list-like structure
        for key, value in sample.items():
            if isinstance(value, list) and len(value) > 0:
                for turn_id, msg in enumerate(value, start=1):
                    turn = extract_turn_info(msg, turn_id)
                    if turn:
                        trace.append(turn)
                break
    
    return {
        "task_id": task_id,
        "trace": trace,
        "raw_data": sample
    }


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
