# MAS Analysis Pipeline - Architecture Documentation

## System Overview

The MAS Analysis Pipeline is a modular research tool designed to analyze Multi-Agent System (MAS) interaction traces. It uses graph-based approaches to extract coordination metrics and classify system architectures.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│                  (Orchestration Layer)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────────┐
│  Dataset-Level│         │  Single-Trace    │
│    Analysis   │         │    Analysis      │
└───────┬───────┘         └────────┬─────────┘
        │                          │
        └──────────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Data Loader  │      │ Visualizer   │
└──────┬───────┘      └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│         Core Analysis Modules         │
├──────────────────────────────────────┤
│ • Trace Parser                       │
│ • Graph Builder                      │
│ • Metrics Computer                   │
│ • Architecture Classifier            │
└──────────────────────────────────────┘
```

## Module Hierarchy

### Layer 1: Entry Point
- **main.py**: Command-line interface and orchestration

### Layer 2: Analysis Modes
- **dataset_analyzer.py**: Aggregate analysis across all traces
- **single_trace_analyzer.py**: Detailed analysis of individual traces

### Layer 3: Core Components
- **data_loader.py**: Dataset loading and parsing
- **trace_parser.py**: Turn extraction and role detection
- **graph_builder.py**: Communication graph construction
- **metrics.py**: Coordination metric computation
- **architecture_classifier.py**: Architecture type classification
- **visualizer.py**: Visualization generation

## Data Flow

### Dataset-Level Analysis Flow

```
1. Load Dataset
   └─> HuggingFace → DataFrame[traces]

2. For Each Trace:
   ├─> Parse Turns → [agent, content, turn]
   ├─> Build Graph → NetworkX DiGraph
   ├─> Extract Features → {density, centrality, ...}
   ├─> Detect Roles → {planner, worker, ...}
   ├─> Classify Architecture → architecture_type
   └─> Compute Metrics → {LI, ADR, CE, ...}

3. Aggregate Results
   └─> DataFrame[task_id, metrics, architecture, ...]

4. Generate Visualizations
   ├─> Architecture distribution
   ├─> Metric distributions
   └─> Correlation heatmaps

5. Export Results
   └─> CSV file
```

### Single-Trace Analysis Flow

```
1. Load Dataset
   └─> Select trace by task_id

2. Detailed Analysis:
   ├─> Parse Turns
   ├─> Build Communication Graph
   ├─> Extract Graph Features
   ├─> Detect Roles
   ├─> Classify Architecture
   ├─> Compute All Metrics
   ├─> Identify Bottlenecks
   └─> Extract Agent Statistics

3. Generate Visualizations:
   ├─> Communication graph (NetworkX)
   ├─> Adjacency matrix heatmap
   ├─> Agent timeline
   └─> Participation chart

4. Export Results:
   ├─> Turn-by-turn breakdown (CSV)
   └─> Summary (JSON)
```

## Key Design Patterns

### 1. Modular Architecture
Each module has a single responsibility:
- **Separation of Concerns**: Data loading, analysis, and visualization are separate
- **Reusability**: Modules can be used independently
- **Testability**: Each module can be tested in isolation

### 2. Graph-Centric Approach
Communication graphs are the core data structure:
- **Nodes**: Agents
- **Edges**: Sequential interactions
- **Weights**: Interaction frequency
- **Benefits**: Enables graph algorithms for metric computation

### 3. Pipeline Pattern
Data flows through a series of transformations:
```
Raw Data → Parsed Traces → Graphs → Features → Metrics → Results
```

### 4. Strategy Pattern
Different analysis strategies for different modes:
- Dataset-level: Aggregate statistics
- Single-trace: Detailed breakdown

## Core Data Structures

### Trace Representation
```python
trace = [
    {"turn": 1, "agent": "planner", "content": "..."},
    {"turn": 2, "agent": "worker", "content": "..."},
    {"turn": 3, "agent": "critic", "content": "..."}
]
```

### Graph Data
```python
graph_data = {
    "graph": nx.DiGraph(),
    "edge_list": [(source, target, weight), ...],
    "transition_matrix": np.ndarray,
    "adjacency_matrix": np.ndarray,
    "agents": [agent_names]
}
```

### Analysis Results
```python
analysis = {
    "task_id": int,
    "trace_info": {...},
    "graph_features": {...},
    "roles": {...},
    "architecture": {...},
    "metrics": {...},
    "agent_statistics": {...}
}
```

## Metric Computation Details

### Loop Index (LI)
```
LI = number_of_cycles / number_of_edges

Uses NetworkX simple_cycles() for cycle detection
Higher values indicate more iterative refinement
```

### Agent Dependency Ratio (ADR)
```
ADR = max_node_degree / (2 * total_edges)

Measures centralization around a single agent
Higher values indicate potential bottlenecks
```

### Communication Entropy (CE)
```
CE = -Σ P(i→j) log₂ P(i→j)

Uses transition probability matrix
Higher values indicate more unpredictable interactions
```

### Coordination Stability Score (CSS)
```
CSS = mean(variance(transition_matrices))

Computed across multiple traces
Lower values indicate more consistent patterns
```

## Architecture Classification Logic

```python
if tool_agents or environment_agents:
    return "environment_interactive"
elif manager_agent and high_centrality:
    return "hierarchical"
elif planner_agent and worker_agent:
    return "planner_executor"
elif high_density and multiple_agents:
    return "collaborative"
elif low_density and no_cycles:
    return "sequential"
elif cycles_present:
    return "iterative_refinement"
else:
    return "unknown"
```

## Extensibility Points

### Adding New Metrics

1. Add function to `src/metrics.py`:
```python
def compute_my_metric(graph_data: Dict[str, Any]) -> float:
    G = graph_data["graph"]
    # Compute metric
    return value
```

2. Update `compute_all_metrics()` to include it

3. Add to dataset analyzer output columns

### Adding New Architecture Types

1. Update `classify_architecture()` in `src/architecture_classifier.py`
2. Add classification logic
3. Update `get_architecture_description()`

### Adding New Visualizations

1. Add function to `src/visualizer.py`:
```python
def plot_my_visualization(data, output_dir):
    # Create plot
    plt.savefig(os.path.join(output_dir, "my_plot.png"))
```

2. Call from `visualize_dataset_analysis()` or `visualize_single_trace()`

## Performance Considerations

### Time Complexity
- **Dataset Loading**: O(n) where n = number of traces
- **Graph Construction**: O(t) where t = number of turns per trace
- **Cycle Detection**: O(V + E) where V = agents, E = edges
- **Overall**: O(n * t) for dataset analysis

### Space Complexity
- **Graphs**: O(V²) for adjacency matrices
- **Results**: O(n * m) where m = number of metrics
- **Visualizations**: O(n) for dataset plots

### Optimization Strategies
1. **Batch Processing**: Process traces in chunks
2. **Lazy Loading**: Load visualizations on demand
3. **Caching**: Cache graph features for reuse
4. **Parallel Processing**: Use multiprocessing for large datasets

## Error Handling

### Graceful Degradation
- Empty traces: Skip with warning
- Missing data: Use default values
- Invalid graphs: Return empty metrics

### Validation
- Check task_id bounds
- Validate trace structure
- Verify graph connectivity

## Testing Strategy

### Unit Tests
- Test each module independently
- Mock external dependencies
- Verify metric calculations

### Integration Tests
- Test full pipeline flow
- Verify output formats
- Check visualization generation

### Example Test Structure
```python
def test_loop_index():
    # Create test graph
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    
    # Compute metric
    li = compute_loop_index(G, list(G.edges()))
    
    # Verify
    assert li > 0  # Has cycle
```

## Configuration

### Environment Variables
- `MAST_CACHE_DIR`: Cache directory for dataset
- `OUTPUT_DIR`: Default output directory
- `VIZ_DIR`: Default visualization directory

### Constants
Defined in module headers:
- Graph layout parameters
- Visualization settings
- Metric thresholds

## Dependencies

### Core Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **networkx**: Graph algorithms
- **matplotlib/seaborn**: Visualization
- **datasets**: HuggingFace integration

### Optional Libraries
- **scipy**: Advanced statistics
- **jupyter**: Interactive analysis

## Future Enhancements

### Planned Features
1. **Interactive Visualizations**: Plotly integration
2. **Real-time Analysis**: Streaming trace analysis
3. **ML-based Classification**: Learn architecture patterns
4. **Temporal Analysis**: Time-series coordination metrics
5. **Comparative Analysis**: Cross-dataset comparison
6. **Graph Database Export**: Neo4j integration

### Research Directions
1. Failure prediction models
2. Optimal coordination patterns
3. Scalability analysis
4. Cross-domain generalization

## References

### Graph Theory
- NetworkX documentation
- Graph centrality measures
- Cycle detection algorithms

### Multi-Agent Systems
- MAS coordination patterns
- Agent communication protocols
- Architecture taxonomies

### Metrics
- Information entropy
- Network analysis metrics
- Coordination measures