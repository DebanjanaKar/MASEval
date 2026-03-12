# MAS Analysis Pipeline

A comprehensive research pipeline for analyzing Multi-Agent System (MAS) interaction traces from the MAST dataset.

## Overview

This pipeline analyzes MAS interaction traces and extracts structural and coordination metrics using communication graphs. It supports both dataset-level aggregate analysis and detailed single-trace analysis.

## Features

### Dataset-Level Analysis
- Analyze all traces in the MAST dataset
- Compute aggregate statistics and distributions
- Classify MAS architectures
- Generate comprehensive visualizations
- Export results to CSV

### Single-Trace Analysis
- Detailed breakdown of individual traces
- Communication graph visualization
- Step-by-step agent interaction analysis
- Coordination metrics computation
- Architecture classification

### Metrics Computed

1. **Loop Index (LI)**: Measures interaction loops using cycle detection
2. **Agent Dependency Ratio (ADR)**: Measures reliance on single agents
3. **Communication Entropy (CE)**: Measures unpredictability of agent transitions
4. **Coordination Stability Score (CSS)**: Measures consistency across traces
5. **Graph Features**: Density, centrality, clustering, cycles, etc.

### Architecture Classification

The pipeline automatically classifies MAS architectures:
- **Hierarchical**: Central manager/coordinator
- **Planner-Executor**: Planning and execution agents
- **Collaborative**: Distributed, high-density interactions
- **Environment-Interactive**: Tool or system agents
- **Sequential**: Linear agent chain
- **Iterative Refinement**: Feedback loops

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd mas_analysis_pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset-Level Analysis

Run analysis on the entire MAST dataset:

```bash
python main.py
```

This will:
1. Load the MAST dataset from HuggingFace
2. Analyze all traces
3. Generate aggregate statistics
4. Create visualizations
5. Export results to `outputs/mast_analysis_results.csv`

### Single-Trace Analysis

Analyze a specific trace by task ID:

```bash
python main.py --task_id 5
```

This will:
1. Load the dataset
2. Analyze the specified trace in detail
3. Generate trace-specific visualizations
4. Export detailed results

### Custom Output Directories

Specify custom output directories:

```bash
python main.py --output_dir my_results --viz_dir my_plots
```

For single-trace analysis:

```bash
python main.py --task_id 10 --output_dir trace_results --viz_dir trace_plots
```

### Command-Line Options

```
usage: main.py [-h] [--task_id TASK_ID] [--output_dir OUTPUT_DIR] 
               [--viz_dir VIZ_DIR] [--split SPLIT]

optional arguments:
  -h, --help            Show help message
  --task_id TASK_ID     Task ID for single-trace analysis (omit for dataset-level)
  --output_dir OUTPUT_DIR
                        Directory for output files (default: outputs)
  --viz_dir VIZ_DIR     Directory for visualizations (default: visualizations)
  --split SPLIT         Dataset split to load (default: train)
```

## Project Structure

```
mas_analysis_pipeline/
├── main.py                          # Main orchestration script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initialization
│   ├── data_loader.py              # Dataset loading and parsing
│   ├── trace_parser.py             # Agent turn parsing
│   ├── graph_builder.py            # Communication graph construction
│   ├── metrics.py                  # Coordination metrics
│   ├── architecture_classifier.py  # Architecture classification
│   ├── dataset_analyzer.py         # Dataset-level analysis
│   ├── single_trace_analyzer.py    # Single-trace analysis
│   └── visualizer.py               # Visualization generation
├── outputs/                        # Analysis results (created automatically)
│   ├── mast_analysis_results.csv   # Dataset analysis results
│   ├── trace_*_breakdown.csv       # Single-trace breakdowns
│   └── trace_*_summary.json        # Single-trace summaries
└── visualizations/                 # Generated plots (created automatically)
    ├── architecture_distribution.png
    ├── loop_index_distribution.png
    ├── correlation_heatmap.png
    ├── trace_*_communication_graph.png
    └── ... (more visualizations)
```

## Module Documentation

### data_loader.py
Handles loading and initial parsing of the MAST dataset from HuggingFace.

**Key Functions:**
- `load_mast_dataset()`: Load dataset from HuggingFace
- `parse_trace_sample()`: Parse individual trace samples
- `get_trace_by_id()`: Retrieve specific trace

### trace_parser.py
Parses agent turns and extracts interaction patterns.

**Key Functions:**
- `parse_agent_turns()`: Extract turns, agents, and transitions
- `detect_roles()`: Identify common MAS roles
- `extract_agent_statistics()`: Compute agent participation stats

### graph_builder.py
Constructs directed communication graphs from traces.

**Key Functions:**
- `build_communication_graph()`: Create NetworkX DiGraph
- `extract_graph_features()`: Compute structural features
- `get_node_importance_ranking()`: Rank agents by importance

### metrics.py
Implements graph-based coordination metrics.

**Key Functions:**
- `compute_loop_index()`: Detect interaction loops
- `compute_agent_dependency_ratio()`: Measure agent centralization
- `compute_communication_entropy()`: Measure transition unpredictability
- `compute_coordination_stability()`: Measure cross-trace consistency

### architecture_classifier.py
Classifies MAS architectures using topology and roles.

**Key Functions:**
- `classify_architecture()`: Classify architecture type
- `analyze_architecture_properties()`: Detailed architecture analysis
- `identify_bottlenecks()`: Detect bottleneck agents

### dataset_analyzer.py
Performs aggregate analysis across all traces.

**Key Functions:**
- `analyze_dataset()`: Run complete dataset analysis
- `export_results()`: Export results to CSV
- `compute_architecture_statistics()`: Architecture-specific stats

### single_trace_analyzer.py
Provides detailed analysis for individual traces.

**Key Functions:**
- `analyze_single_trace()`: Comprehensive single-trace analysis
- `export_trace_analysis()`: Export trace results
- `get_turn_by_turn_breakdown()`: Create turn breakdown

### visualizer.py
Creates visualizations for both analysis modes.

**Key Functions:**
- `visualize_dataset_analysis()`: Generate dataset visualizations
- `visualize_single_trace()`: Generate trace visualizations
- `plot_communication_graph()`: Visualize agent interactions

## Output Files

### Dataset-Level Outputs

**mast_analysis_results.csv**
Contains one row per trace with columns:
- `task_id`: Trace identifier
- `architecture_type`: Classified architecture
- `num_agents`: Number of agents
- `num_edges`: Number of interactions
- `graph_density`: Graph density metric
- `loop_index`: Loop index value
- `agent_dependency_ratio`: ADR value
- `communication_entropy`: CE value
- `has_*`: Boolean flags for detected roles

**Visualizations:**
- `architecture_distribution.png`: Bar chart of architecture types
- `loop_index_distribution.png`: Histogram of loop indices
- `agent_dependency_ratio_distribution.png`: ADR distribution
- `communication_entropy_distribution.png`: CE distribution
- `agent_dependency_ratio_by_architecture.png`: Box plot by architecture
- `density_vs_agents_by_architecture.png`: Scatter plot
- `correlation_heatmap.png`: Metric correlations

### Single-Trace Outputs

**trace_{id}_breakdown.csv**
Turn-by-turn breakdown with columns:
- `turn`: Turn number
- `agent`: Agent name
- `content_length`: Message length
- `content_preview`: Message preview

**trace_{id}_summary.json**
JSON file with:
- Trace information
- Detected roles
- Architecture classification
- Computed metrics
- Agent statistics
- Bottleneck analysis

**Visualizations:**
- `trace_{id}_communication_graph.png`: Network graph
- `trace_{id}_adjacency_matrix.png`: Heatmap of interactions
- `trace_{id}_agent_timeline.png`: Timeline of agent activity
- `trace_{id}_agent_participation.png`: Bar chart of participation

## Research Applications

This pipeline is designed for research on:

1. **MAS Reliability**: Analyze coordination patterns and failure modes
2. **Architecture Comparison**: Compare different MAS designs
3. **Coordination Metrics**: Develop and validate new metrics
4. **Pattern Discovery**: Identify common interaction patterns
5. **Bottleneck Detection**: Find communication bottlenecks
6. **Scalability Analysis**: Study how systems scale with agents

## Extending the Pipeline

### Adding New Metrics

Add new metric functions to `src/metrics.py`:

```python
def compute_my_metric(graph_data: Dict[str, Any]) -> float:
    """Compute custom metric."""
    G = graph_data["graph"]
    # Your metric computation
    return metric_value
```

Then update `compute_all_metrics()` to include it.

### Adding New Visualizations

Add visualization functions to `src/visualizer.py`:

```python
def plot_my_visualization(analysis: Dict[str, Any], output_dir: str):
    """Create custom visualization."""
    # Your plotting code
    plt.savefig(os.path.join(output_dir, "my_plot.png"))
```

### Custom Architecture Types

Extend `classify_architecture()` in `src/architecture_classifier.py`:

```python
# Add new classification logic
if custom_condition:
    return "my_architecture_type"
```

## Dataset Information

**MAST Dataset**: Multi-Agent System Traces
- **Source**: https://huggingface.co/datasets/mcemri/MAST-Data
- **Description**: Collection of multi-agent LLM system interaction traces
- **Format**: Conversational traces with agent turns

## Troubleshooting

### Dataset Loading Issues

If you encounter issues loading the dataset:

1. Check internet connection
2. Verify HuggingFace datasets library is installed
3. Try clearing cache: `rm -rf ~/.cache/huggingface/datasets`

### Memory Issues

For large datasets:

1. Process traces in batches
2. Reduce visualization resolution
3. Use a machine with more RAM

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Performance Considerations

- **Dataset Loading**: First load may take time (downloads dataset)
- **Analysis Time**: Depends on dataset size (~1-2 seconds per trace)
- **Memory Usage**: Scales with dataset size and graph complexity
- **Visualization**: Can be slow for large graphs (>20 nodes)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mas_analysis_pipeline,
  title={MAS Analysis Pipeline: A Research Tool for Multi-Agent System Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mas_analysis_pipeline}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## Acknowledgments

- MAST Dataset creators
- NetworkX development team
- HuggingFace datasets library
- Research community feedback

## Version History

### v1.0.0 (2024)
- Initial release
- Dataset-level analysis
- Single-trace analysis
- Core metrics implementation
- Visualization suite
- Architecture classification

## Future Enhancements

Planned features:
- [ ] Interactive visualizations (Plotly)
- [ ] Real-time analysis dashboard
- [ ] Comparative analysis across datasets
- [ ] Machine learning-based architecture prediction
- [ ] Temporal analysis of coordination patterns
- [ ] Export to graph databases (Neo4j)
- [ ] Integration with LLM evaluation frameworks