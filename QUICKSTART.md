# Quick Start Guide

Get started with the MAS Analysis Pipeline in 5 minutes!

## Installation

```bash
# Navigate to the project directory
cd mas_analysis_pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Dataset-Level Analysis

Analyze the entire MAST dataset:

```bash
python main.py
```

**What happens:**
- Loads MAST dataset from HuggingFace
- Analyzes all traces
- Generates statistics and visualizations
- Saves results to `outputs/mast_analysis_results.csv`

**Expected output:**
```
Loading MAST dataset...
Dataset loaded successfully. Total samples: 1000
Analyzing traces: 100%|████████████| 1000/1000
DATASET SUMMARY STATISTICS
Total traces analyzed: 1000
Architecture Distribution:
  planner_executor: 450 (45.0%)
  collaborative: 300 (30.0%)
  ...
Results exported to: outputs/mast_analysis_results.csv
```

### 2. Single-Trace Analysis

Analyze a specific trace:

```bash
python main.py --task_id 5
```

**What happens:**
- Loads dataset
- Analyzes trace #5 in detail
- Creates communication graph
- Computes all metrics
- Generates trace-specific visualizations

**Expected output:**
```
SINGLE TRACE ANALYSIS - Task ID: 5
--- Trace Information ---
Number of turns: 15
Number of agents: 4
Agents: planner, worker, critic, verifier

--- Architecture Analysis ---
Type: planner_executor
Description: Planner-Executor architecture...

--- Coordination Metrics ---
Loop Index (LI): 0.2500
Agent Dependency Ratio (ADR): 0.3750
Communication Entropy (CE): 1.8562
```

## Understanding the Output

### Dataset Analysis Results

**CSV File** (`outputs/mast_analysis_results.csv`):
- One row per trace
- Columns: architecture_type, num_agents, graph_density, loop_index, etc.
- Import into Excel, pandas, or R for further analysis

**Visualizations** (`visualizations/`):
- `architecture_distribution.png` - Bar chart of architecture types
- `loop_index_distribution.png` - Histogram of loop indices
- `correlation_heatmap.png` - Metric correlations
- And more...

### Single-Trace Results

**Breakdown CSV** (`outputs/trace_5_breakdown.csv`):
- Turn-by-turn agent interactions
- Message content previews

**Summary JSON** (`outputs/trace_5_summary.json`):
- Complete analysis results
- Metrics, roles, architecture
- Machine-readable format

**Visualizations** (`visualizations/`):
- `trace_5_communication_graph.png` - Network diagram
- `trace_5_adjacency_matrix.png` - Interaction heatmap
- `trace_5_agent_timeline.png` - Timeline of activity
- `trace_5_agent_participation.png` - Participation bar chart

## Key Metrics Explained

### Loop Index (LI)
- **Range**: 0 to ∞
- **Meaning**: Higher values indicate more interaction loops
- **Interpretation**: 
  - LI < 0.1: Few loops (linear workflow)
  - LI > 0.5: Many loops (iterative refinement)

### Agent Dependency Ratio (ADR)
- **Range**: 0 to 1
- **Meaning**: Measures centralization around one agent
- **Interpretation**:
  - ADR < 0.3: Distributed coordination
  - ADR > 0.7: Highly centralized (potential bottleneck)

### Communication Entropy (CE)
- **Range**: 0 to log₂(n)
- **Meaning**: Unpredictability of agent transitions
- **Interpretation**:
  - CE < 1: Predictable patterns
  - CE > 2: High variability in interactions

## Common Use Cases

### Compare Architectures

```bash
# Run dataset analysis
python main.py

# Open results in Python
import pandas as pd
df = pd.read_csv('outputs/mast_analysis_results.csv')

# Compare metrics by architecture
df.groupby('architecture_type')['loop_index'].mean()
```

### Find Interesting Traces

```python
import pandas as pd
df = pd.read_csv('outputs/mast_analysis_results.csv')

# Find traces with high loop index
high_loops = df[df['loop_index'] > 0.5]
print(high_loops[['task_id', 'architecture_type', 'loop_index']])

# Analyze one of them
# python main.py --task_id <id>
```

### Batch Analysis

```bash
# Analyze multiple specific traces
for id in 5 10 15 20; do
    python main.py --task_id $id
done
```

## Troubleshooting

### "Dataset not found"
- Check internet connection
- Dataset downloads on first run
- May take a few minutes

### "Out of memory"
- Reduce dataset size
- Process traces in batches
- Use a machine with more RAM

### "Import errors"
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt --upgrade`

## Next Steps

1. **Explore the data**: Open CSV files in Excel or pandas
2. **Customize analysis**: Modify `src/` modules for your needs
3. **Add metrics**: Implement custom metrics in `src/metrics.py`
4. **Create visualizations**: Add plots in `src/visualizer.py`

## Getting Help

- Read the full [README.md](README.md)
- Check module documentation in source files
- Open an issue on GitHub

## Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run dataset analysis
python main.py

# 3. Examine results
cat outputs/mast_analysis_results.csv | head

# 4. Find interesting trace
# (e.g., task_id 42 has high loop_index)

# 5. Analyze that trace
python main.py --task_id 42

# 6. View visualizations
open visualizations/trace_42_communication_graph.png
```

Happy analyzing! 🚀