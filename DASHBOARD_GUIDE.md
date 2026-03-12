# Interactive Dashboard Guide

## Overview

The MAS Analysis Pipeline includes an interactive web-based dashboard built with Streamlit and Plotly. This dashboard provides real-time visualization and exploration of both dataset-level and single-trace analysis.

## Features

### 🎯 Two Analysis Modes

1. **Dataset-Level Analysis**
   - View aggregate statistics across all traces
   - Interactive architecture distribution charts
   - Metric distributions with statistical overlays
   - Correlation heatmaps
   - Scatter matrices for multi-dimensional analysis
   - Filterable data tables
   - Downloadable results

2. **Single-Trace Analysis**
   - Interactive communication graphs
   - Real-time metric computation
   - Agent timeline visualization
   - Participation charts
   - Turn-by-turn breakdown
   - Downloadable trace details

### 🎨 Interactive Visualizations

All visualizations are interactive with:
- **Zoom & Pan**: Explore data in detail
- **Hover Information**: See detailed data on hover
- **Click to Filter**: Interactive filtering
- **Download**: Save plots as PNG
- **Responsive**: Adapts to screen size

## Installation

### Prerequisites

```bash
# Ensure you have Python 3.8+
python --version

# Navigate to project directory
cd mas_analysis_pipeline
```

### Install Dependencies

```bash
# Install all requirements including dashboard dependencies
pip install -r requirements.txt
```

The dashboard requires:
- `streamlit>=1.28.0` - Web framework
- `plotly>=5.17.0` - Interactive visualizations

## Running the Dashboard

### Basic Usage

```bash
streamlit run dashboard.py
```

The dashboard will automatically:
1. Start a local web server
2. Open your default browser
3. Load at `http://localhost:8501`

### Custom Port

```bash
streamlit run dashboard.py --server.port 8502
```

### Network Access

To access from other devices on your network:

```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP:8501`

## Dashboard Interface

### Sidebar

The sidebar contains:
- **Mode Selection**: Switch between dataset and single-trace analysis
- **About Section**: Quick reference information
- **Metrics Guide**: Explanation of key metrics

### Dataset-Level Analysis

#### Tab 1: Architecture Analysis
- **Architecture Distribution**: Bar chart showing count of each architecture type
- **Architecture Statistics**: Table with average metrics per architecture
- **Use Case**: Understand the distribution of MAS patterns in your dataset

#### Tab 2: Metric Distributions
- **Four Distribution Plots**: 
  - Loop Index
  - Agent Dependency Ratio
  - Communication Entropy
  - Graph Density
- **Statistical Overlays**: Mean and median lines
- **Box Plots**: Compare metrics across architectures
- **Use Case**: Identify outliers and understand metric ranges

#### Tab 3: Correlations
- **Correlation Heatmap**: See relationships between metrics
- **Scatter Matrix**: Multi-dimensional view of key metrics
- **Use Case**: Discover metric relationships and dependencies

#### Tab 4: Data Table
- **Filterable Table**: Filter by architecture and agent count
- **Sortable Columns**: Click headers to sort
- **Download Button**: Export filtered data as CSV
- **Use Case**: Detailed data exploration and export

### Single-Trace Analysis

#### Trace Selection
1. Enter trace ID (0 to dataset size - 1)
2. Click "Analyze Trace" button
3. View comprehensive analysis

#### Tab 1: Communication Graph
- **Interactive Network Graph**: 
  - Nodes = Agents (size = degree)
  - Edges = Interactions (width = frequency)
  - Hover for details
  - Drag to rearrange
- **Adjacency Matrix**: Heatmap of interaction frequencies
- **Use Case**: Understand agent communication patterns

#### Tab 2: Metrics & Statistics
- **Coordination Metrics Cards**:
  - Loop Index (LI)
  - Agent Dependency Ratio (ADR)
  - Communication Entropy (CE)
- **Architecture Analysis**: Type, description, centrality
- **Detected Roles**: Active agent roles
- **Graph Features**: Structural properties
- **Use Case**: Quantitative analysis of coordination

#### Tab 3: Timeline & Participation
- **Agent Timeline**: Scatter plot showing when each agent acts
- **Participation Chart**: Bar chart of turn counts per agent
- **Agent Statistics Table**: Detailed participation percentages
- **Use Case**: Temporal analysis and workload distribution

#### Tab 4: Turn Details
- **Turn-by-Turn Table**: Complete conversation breakdown
- **Content Preview**: First 200 characters of each message
- **Download Button**: Export turn details as CSV
- **Use Case**: Detailed trace inspection

## Key Metrics Explained

### Loop Index (LI)
- **Formula**: `number_of_cycles / number_of_edges`
- **Range**: 0 to ∞
- **Interpretation**:
  - LI < 0.1: Linear workflow
  - LI 0.1-0.5: Some iteration
  - LI > 0.5: Heavy iteration/refinement
- **Dashboard**: Shown in distribution plots and trace cards

### Agent Dependency Ratio (ADR)
- **Formula**: `max_node_degree / (2 * total_edges)`
- **Range**: 0 to 1
- **Interpretation**:
  - ADR < 0.3: Distributed coordination
  - ADR 0.3-0.7: Moderate centralization
  - ADR > 0.7: Highly centralized (bottleneck risk)
- **Dashboard**: Color-coded in visualizations

### Communication Entropy (CE)
- **Formula**: `-Σ P(i→j) log₂ P(i→j)`
- **Range**: 0 to log₂(n)
- **Interpretation**:
  - CE < 1: Predictable patterns
  - CE 1-2: Moderate variability
  - CE > 2: High unpredictability
- **Dashboard**: Shown with distribution statistics

## Tips & Best Practices

### Performance Optimization

1. **First Load**: Dataset analysis takes time on first run (cached afterward)
2. **Large Datasets**: Consider analyzing a subset first
3. **Browser**: Use Chrome or Firefox for best performance
4. **Memory**: Close other applications if dashboard is slow

### Effective Analysis Workflow

1. **Start with Dataset View**:
   - Get overview of architecture distribution
   - Identify interesting patterns in metrics
   - Note outlier traces

2. **Drill Down to Traces**:
   - Switch to single-trace mode
   - Analyze specific interesting traces
   - Compare different architecture types

3. **Export Results**:
   - Download filtered datasets
   - Save trace details
   - Use in external tools (Excel, R, Python)

### Filtering & Exploration

- **Architecture Filter**: Focus on specific patterns
- **Agent Count Filter**: Study scalability
- **Metric Sorting**: Find extremes (highest/lowest)
- **Hover Details**: Quick information without clicking

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Streamlit installation
pip show streamlit

# Reinstall if needed
pip install streamlit --upgrade

# Try with verbose output
streamlit run dashboard.py --logger.level=debug
```

### Slow Performance

1. **Clear Cache**: Click "Clear Cache" in dashboard menu (top-right)
2. **Reduce Data**: Filter to smaller subset
3. **Close Tabs**: Keep only one dashboard tab open
4. **Restart**: Stop (Ctrl+C) and restart dashboard

### Visualization Issues

1. **Blank Plots**: Check browser console for errors
2. **Layout Problems**: Try different browser or zoom level
3. **Interactive Features**: Ensure JavaScript is enabled

### Data Loading Errors

```bash
# Check dataset access
python -c "from datasets import load_dataset; load_dataset('mcemri/MAST-Data')"

# Clear HuggingFace cache if needed
rm -rf ~/.cache/huggingface/datasets
```

## Advanced Features

### Session State

The dashboard maintains state across interactions:
- Analyzed dataset is cached
- Current trace selection is remembered
- Filters persist within session

### Keyboard Shortcuts

- **Ctrl+R**: Refresh dashboard
- **Ctrl+C**: Stop server (in terminal)
- **Ctrl+Shift+R**: Hard refresh (clear cache)

### URL Parameters

Access specific views directly:
```
http://localhost:8501/?mode=dataset
http://localhost:8501/?mode=trace&id=5
```

## Customization

### Modify Visualizations

Edit `dashboard.py` to customize:

```python
# Change color scheme
fig = px.bar(..., color_discrete_sequence=px.colors.qualitative.Set2)

# Adjust plot size
fig.update_layout(height=800)

# Modify hover information
fig.update_traces(hovertemplate='Custom: %{y}')
```

### Add New Metrics

1. Compute metric in analysis modules
2. Add to dashboard display:

```python
st.metric("My Metric", f"{value:.3f}")
```

3. Create visualization:

```python
fig = px.line(df, x="x", y="my_metric")
st.plotly_chart(fig)
```

### Theme Customization

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Deployment

### Local Network

```bash
# Find your IP
# macOS/Linux: ifconfig | grep inet
# Windows: ipconfig

# Run with network access
streamlit run dashboard.py --server.address 0.0.0.0

# Access from other devices
http://YOUR_IP:8501
```

### Cloud Deployment

#### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

#### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py"]
```

## Comparison: Dashboard vs CLI

| Feature | Dashboard | CLI (main.py) |
|---------|-----------|---------------|
| Interactivity | ✅ High | ❌ None |
| Real-time exploration | ✅ Yes | ❌ No |
| Batch processing | ❌ Limited | ✅ Yes |
| Automation | ❌ No | ✅ Yes |
| Visualization | ✅ Interactive | ✅ Static PNG |
| Export | ✅ Filtered | ✅ Complete |
| Learning curve | ✅ Easy | ⚠️ Moderate |
| Performance | ⚠️ Moderate | ✅ Fast |

**Recommendation**: 
- Use **Dashboard** for exploration and presentation
- Use **CLI** for batch processing and automation

## FAQ

**Q: Can I run both dashboard and CLI simultaneously?**
A: Yes, they're independent. Run CLI for batch processing while exploring in dashboard.

**Q: How do I save my analysis?**
A: Use download buttons in each tab. Results are saved as CSV/JSON.

**Q: Can I analyze custom datasets?**
A: Yes, modify `load_data()` function to load your data format.

**Q: Is the dashboard secure?**
A: For local use, yes. For public deployment, add authentication.

**Q: Can I embed visualizations?**
A: Yes, use Plotly's export features or screenshot functionality.

**Q: How do I update the dashboard?**
A: Pull latest code and restart: `git pull && streamlit run dashboard.py`

## Support

For issues or questions:
1. Check this guide
2. Review error messages in terminal
3. Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
4. Open GitHub issue

## Next Steps

1. **Explore**: Try both analysis modes
2. **Customize**: Modify visualizations for your needs
3. **Share**: Deploy for team access
4. **Extend**: Add custom metrics and views

Happy analyzing! 🚀