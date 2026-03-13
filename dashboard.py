"""
Interactive Dashboard for MAS Analysis Pipeline

This script creates an interactive web dashboard using Streamlit for visualizing
both dataset-level and single-trace analysis results.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_mast_dataset
from src.dataset_analyzer import analyze_dataset
from src.single_trace_analyzer import analyze_single_trace
from src.trace_parser import parse_agent_turns


# Page configuration
st.set_page_config(
    page_title="MAS Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #262730;
    }
    .metric-card strong {
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .metric-card small {
        color: #555555;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the MAST dataset."""
    with st.spinner("Loading MAST dataset..."):
        df = load_mast_dataset()
    return df


@st.cache_data
def analyze_full_dataset(_df):
    """Analyze and cache dataset-level results."""
    with st.spinner("Analyzing dataset... This may take a few minutes."):
        results_df = analyze_dataset(_df)
    return results_df


def plot_architecture_distribution(results_df):
    """Create interactive architecture distribution plot."""
    arch_counts = results_df["architecture_type"].value_counts().reset_index()
    arch_counts.columns = ["Architecture", "Count"]
    
    fig = px.bar(
        arch_counts,
        x="Architecture",
        y="Count",
        title="MAS Architecture Distribution",
        color="Architecture",
        text="Count"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=500)
    return fig


def plot_metric_distribution(results_df, metric, title):
    """Create interactive metric distribution plot."""
    fig = px.histogram(
        results_df,
        x=metric,
        nbins=30,
        title=title,
        marginal="box"
    )
    fig.add_vline(
        x=results_df[metric].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {results_df[metric].mean():.3f}"
    )
    fig.update_layout(height=400)
    return fig


def plot_metric_by_architecture(results_df, metric, title):
    """Create box plot of metric by architecture."""
    fig = px.box(
        results_df,
        x="architecture_type",
        y=metric,
        title=title,
        color="architecture_type",
        points="all"
    )
    fig.update_layout(showlegend=False, height=500)
    fig.update_xaxes(tickangle=45)
    return fig


def plot_correlation_heatmap(results_df):
    """Create correlation heatmap."""
    numeric_cols = [
        "num_agents", "num_edges", "graph_density", "avg_degree",
        "clustering_coefficient", "num_cycles", "loop_index",
        "agent_dependency_ratio", "communication_entropy"
    ]
    numeric_cols = [col for col in numeric_cols if col in results_df.columns]
    
    corr_matrix = results_df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Correlation Heatmap of Metrics",
        height=600,
        xaxis_tickangle=45
    )
    return fig


def plot_scatter_matrix(results_df):
    """Create scatter matrix for key metrics."""
    metrics = ["num_agents", "graph_density", "loop_index", "agent_dependency_ratio"]
    metrics = [m for m in metrics if m in results_df.columns]
    
    fig = px.scatter_matrix(
        results_df,
        dimensions=metrics,
        color="architecture_type",
        title="Scatter Matrix of Key Metrics"
    )
    fig.update_layout(height=800)
    return fig


def plot_communication_graph_interactive(analysis):
    """Create interactive communication graph using Plotly."""
    G = analysis["graph_data"]["graph"]
    
    if G.number_of_nodes() == 0:
        return None
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]].get('weight', 1)
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight * 2, color='#888'),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}<br>Weight: {weight}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(f"{node}<br>Degree: {degree}")
        node_size.append(20 + degree * 10)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=f"Communication Graph - Trace {analysis['task_id']}",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig


def plot_agent_timeline_interactive(analysis):
    """Create interactive agent timeline."""
    turns = analysis["turns"]
    
    if not turns:
        return None
    
    df = pd.DataFrame([
        {
            "Turn": t.get("turn", 0),
            "Agent": t.get("agent", "unknown"),
            "Content_Length": len(t.get("content", ""))
        }
        for t in turns
    ])
    
    fig = px.scatter(
        df,
        x="Turn",
        y="Agent",
        size="Content_Length",
        color="Agent",
        title=f"Agent Activity Timeline - Trace {analysis['task_id']}",
        hover_data=["Content_Length"]
    )
    
    fig.update_layout(height=400)
    return fig


def plot_adjacency_matrix_interactive(analysis):
    """Create interactive adjacency matrix heatmap."""
    adjacency_matrix = analysis["graph_data"]["adjacency_matrix"]
    agents = analysis["graph_data"]["agents"]
    
    if adjacency_matrix.size == 0:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=adjacency_matrix,
        x=agents,
        y=agents,
        colorscale="YlOrRd",
        text=adjacency_matrix,
        texttemplate='%{text:.0f}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f"Adjacency Matrix - Trace {analysis['task_id']}",
        xaxis_title="To Agent",
        yaxis_title="From Agent",
        height=500
    )
    
    return fig


def plot_agent_participation_interactive(analysis):
    """Create interactive agent participation chart."""
    agent_stats = analysis["agent_statistics"]
    
    if not agent_stats or "agent_turn_counts" not in agent_stats:
        return None
    
    df = pd.DataFrame([
        {"Agent": agent, "Turns": count}
        for agent, count in agent_stats["agent_turn_counts"].items()
    ])
    
    fig = px.bar(
        df,
        x="Agent",
        y="Turns",
        title=f"Agent Participation - Trace {analysis['task_id']}",
        color="Agent",
        text="Turns"
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    return fig


def dataset_level_view(df, results_df):
    """Render dataset-level analysis view."""
    st.markdown('<p class="main-header">📊 Dataset-Level Analysis</p>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Traces", len(results_df))
    with col2:
        st.metric("Avg Agents/Trace", f"{results_df['num_agents'].mean():.1f}")
    with col3:
        st.metric("Avg Graph Density", f"{results_df['graph_density'].mean():.3f}")
    with col4:
        st.metric("Avg Loop Index", f"{results_df['loop_index'].mean():.3f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Architecture Analysis",
        "📊 Metric Distributions",
        "🔗 Correlations",
        "📋 Data Table"
    ])
    
    with tab1:
        st.subheader("Architecture Distribution")
        fig = plot_architecture_distribution(results_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Architecture Statistics")
        arch_stats = results_df.groupby("architecture_type").agg({
            "num_agents": "mean",
            "graph_density": "mean",
            "loop_index": "mean",
            "agent_dependency_ratio": "mean",
            "communication_entropy": "mean"
        }).round(3)
        st.dataframe(arch_stats, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_metric_distribution(results_df, "loop_index", "Loop Index Distribution"),
                use_container_width=True
            )
            st.plotly_chart(
                plot_metric_distribution(results_df, "communication_entropy", "Communication Entropy Distribution"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_metric_distribution(results_df, "agent_dependency_ratio", "Agent Dependency Ratio Distribution"),
                use_container_width=True
            )
            st.plotly_chart(
                plot_metric_distribution(results_df, "graph_density", "Graph Density Distribution"),
                use_container_width=True
            )
        
        st.subheader("Metrics by Architecture")
        metric_choice = st.selectbox(
            "Select metric to compare",
            ["loop_index", "agent_dependency_ratio", "communication_entropy", "graph_density"]
        )
        fig = plot_metric_by_architecture(results_df, metric_choice, f"{metric_choice.replace('_', ' ').title()} by Architecture")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        fig = plot_correlation_heatmap(results_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Scatter Matrix")
        fig = plot_scatter_matrix(results_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Complete Results Table")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            arch_filter = st.multiselect(
                "Filter by Architecture",
                options=results_df["architecture_type"].unique(),
                default=results_df["architecture_type"].unique()
            )
        with col2:
            min_agents = st.slider(
                "Minimum number of agents",
                min_value=int(results_df["num_agents"].min()),
                max_value=int(results_df["num_agents"].max()),
                value=int(results_df["num_agents"].min())
            )
        
        filtered_df = results_df[
            (results_df["architecture_type"].isin(arch_filter)) &
            (results_df["num_agents"] >= min_agents)
        ]
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="mas_analysis_filtered.csv",
            mime="text/csv"
        )


def single_trace_view(df):
    """Render single-trace analysis view."""
    st.markdown('<p class="main-header">🔍 Single Trace Analysis</p>', unsafe_allow_html=True)
    
    # Trace selection
    col1, col2 = st.columns([3, 1])
    with col1:
        task_id = st.number_input(
            "Enter Trace ID",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )
    with col2:
        analyze_button = st.button("🔍 Analyze Trace", type="primary")
    
    if analyze_button or 'current_trace_id' in st.session_state:
        if analyze_button:
            st.session_state.current_trace_id = task_id
        
        task_id = st.session_state.current_trace_id
        
        with st.spinner(f"Analyzing trace {task_id}..."):
            analysis = analyze_single_trace(df, task_id)
        
        # Summary metrics
        st.subheader("📋 Trace Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Trace ID", task_id)
        with col2:
            st.metric("Agents", len(analysis["trace_info"]["agents"]))
        with col3:
            st.metric("Turns", analysis["trace_info"]["num_turns"])
        with col4:
            st.metric("Architecture", analysis["architecture"]["architecture_type"])
        with col5:
            st.metric("Loop Index", f"{analysis['metrics']['loop_index']:.3f}")
        
        st.markdown("---")
        
        # Display Agent Roles
        st.subheader("👥 Agent Roles")
        trace_data = df.iloc[task_id]
        if 'agent_roles' in trace_data and trace_data['agent_roles']:
            agent_roles = trace_data['agent_roles']
            
            # Create columns for agent roles
            num_cols = min(len(agent_roles), 3)
            cols = st.columns(num_cols)
            
            for idx, (normalized, full_name) in enumerate(sorted(agent_roles.items())):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{normalized}</strong><br/>
                        <small>{full_name}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Agent role information not available for this trace")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🕸️ Communication Graph",
            "📊 Metrics & Statistics",
            "📈 Timeline & Participation",
            "📝 Turn Details"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Communication Graph")
                fig = plot_communication_graph_interactive(analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Empty graph")
            
            with col2:
                st.subheader("Adjacency Matrix")
                fig = plot_adjacency_matrix_interactive(analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Empty adjacency matrix")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Coordination Metrics")
                metrics = analysis["metrics"]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">Loop Index (LI)</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #262730; margin: 0.5rem 0;">{metrics['loop_index']:.4f}</p>
                    <p style="color: #555555; margin: 0;">Measures interaction loops</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">Agent Dependency Ratio (ADR)</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #262730; margin: 0.5rem 0;">{metrics['agent_dependency_ratio']:.4f}</p>
                    <p style="color: #555555; margin: 0;">Measures centralization</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">Communication Entropy (CE)</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #262730; margin: 0.5rem 0;">{metrics['communication_entropy']:.4f}</p>
                    <p style="color: #555555; margin: 0;">Measures unpredictability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Architecture Analysis")
                arch = analysis["architecture"]
                
                st.info(f"**Type:** {arch['architecture_type']}")
                st.write(f"**Description:** {arch['description']}")
                
                if "centralization" in arch["properties"]:
                    cent = arch["properties"]["centralization"]
                    st.write(f"**Max Centrality:** {cent['max']:.3f}")
                    st.write(f"**Avg Centrality:** {cent['avg']:.3f}")
                
                st.subheader("Detected Roles")
                roles = analysis["roles"]
                active_roles = [role for role, present in roles.items() if present]
                if active_roles:
                    st.write(", ".join(active_roles))
                else:
                    st.write("No specific roles detected")
                
                st.subheader("Graph Features")
                features = analysis["graph_features"]
                st.write(f"**Edges:** {features['number_of_edges']}")
                st.write(f"**Density:** {features['graph_density']:.3f}")
                st.write(f"**Avg Degree:** {features['average_degree']:.3f}")
                st.write(f"**Cycles:** {features['number_of_cycles']}")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Agent Timeline")
                fig = plot_agent_timeline_interactive(analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Agent Participation")
                fig = plot_agent_participation_interactive(analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Agent Statistics")
            agent_stats = analysis["agent_statistics"]
            if agent_stats:
                stats_df = pd.DataFrame([
                    {
                        "Agent": agent,
                        "Turns": count,
                        "Participation %": f"{agent_stats['participation_ratios'][agent]*100:.1f}%"
                    }
                    for agent, count in agent_stats["agent_turn_counts"].items()
                ]).sort_values("Turns", ascending=False)
                
                st.dataframe(stats_df, use_container_width=True)
        
        with tab4:
            st.subheader("Turn-by-Turn Breakdown")
            
            turns_df = pd.DataFrame([
                {
                    "Turn": t.get("turn", 0),
                    "Agent": t.get("agent", "unknown"),
                    "Content Length": len(t.get("content", "")),
                    "Content Preview": t.get("content", "")[:200] + "..." if len(t.get("content", "")) > 200 else t.get("content", "")
                }
                for t in analysis["turns"]
            ])
            
            st.dataframe(turns_df, use_container_width=True, height=400)
            
            # Download button
            csv = turns_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Turn Details as CSV",
                data=csv,
                file_name=f"trace_{task_id}_turns.csv",
                mime="text/csv"
            )


def main():
    """Main dashboard application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=MAS", width=150)
        st.title("MAS Analysis Dashboard")
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Select Analysis Mode",
            ["📊 Dataset-Level Analysis", "🔍 Single Trace Analysis"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        ### About
        This dashboard provides interactive visualization for Multi-Agent System (MAS) interaction traces.
        
        **Features:**
        - Dataset-level aggregate analysis
        - Single-trace detailed analysis
        - Interactive visualizations
        - Downloadable results
        
        **Metrics:**
        - Loop Index (LI)
        - Agent Dependency Ratio (ADR)
        - Communication Entropy (CE)
        - Graph features
        """)
    
    # Load data
    try:
        df = load_data()
        
        if mode == "📊 Dataset-Level Analysis":
            # Check if we need to analyze
            if 'results_df' not in st.session_state:
                st.session_state.results_df = analyze_full_dataset(df)
            
            dataset_level_view(df, st.session_state.results_df)
        
        else:  # Single Trace Analysis
            single_trace_view(df)
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()

# Made with Bob
