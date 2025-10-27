#!/usr/bin/env python3
"""
ReCoN Real-time Audio Recognition Streamlit App

This is a web-based version that demonstrates the ReCoN network
and can work without audio hardware by using simulated audio data.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import after path modification
from recon_core.compiler import compile_from_file  # type: ignore
from recon_core.config import EngineConfig  # type: ignore
from recon_core.engine import Engine  # type: ignore
from recon_core.enums import State, UnitType  # type: ignore

# Page configuration
st.set_page_config(
    page_title="ReCoN Real-time Audio Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "recon_graph" not in st.session_state:
    st.session_state.recon_graph = None
if "recon_engine" not in st.session_state:
    st.session_state.recon_engine = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "activation_history" not in st.session_state:
    st.session_state.activation_history = []
if "audio_features" not in st.session_state:
    st.session_state.audio_features = {
        "mfcc_low": 0.0,
        "pitch_high": 0.0,
        "rhythm": 0.0,
        "noise_level": 0.0,
        "formant": 0.0,
        "spectrogram": 0.0,
    }


def setup_recon_network():
    """Setup the ReCoN network."""
    if st.session_state.recon_graph is None:
        try:
            # Compile the graph
            yaml_path = os.path.join(
                project_root, "scripts", "engage_active_perception.yaml"
            )
            st.session_state.recon_graph = compile_from_file(yaml_path)

            # Create engine with configuration
            config = EngineConfig(
                sur_positive=0.4,
                por_positive=0.6,
                ret_positive=0.2,
                confirmation_ratio=0.75,
                deterministic_order=True,
                ret_feedback_enabled=True,
            )

            st.session_state.recon_engine = Engine(st.session_state.recon_graph, config)

            st.success("✓ ReCoN network initialized successfully")

        except Exception as e:
            st.error(f"Failed to initialize ReCoN network: {e}")
            return False

    return True


def generate_simulated_audio():
    """Generate simulated audio features."""
    t = time.time()

    # Create some interesting patterns
    st.session_state.audio_features = {
        "mfcc_low": 0.3 + 0.2 * np.sin(t * 0.5),
        "pitch_high": 0.4 + 0.3 * np.sin(t * 0.8),
        "rhythm": 0.5 + 0.2 * np.sin(t * 1.2),
        "noise_level": 0.2 + 0.1 * np.sin(t * 2.0),
        "formant": 0.6 + 0.2 * np.sin(t * 0.3),
        "spectrogram": 0.4 + 0.3 * np.sin(t * 0.7),
    }

    # Normalize features
    for key in st.session_state.audio_features:
        st.session_state.audio_features[key] = np.clip(
            st.session_state.audio_features[key], 0.0, 1.0
        )


def update_recon_network():
    """Update the ReCoN network with audio features."""
    try:
        # Map audio features to terminal units
        terminal_mapping = {
            "t_mfcc_low": st.session_state.audio_features.get("mfcc_low", 0.0),
            "t_pitch_high": st.session_state.audio_features.get("pitch_high", 0.0),
            "t_rhythm": st.session_state.audio_features.get("rhythm", 0.0),
            "t_noise_level": st.session_state.audio_features.get("noise_level", 0.0),
            "t_formant": st.session_state.audio_features.get("formant", 0.0),
            "t_spectrogram": st.session_state.audio_features.get("spectrogram", 0.0),
        }

        # Update terminal activations
        for terminal_id, activation in terminal_mapping.items():
            if terminal_id in st.session_state.recon_graph.units:
                unit = st.session_state.recon_graph.units[terminal_id]
                unit.a = activation

                # Set state based on activation
                if activation > unit.thresh:
                    unit.state = State.TRUE
                else:
                    unit.state = State.INACTIVE

        # Activate phrase if not already active
        if st.session_state.recon_graph.units["u_phrase"].state == State.INACTIVE:
            st.session_state.recon_graph.units["u_phrase"].a = 1.0
            st.session_state.recon_graph.units["u_phrase"].state = State.ACTIVE

    except Exception as e:
        st.error(f"Error updating ReCoN network: {e}")


def step_network():
    """Perform a single network step."""
    if st.session_state.recon_engine:
        # Generate simulated audio
        generate_simulated_audio()

        # Update ReCoN network
        update_recon_network()

        # Step the engine
        st.session_state.recon_engine.step(1)
        st.session_state.step_count += 1

        # Record activation history
        active_units = [
            u
            for u in st.session_state.recon_graph.units.values()
            if u.state in [State.ACTIVE, State.TRUE, State.CONFIRMED]
        ]
        confirmed_units = [
            u
            for u in st.session_state.recon_graph.units.values()
            if u.state == State.CONFIRMED
        ]

        st.session_state.activation_history.append(
            {
                "step": st.session_state.step_count,
                "active_count": len(active_units),
                "confirmed_count": len(confirmed_units),
                "phrase_activation": st.session_state.recon_graph.units["u_phrase"].a,
                "timestamp": time.time(),
            }
        )

        # Keep only last 100 steps
        if len(st.session_state.activation_history) > 100:
            st.session_state.activation_history = st.session_state.activation_history[
                -100:
            ]


def reset_network():
    """Reset the ReCoN network."""
    if st.session_state.recon_engine:
        st.session_state.recon_engine.reset()

        # Reset all units
        for unit in st.session_state.recon_graph.units.values():
            unit.state = State.INACTIVE
            unit.a = 0.0
            unit.inbox = []
            unit.outbox = []

        # Reset simulation state
        st.session_state.step_count = 0
        st.session_state.activation_history = []
        st.session_state.is_running = False


def create_network_graph():
    """Create a network visualization using Plotly."""
    if not st.session_state.recon_graph:
        return None

    # Create NetworkX graph
    import networkx as nx

    G = nx.DiGraph()

    # Add units
    for unit_id, unit in st.session_state.recon_graph.units.items():
        G.add_node(
            unit_id,
            kind=unit.kind.name,
            state=unit.state.name,
            activation=unit.a,
            threshold=unit.thresh,
        )

    # Add edges
    for src_id, edges in st.session_state.recon_graph.out_edges.items():
        for edge in edges:
            G.add_edge(edge.src, edge.dst, type=edge.type.name, weight=edge.w)

    # Create layout
    pos = create_network_layout()

    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Get edge info
        edge_data = G[edge[0]][edge[1]]
        edge_info.append(
            f"Type: {edge_data['type']}<br>Weight: {edge_data['weight']:.2f}"
        )

    # Create edge traces by type
    edge_styles = {
        "SUB": {"color": "#2E8B57", "width": 3},
        "SUR": {"color": "#DC143C", "width": 2},
        "POR": {"color": "#8A2BE2", "width": 3, "dash": "dash"},
        "RET": {"color": "#FF8C00", "width": 2, "dash": "dot"},
    }

    edge_traces = []
    for link_type, style in edge_styles.items():
        edges_of_type = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("type") == link_type
        ]
        if edges_of_type:
            edge_x_type = []
            edge_y_type = []
            for edge in edges_of_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x_type.extend([x0, x1, None])
                edge_y_type.extend([y0, y1, None])

            edge_traces.append(
                go.Scatter(
                    x=edge_x_type,
                    y=edge_y_type,
                    line=dict(
                        width=style["width"],
                        color=style["color"],
                        dash=style.get("dash", "solid"),
                    ),
                    hoverinfo="none",
                    mode="lines",
                    name=f"{link_type} links",
                    showlegend=True,
                )
            )

    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for unit_id in G.nodes():
        unit = st.session_state.recon_graph.units[unit_id]
        x, y = pos[unit_id]
        node_x.append(x)
        node_y.append(y)

        # Create node text
        if unit.kind == UnitType.SCRIPT:
            node_text.append(f"{unit_id}<br>{unit.state.name}<br>a={unit.a:.2f}")
        else:
            node_text.append(f"{unit_id}<br>a={unit.a:.2f}")

        # Color based on state
        if unit.state == State.CONFIRMED:
            color = "#2CA25F"  # Green
            size = 20
        elif unit.state == State.ACTIVE:
            color = "#4A90E2"  # Blue
            size = 18
        elif unit.state == State.TRUE:
            color = "#31A354"  # Dark green
            size = 16
        elif unit.state == State.REQUESTED:
            color = "#6BAED6"  # Light blue
            size = 14
        else:
            color = "#CCCCCC"  # Gray
            size = 12

        node_colors.append(color)
        node_sizes.append(size)

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        marker=dict(
            size=node_sizes, color=node_colors, line=dict(width=2, color="black")
        ),
        name="Units",
        showlegend=True,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(
            text="ReCoN Network: 'Engage Active Perception'",
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="ReCoN Network Visualization",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="black", size=12),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )

    return fig


def create_network_layout():
    """Create layout for the network."""
    pos = {}

    # Phrase level (top)
    pos["u_phrase"] = (0, 3)

    # Word level (second row)
    pos["u_engage"] = (-2, 2)
    pos["u_active"] = (0, 2)
    pos["u_perception"] = (2, 2)

    # Phoneme level (third row)
    pos["u_en_phoneme"] = (-2.5, 1)
    pos["u_gei_phoneme"] = (-2, 1)
    pos["u_dj_phoneme"] = (-1.5, 1)
    pos["u_ak_phoneme"] = (-0.5, 1)
    pos["u_ti_phoneme"] = (0, 1)
    pos["u_v_phoneme"] = (0.5, 1)
    pos["u_per_phoneme"] = (1.5, 1)
    pos["u_sep_phoneme"] = (2, 1)
    pos["u_shun_phoneme"] = (2.5, 1)

    # Terminal level (bottom)
    terminals = [
        uid
        for uid, u in st.session_state.recon_graph.units.items()
        if u.kind == UnitType.TERMINAL
    ]
    for i, term_id in enumerate(terminals):
        x = -3 + (i * 6) / (len(terminals) - 1) if len(terminals) > 1 else 0
        pos[term_id] = (x, 0)

    return pos


def create_audio_features_chart():
    """Create audio features bar chart."""
    features = list(st.session_state.audio_features.keys())
    values = list(st.session_state.audio_features.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=features,
                y=values,
                marker_color=[
                    "#FF6B6B",
                    "#4ECDC4",
                    "#45B7D1",
                    "#96CEB4",
                    "#FECA57",
                    "#FF9FF3",
                ],
            )
        ]
    )

    fig.update_layout(
        title="Real-time Audio Features",
        xaxis_title="Feature",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    return fig


def create_activation_timeline():
    """Create activation timeline chart."""
    if not st.session_state.activation_history:
        return None

    df = pd.DataFrame(st.session_state.activation_history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["active_count"],
            mode="lines+markers",
            name="Active Units",
            line=dict(color="#4A90E2", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["confirmed_count"],
            mode="lines+markers",
            name="Confirmed Units",
            line=dict(color="#2CA25F", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["phrase_activation"],
            mode="lines+markers",
            name="Phrase Activation",
            line=dict(color="#FF6B6B", width=3),
        )
    )

    fig.update_layout(
        title="Activation Timeline",
        xaxis_title="Step",
        yaxis_title="Count/Activation",
        height=400,
        hovermode="x unified",
    )

    return fig


def main():
    """Main Streamlit app."""

    # Header
    st.title("🎵 ReCoN Real-time Audio Recognition")
    st.subheader("'Engage Active Perception' Hypothesis")

    # Initialize network
    if not setup_recon_network():
        st.stop()

    # Sidebar controls
    st.sidebar.header("Controls")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Start Simulation", disabled=st.session_state.is_running):
            st.session_state.is_running = True
            st.rerun()

    with col2:
        if st.button("Stop Simulation", disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            st.rerun()

    if st.sidebar.button("Reset Network"):
        reset_network()
        st.rerun()

    if st.sidebar.button("Single Step"):
        step_network()
        st.rerun()

    # Simulation speed
    speed = st.sidebar.slider("Simulation Speed", 0.1, 5.0, 1.0, 0.1)

    # Feature intensity
    intensity = st.sidebar.slider("Feature Intensity", 0.0, 1.0, 0.5, 0.1)

    # Apply intensity to features
    for key in st.session_state.audio_features:
        st.session_state.audio_features[key] *= intensity

    # Auto-step if running
    if st.session_state.is_running:
        step_network()
        time.sleep(1.0 / speed)
        st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Network visualization
        st.header("ReCoN Network")
        network_fig = create_network_graph()
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("Network not initialized")

    with col2:
        # Status panel
        st.header("Status")

        st.metric("Step Count", st.session_state.step_count)

        if st.session_state.recon_graph:
            active_units = [
                u
                for u in st.session_state.recon_graph.units.values()
                if u.state in [State.ACTIVE, State.TRUE, State.CONFIRMED]
            ]
            confirmed_units = [
                u
                for u in st.session_state.recon_graph.units.values()
                if u.state == State.CONFIRMED
            ]

            st.metric("Active Units", len(active_units))
            st.metric("Confirmed Units", len(confirmed_units))

            phrase_activation = st.session_state.recon_graph.units["u_phrase"].a
            st.metric("Phrase Activation", f"{phrase_activation:.3f}")

        # Audio features
        st.header("Audio Features")
        audio_fig = create_audio_features_chart()
        st.plotly_chart(audio_fig, use_container_width=True)

    # Timeline
    st.header("Activation Timeline")
    timeline_fig = create_activation_timeline()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("No activation history yet. Start the simulation to see the timeline.")

    # Network information
    with st.expander("Network Information"):
        if st.session_state.recon_graph:
            st.write(f"**Total Units:** {len(st.session_state.recon_graph.units)}")

            # Count units by type
            script_units = [
                u
                for u in st.session_state.recon_graph.units.values()
                if u.kind == UnitType.SCRIPT
            ]
            terminal_units = [
                u
                for u in st.session_state.recon_graph.units.values()
                if u.kind == UnitType.TERMINAL
            ]

            st.write(f"**Script Units:** {len(script_units)}")
            st.write(f"**Terminal Units:** {len(terminal_units)}")

            # Count edges by type
            link_counts = {}
            for edges in st.session_state.recon_graph.out_edges.values():
                for edge in edges:
                    link_type = edge.type.name
                    link_counts[link_type] = link_counts.get(link_type, 0) + 1

            st.write("**Link Types:**")
            for link_type, count in sorted(link_counts.items()):
                st.write(f"- {link_type}: {count}")

            st.write("**Link Descriptions:**")
            st.write("- **SUB (Subordinate):** Evidence propagation (bottom-up)")
            st.write("- **SUR (Superior):** Request propagation (top-down)")
            st.write("- **POR (Precedence):** Temporal sequencing")
            st.write("- **RET (Return):** Temporal feedback")


if __name__ == "__main__":
    main()
