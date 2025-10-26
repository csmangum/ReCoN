#!/usr/bin/env python3
"""
Comprehensive ReCoN Graph Visualization for "Engage Active Perception" Hypothesis

This script generates a detailed visualization of the ReCoN network showing:
- Full hierarchical structure from phrase → words → phonemes → terminals
- Word-level to syllable-level hypothesis activation sequence
- Link types and their roles in the recognition process
- Terminal-level audio feature detection
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches

# Add project root to Python path
sys.path.insert(0, '/workspace')

from recon_core.compiler import compile_from_file
from recon_core.enums import LinkType, UnitType, State
from recon_core.engine import Engine
from recon_core.config import EngineConfig

def create_comprehensive_visualization():
    """Create a comprehensive visualization of the Engage Active Perception ReCoN graph."""
    
    # Compile the graph from YAML
    print("Compiling ReCoN graph from engage_active_perception.yaml...")
    graph = compile_from_file('/workspace/scripts/engage_active_perception.yaml')
    
    print(f"Graph compiled: {len(graph.units)} units, {sum(len(edges) for edges in graph.out_edges.values())} edges")
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.2)
    
    # Main graph visualization
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Create NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Add all units to NetworkX graph
    for unit_id, unit in graph.units.items():
        G.add_node(unit_id, 
                  kind=unit.kind.name,
                  state=unit.state.name,
                  activation=unit.a,
                  threshold=unit.thresh)
    
    # Add all edges
    for src_id, edges in graph.out_edges.items():
        for edge in edges:
            G.add_edge(edge.src, edge.dst, 
                      type=edge.type.name, 
                      weight=edge.w)
    
    # Define hierarchical layout positions
    pos = create_hierarchical_layout(graph)
    
    # Define colors and styles
    node_colors, node_sizes = get_node_styling(graph)
    edge_styles = get_edge_styling()
    
    # Draw the graph
    draw_recon_graph(ax_main, G, pos, node_colors, node_sizes, edge_styles, graph)
    
    # Add legend and annotations
    add_legend_and_annotations(fig, gs)
    
    # Add activation sequence diagram
    ax_sequence = fig.add_subplot(gs[0, :])
    create_activation_sequence_diagram(ax_sequence, graph)
    
    # Add detailed unit information
    ax_info = fig.add_subplot(gs[2, 0])
    create_unit_information_panel(ax_info, graph)
    
    # Add link type explanation
    ax_links = fig.add_subplot(gs[1, 1])
    create_link_type_explanation(ax_links, graph)
    
    # Add terminal features explanation
    ax_terminals = fig.add_subplot(gs[2, 1])
    create_terminal_features_explanation(ax_terminals, graph)
    
    plt.suptitle('ReCoN Graph: "Engage Active Perception" - Word-Level to Syllable-Level Hypothesis Activation', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig('/workspace/engage_active_perception_full_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return graph

def create_hierarchical_layout(graph):
    """Create a hierarchical layout for the ReCoN graph."""
    pos = {}
    
    # Phrase level (top)
    pos['u_phrase'] = (0, 4)
    
    # Word level (second row)
    word_y = 3
    pos['u_engage'] = (-2, word_y)
    pos['u_active'] = (0, word_y)
    pos['u_perception'] = (2, word_y)
    
    # Phoneme level (third row) - grouped by word
    phoneme_y = 2
    
    # Engage phonemes
    pos['u_en_phoneme'] = (-2.5, phoneme_y)
    pos['u_gei_phoneme'] = (-2, phoneme_y)
    pos['u_dj_phoneme'] = (-1.5, phoneme_y)
    
    # Active phonemes
    pos['u_ak_phoneme'] = (-0.5, phoneme_y)
    pos['u_ti_phoneme'] = (0, phoneme_y)
    pos['u_v_phoneme'] = (0.5, phoneme_y)
    
    # Perception phonemes
    pos['u_per_phoneme'] = (1.5, phoneme_y)
    pos['u_sep_phoneme'] = (2, phoneme_y)
    pos['u_shun_phoneme'] = (2.5, phoneme_y)
    
    # Terminal level (bottom) - audio features
    terminal_y = 1
    terminals = [uid for uid, u in graph.units.items() if u.kind == UnitType.TERMINAL]
    
    # Distribute terminals evenly
    for i, term_id in enumerate(terminals):
        x = -3 + (i * 6) / (len(terminals) - 1) if len(terminals) > 1 else 0
        pos[term_id] = (x, terminal_y)
    
    return pos

def get_node_styling(graph):
    """Get node colors and sizes based on unit types and states."""
    node_colors = []
    node_sizes = []
    
    # Color mapping
    type_colors = {
        'SCRIPT': '#4A90E2',  # Blue for scripts
        'TERMINAL': '#7ED321'  # Green for terminals
    }
    
    # Size mapping
    type_sizes = {
        'SCRIPT': 800,
        'TERMINAL': 400
    }
    
    for unit_id, unit in graph.units.items():
        node_colors.append(type_colors[unit.kind.name])
        node_sizes.append(type_sizes[unit.kind.name])
    
    return node_colors, node_sizes

def get_edge_styling():
    """Get edge styling for different link types."""
    return {
        'SUB': {'color': '#2E8B57', 'style': 'solid', 'width': 2, 'alpha': 0.8},  # Green for evidence
        'SUR': {'color': '#DC143C', 'style': 'solid', 'width': 2, 'alpha': 0.8},  # Red for requests
        'POR': {'color': '#8A2BE2', 'style': 'dashed', 'width': 2, 'alpha': 0.6},  # Purple for precedence
        'RET': {'color': '#FF8C00', 'style': 'dotted', 'width': 2, 'alpha': 0.6}   # Orange for return
    }

def draw_recon_graph(ax, G, pos, node_colors, node_sizes, edge_styles, graph):
    """Draw the ReCoN graph with proper styling."""
    
    # Draw edges by type
    for link_type, style in edge_styles.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == link_type]
        if edges_of_type:
            nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                 edge_color=style['color'],
                                 style=style['style'],
                                 width=style['width'],
                                 alpha=style['alpha'],
                                 arrows=True,
                                 arrowsize=20,
                                 ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, ax=ax)
    
    # Draw labels with better positioning
    labels = {}
    for node in G.nodes():
        unit = graph.units[node]
        if unit.kind == UnitType.SCRIPT:
            # Show unit name and threshold
            labels[node] = f"{node}\n(thresh={unit.thresh})"
        else:
            # Show terminal name and frequency range if available
            freq_range = unit.meta.get('freq_range', '')
            labels[node] = f"{node}\n{freq_range}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Add edge labels for weights
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0)
        if abs(weight) > 0.1:  # Only show significant weights
            edge_labels[(u, v)] = f"{weight:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
    
    ax.set_title('ReCoN Network Structure: Phrase → Words → Phonemes → Audio Features', 
                 fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')

def add_legend_and_annotations(fig, gs):
    """Add legend and annotations to the figure."""
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#4A90E2', label='Script Units (Hypotheses)'),
        mpatches.Patch(color='#7ED321', label='Terminal Units (Audio Features)'),
        mpatches.Patch(color='#2E8B57', label='SUB Links (Evidence Flow)'),
        mpatches.Patch(color='#DC143C', label='SUR Links (Request Flow)'),
        mpatches.Patch(color='#8A2BE2', label='POR Links (Temporal Precedence)'),
        mpatches.Patch(color='#FF8C00', label='RET Links (Temporal Feedback)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

def create_activation_sequence_diagram(ax, graph):
    """Create a diagram showing the activation sequence."""
    
    # Define the activation sequence
    sequence = [
        ("1. Phrase Activation", "u_phrase", "Root phrase 'Engage Active Perception' activated"),
        ("2. Word Requests", "u_engage, u_active, u_perception", "SUR links request word-level recognition"),
        ("3. Phoneme Requests", "All phoneme units", "Each word requests its constituent phonemes"),
        ("4. Terminal Requests", "All terminals", "Phonemes request audio feature detection"),
        ("5. Feature Detection", "Audio terminals", "Terminals detect MFCC, pitch, formant features"),
        ("6. Phoneme Confirmation", "Phoneme units", "Phonemes confirm based on terminal evidence"),
        ("7. Word Confirmation", "Word units", "Words confirm based on phoneme evidence"),
        ("8. Phrase Confirmation", "u_phrase", "Full phrase confirmed with temporal sequencing")
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(sequence))
    
    for i, (step, units, description) in enumerate(sequence):
        # Draw step box
        box = FancyBboxPatch((0.05, y_positions[i] - 0.05), 0.9, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='lightblue' if i % 2 == 0 else 'lightgray',
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)
        
        # Add text
        ax.text(0.1, y_positions[i], f"{step}: {description}", 
                fontsize=10, fontweight='bold', va='center')
        
        # Add arrow to next step
        if i < len(sequence) - 1:
            ax.arrow(0.5, y_positions[i] - 0.05, 0, -0.05, 
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Hypothesis Activation Sequence: Word-Level → Syllable-Level → Terminal-Level', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')

def create_unit_information_panel(ax, graph):
    """Create a panel showing detailed unit information."""
    
    # Group units by type
    scripts = [(uid, u) for uid, u in graph.units.items() if u.kind == UnitType.SCRIPT]
    terminals = [(uid, u) for uid, u in graph.units.items() if u.kind == UnitType.TERMINAL]
    
    y_pos = 0.9
    ax.text(0.05, y_pos, "Script Units (Hypotheses):", fontsize=12, fontweight='bold')
    y_pos -= 0.1
    
    for uid, unit in sorted(scripts):
        ax.text(0.1, y_pos, f"• {uid}: threshold={unit.thresh}", fontsize=10)
        y_pos -= 0.08
    
    y_pos -= 0.05
    ax.text(0.05, y_pos, "Terminal Units (Audio Features):", fontsize=12, fontweight='bold')
    y_pos -= 0.1
    
    for uid, unit in sorted(terminals):
        freq_range = unit.meta.get('freq_range', 'N/A')
        ax.text(0.1, y_pos, f"• {uid}: {freq_range}, threshold={unit.thresh}", fontsize=10)
        y_pos -= 0.08
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Unit Details', fontsize=12, fontweight='bold')
    ax.axis('off')

def create_link_type_explanation(ax, graph):
    """Create an explanation of link types."""
    
    explanations = [
        ("SUB (Subordinate)", "Evidence flow from children to parents", "#2E8B57"),
        ("SUR (Superior)", "Request flow from parents to children", "#DC143C"),
        ("POR (Precedence)", "Temporal sequencing between units", "#8A2BE2"),
        ("RET (Return)", "Temporal feedback between units", "#FF8C00")
    ]
    
    y_pos = 0.9
    ax.text(0.05, y_pos, "Link Types:", fontsize=12, fontweight='bold')
    y_pos -= 0.15
    
    for link_type, description, color in explanations:
        # Draw colored line
        ax.plot([0.05, 0.15], [y_pos, y_pos], color=color, linewidth=3)
        ax.text(0.2, y_pos, f"{link_type}: {description}", fontsize=10, va='center')
        y_pos -= 0.12
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Link Type Meanings', fontsize=12, fontweight='bold')
    ax.axis('off')

def create_terminal_features_explanation(ax, graph):
    """Create an explanation of terminal audio features."""
    
    features = [
        ("t_mfcc_low", "MFCC coefficients for low frequencies (0-500Hz)"),
        ("t_pitch_high", "Pitch detection for high frequencies (2000-5000Hz)"),
        ("t_rhythm", "Rhythmic pattern detection"),
        ("t_noise_level", "Background noise level (inhibitory)"),
        ("t_formant", "Formant frequency detection for vowels"),
        ("t_spectrogram", "Full spectrum analysis")
    ]
    
    y_pos = 0.9
    ax.text(0.05, y_pos, "Audio Feature Terminals:", fontsize=12, fontweight='bold')
    y_pos -= 0.1
    
    for feature, description in features:
        ax.text(0.05, y_pos, f"• {feature}:", fontsize=10, fontweight='bold')
        ax.text(0.1, y_pos - 0.03, description, fontsize=9, style='italic')
        y_pos -= 0.12
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Audio Feature Detection', fontsize=12, fontweight='bold')
    ax.axis('off')

def run_simulation_demo(graph):
    """Run a brief simulation to show activation flow."""
    
    print("\nRunning simulation demo...")
    
    # Create engine with configuration
    config = EngineConfig(
        sur_positive=0.4,
        por_positive=0.6,
        ret_positive=0.2,
        confirmation_ratio=0.75,
        deterministic_order=True,
        ret_feedback_enabled=True
    )
    
    engine = Engine(graph, config)
    
    # Activate the root phrase
    graph.units['u_phrase'].a = 1.0
    graph.units['u_phrase'].state = State.ACTIVE
    
    print("Initial state:")
    print(f"  u_phrase: {graph.units['u_phrase'].state.name} (a={graph.units['u_phrase'].a:.2f})")
    
    # Run simulation for several steps
    for step in range(1, 6):
        snapshot = engine.step(1)
        
        print(f"\nStep {step}:")
        # Show key unit states
        key_units = ['u_phrase', 'u_engage', 'u_active', 'u_perception']
        for unit_id in key_units:
            if unit_id in snapshot['units']:
                unit_data = snapshot['units'][unit_id]
                print(f"  {unit_id}: {unit_data['state']} (a={unit_data['a']:.2f})")
        
        # Show some terminal activations
        terminals = [uid for uid, data in snapshot['units'].items() 
                    if data['kind'] == 'TERMINAL' and data['a'] > 0.1]
        if terminals:
            print(f"  Active terminals: {terminals}")

if __name__ == "__main__":
    print("Creating comprehensive ReCoN graph visualization for 'Engage Active Perception'...")
    
    # Create the visualization
    graph = create_comprehensive_visualization()
    
    # Run a simulation demo
    run_simulation_demo(graph)
    
    print(f"\nVisualization saved as: engage_active_perception_full_graph.png")
    print("The graph shows the complete hierarchical structure from phrase-level")
    print("hypotheses down to syllable-level phonemes and terminal audio features.")