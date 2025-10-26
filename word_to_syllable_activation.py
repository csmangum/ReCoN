#!/usr/bin/env python3
"""
Word-to-Syllable Activation Sequence Visualization

This script creates a focused visualization showing how the "Engage Active Perception" 
hypothesis activates sequentially from word-level to syllable-level (phoneme-level) 
hypotheses when confirmed at the terminal level.
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.animation as animation

# Add project root to Python path
sys.path.insert(0, '/workspace')

from recon_core.compiler import compile_from_file
from recon_core.enums import LinkType, UnitType, State
from recon_core.engine import Engine
from recon_core.config import EngineConfig

def create_activation_sequence_visualization():
    """Create a focused visualization of the word-to-syllable activation sequence."""
    
    # Compile the graph
    graph = compile_from_file('/workspace/scripts/engage_active_perception.yaml')
    
    # Create the main figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Show the complete network structure
    show_network_structure(ax1, graph)
    
    # 2. Show the hierarchical breakdown
    show_hierarchical_breakdown(ax2, graph)
    
    # 3. Show the activation flow
    show_activation_flow(ax3, graph)
    
    # 4. Show the temporal sequencing
    show_temporal_sequencing(ax4, graph)
    
    plt.suptitle('"Engage Active Perception" - Word-Level to Syllable-Level Hypothesis Activation', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/word_to_syllable_activation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return graph

def show_network_structure(ax, graph):
    """Show the complete network structure with emphasis on hierarchy."""
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add units
    for unit_id, unit in graph.units.items():
        G.add_node(unit_id, kind=unit.kind.name, threshold=unit.thresh)
    
    # Add edges
    for src_id, edges in graph.out_edges.items():
        for edge in edges:
            G.add_edge(edge.src, edge.dst, type=edge.type.name, weight=edge.w)
    
    # Create hierarchical layout
    pos = create_focused_layout(graph)
    
    # Color nodes by type
    node_colors = []
    node_sizes = []
    for unit_id in G.nodes():
        unit = graph.units[unit_id]
        if unit.kind == UnitType.SCRIPT:
            if 'phrase' in unit_id:
                node_colors.append('#FF6B6B')  # Red for phrase
                node_sizes.append(1200)
            elif any(word in unit_id for word in ['engage', 'active', 'perception']):
                node_colors.append('#4ECDC4')  # Teal for words
                node_sizes.append(800)
            else:
                node_colors.append('#45B7D1')  # Blue for phonemes
                node_sizes.append(500)
        else:
            node_colors.append('#96CEB4')  # Green for terminals
            node_sizes.append(300)
    
    # Draw edges by type
    edge_styles = {
        'SUB': {'color': '#2E8B57', 'style': 'solid', 'width': 2},
        'SUR': {'color': '#DC143C', 'style': 'solid', 'width': 1.5},
        'POR': {'color': '#8A2BE2', 'style': 'dashed', 'width': 2},
        'RET': {'color': '#FF8C00', 'style': 'dotted', 'width': 1.5}
    }
    
    for link_type, style in edge_styles.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == link_type]
        if edges_of_type:
            nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                 edge_color=style['color'],
                                 style=style['style'],
                                 width=style['width'],
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=15,
                                 ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, ax=ax)
    
    # Draw labels
    labels = {}
    for unit_id in G.nodes():
        unit = graph.units[unit_id]
        if unit.kind == UnitType.SCRIPT:
            if 'phrase' in unit_id:
                labels[unit_id] = 'PHRASE\n"Engage Active\nPerception"'
            elif any(word in unit_id for word in ['engage', 'active', 'perception']):
                word = unit_id.split('_')[1]
                labels[unit_id] = f'WORD\n"{word.title()}"'
            else:
                phoneme = unit_id.split('_')[1:3]
                labels[unit_id] = f'PHONEME\n{"/".join(phoneme)}'
        else:
            feature = unit_id.split('_')[1]
            labels[unit_id] = f'TERMINAL\n{feature}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title('Complete ReCoN Network Structure', fontsize=12, fontweight='bold')
    ax.axis('off')

def create_focused_layout(graph):
    """Create a focused layout for the network."""
    pos = {}
    
    # Phrase level (top center)
    pos['u_phrase'] = (0, 3)
    
    # Word level (second row)
    pos['u_engage'] = (-2, 2)
    pos['u_active'] = (0, 2)
    pos['u_perception'] = (2, 2)
    
    # Phoneme level (third row) - grouped by word
    # Engage phonemes
    pos['u_en_phoneme'] = (-2.5, 1)
    pos['u_gei_phoneme'] = (-2, 1)
    pos['u_dj_phoneme'] = (-1.5, 1)
    
    # Active phonemes
    pos['u_ak_phoneme'] = (-0.5, 1)
    pos['u_ti_phoneme'] = (0, 1)
    pos['u_v_phoneme'] = (0.5, 1)
    
    # Perception phonemes
    pos['u_per_phoneme'] = (1.5, 1)
    pos['u_sep_phoneme'] = (2, 1)
    pos['u_shun_phoneme'] = (2.5, 1)
    
    # Terminal level (bottom)
    terminals = [uid for uid, u in graph.units.items() if u.kind == UnitType.TERMINAL]
    for i, term_id in enumerate(terminals):
        x = -3 + (i * 6) / (len(terminals) - 1) if len(terminals) > 1 else 0
        pos[term_id] = (x, 0)
    
    return pos

def show_hierarchical_breakdown(ax, graph):
    """Show the hierarchical breakdown of the hypothesis."""
    
    # Define the hierarchy levels
    levels = [
        ("Phrase Level", ["u_phrase"], "Complete phrase 'Engage Active Perception'"),
        ("Word Level", ["u_engage", "u_active", "u_perception"], "Individual word recognition"),
        ("Phoneme Level", [
            "u_en_phoneme", "u_gei_phoneme", "u_dj_phoneme",  # engage
            "u_ak_phoneme", "u_ti_phoneme", "u_v_phoneme",    # active
            "u_per_phoneme", "u_sep_phoneme", "u_shun_phoneme"  # perception
        ], "Syllable-level phoneme recognition"),
        ("Terminal Level", [uid for uid, u in graph.units.items() if u.kind == UnitType.TERMINAL], 
         "Audio feature detection (MFCC, pitch, formants, etc.)")
    ]
    
    y_pos = 0.9
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (level_name, units, description) in enumerate(levels):
        # Draw level box
        box = FancyBboxPatch((0.05, y_pos - 0.12), 0.9, 0.1,
                            boxstyle="round,pad=0.02",
                            facecolor=colors[i],
                            alpha=0.3,
                            edgecolor=colors[i],
                            linewidth=2)
        ax.add_patch(box)
        
        # Add level title
        ax.text(0.1, y_pos - 0.02, level_name, fontsize=12, fontweight='bold', color=colors[i])
        
        # Add description
        ax.text(0.1, y_pos - 0.06, description, fontsize=10, style='italic')
        
        # Add unit count
        ax.text(0.8, y_pos - 0.06, f"{len(units)} units", fontsize=9, ha='right')
        
        y_pos -= 0.2
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Hierarchical Breakdown', fontsize=12, fontweight='bold')
    ax.axis('off')

def show_activation_flow(ax, graph):
    """Show the activation flow sequence."""
    
    # Define the activation sequence
    sequence = [
        ("1. Phrase Activation", "Root phrase 'Engage Active Perception' becomes active"),
        ("2. Word Requests", "SUR links send requests to word-level units"),
        ("3. Phoneme Requests", "Each word requests its constituent phonemes"),
        ("4. Terminal Requests", "Phonemes request audio feature detection"),
        ("5. Feature Detection", "Terminals detect MFCC, pitch, formant features"),
        ("6. Evidence Flow", "SUB links carry evidence from terminals to phonemes"),
        ("7. Phoneme Confirmation", "Phonemes confirm based on terminal evidence"),
        ("8. Word Confirmation", "Words confirm based on phoneme evidence"),
        ("9. Phrase Confirmation", "Full phrase confirms with temporal sequencing")
    ]
    
    y_positions = np.linspace(0.95, 0.05, len(sequence))
    
    for i, (step, description) in enumerate(sequence):
        # Draw step box
        color = '#E8F4FD' if i % 2 == 0 else '#F0F8FF'
        box = FancyBboxPatch((0.05, y_positions[i] - 0.08), 0.9, 0.06,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='#4A90E2',
                            linewidth=1)
        ax.add_patch(box)
        
        # Add step text
        ax.text(0.1, y_positions[i] - 0.02, step, fontsize=10, fontweight='bold')
        ax.text(0.1, y_positions[i] - 0.05, description, fontsize=9)
        
        # Add arrow to next step
        if i < len(sequence) - 1:
            ax.arrow(0.5, y_positions[i] - 0.08, 0, -0.05, 
                    head_width=0.02, head_length=0.01, fc='#4A90E2', ec='#4A90E2')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Activation Flow Sequence', fontsize=12, fontweight='bold')
    ax.axis('off')

def show_temporal_sequencing(ax, graph):
    """Show the temporal sequencing between words."""
    
    # Define the temporal sequence
    temporal_sequence = [
        ("u_engage", "Word: 'Engage'", "First in sequence"),
        ("u_active", "Word: 'Active'", "Second in sequence"),
        ("u_perception", "Word: 'Perception'", "Third in sequence")
    ]
    
    # Create a timeline
    x_positions = [0.2, 0.5, 0.8]
    y_center = 0.5
    
    for i, (unit_id, label, description) in enumerate(temporal_sequence):
        # Draw unit box
        box = FancyBboxPatch((x_positions[i] - 0.1, y_center - 0.15), 0.2, 0.3,
                            boxstyle="round,pad=0.02",
                            facecolor='#4ECDC4',
                            alpha=0.7,
                            edgecolor='#2E8B57',
                            linewidth=2)
        ax.add_patch(box)
        
        # Add unit label
        ax.text(x_positions[i], y_center + 0.05, label, fontsize=10, fontweight='bold', ha='center')
        ax.text(x_positions[i], y_center - 0.05, description, fontsize=8, ha='center')
        
        # Add sequence number
        ax.text(x_positions[i], y_center - 0.12, f"Step {i+1}", fontsize=8, ha='center', style='italic')
        
        # Add arrow to next unit
        if i < len(temporal_sequence) - 1:
            ax.arrow(x_positions[i] + 0.1, y_center, 0.15, 0, 
                    head_width=0.05, head_length=0.02, fc='#8A2BE2', ec='#8A2BE2')
    
    # Add POR/RET explanation
    ax.text(0.5, 0.1, "POR links ensure sequential activation", fontsize=10, ha='center', style='italic')
    ax.text(0.5, 0.05, "RET links provide completion feedback", fontsize=10, ha='center', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Temporal Sequencing (POR/RET Links)', fontsize=12, fontweight='bold')
    ax.axis('off')

def run_detailed_simulation(graph):
    """Run a detailed simulation showing the activation sequence."""
    
    print("\nRunning detailed simulation of word-to-syllable activation...")
    
    # Create engine
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
    
    print("Initial activation: u_phrase (phrase level)")
    
    # Track activation sequence
    activation_sequence = []
    
    for step in range(1, 8):
        snapshot = engine.step(1)
        
        # Track which units became active this step
        newly_active = []
        for unit_id, unit_data in snapshot['units'].items():
            if unit_data['state'] in ['ACTIVE', 'CONFIRMED'] and unit_data['a'] > 0.5:
                if unit_id not in [u for u, _ in activation_sequence]:
                    newly_active.append(unit_id)
        
        if newly_active:
            activation_sequence.append((step, newly_active))
        
        print(f"\nStep {step}:")
        
        # Show phrase level
        phrase_state = snapshot['units']['u_phrase']
        print(f"  Phrase: {phrase_state['state']} (a={phrase_state['a']:.2f})")
        
        # Show word level
        word_units = ['u_engage', 'u_active', 'u_perception']
        for word in word_units:
            if word in snapshot['units']:
                word_state = snapshot['units'][word]
                print(f"  {word}: {word_state['state']} (a={word_state['a']:.2f})")
        
        # Show phoneme level (first few)
        phoneme_units = [uid for uid, data in snapshot['units'].items() 
                        if 'phoneme' in uid and data['a'] > 0.1]
        if phoneme_units:
            print(f"  Active phonemes: {phoneme_units[:3]}{'...' if len(phoneme_units) > 3 else ''}")
        
        # Show terminal level
        terminal_units = [uid for uid, data in snapshot['units'].items() 
                         if data['kind'] == 'TERMINAL' and data['a'] > 0.1]
        if terminal_units:
            print(f"  Active terminals: {terminal_units[:3]}{'...' if len(terminal_units) > 3 else ''}")
    
    print(f"\nActivation sequence summary:")
    for step, units in activation_sequence:
        print(f"  Step {step}: {units}")

if __name__ == "__main__":
    print("Creating word-to-syllable activation sequence visualization...")
    
    # Create the visualization
    graph = create_activation_sequence_visualization()
    
    # Run detailed simulation
    run_detailed_simulation(graph)
    
    print(f"\nVisualization saved as: word_to_syllable_activation.png")
    print("This shows how the 'Engage Active Perception' hypothesis activates")
    print("sequentially from word-level to syllable-level hypotheses.")