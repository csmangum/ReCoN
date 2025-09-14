#!/usr/bin/env python3
"""
Demo script showing GraphML export functionality for ReCoN graphs.

This script compiles a ReCoN graph from a YAML script and exports it to GraphML format,
which can be imported into graph analysis tools like Gephi, yEd, or other NetworkX-compatible software.
"""

import sys
import os
# Add parent directory to path so we can import recon_core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core.compiler import compile_from_file

def main():
    """Demo GraphML export functionality."""

    # Compile a graph from the house script
    print("Compiling ReCoN graph from scripts/house.yaml...")
    graph = compile_from_file('scripts/house.yaml')

    print(f"Graph contains {len(graph.units)} units and {sum(len(edges) for edges in graph.out_edges.values())} edges")

    # Export to GraphML
    output_file = 'house_recon_graph.graphml'
    print(f"Exporting to {output_file}...")
    graph.export_graphml(output_file)

    print("✓ GraphML export completed successfully!")
    print(f"✓ File saved as: {output_file}")
    print()
    print("The GraphML file contains:")
    print("- Node attributes: kind (SCRIPT/TERMINAL), state (INACTIVE/ACTIVE/etc.), activation, threshold")
    print("- Edge attributes: type (SUB/SUR/POR/RET), weight")
    print("- Can be imported into Gephi, yEd, or loaded with NetworkX")

    # Also demonstrate NetworkX conversion
    print()
    print("NetworkX conversion available via graph.to_networkx() for programmatic analysis")

if __name__ == '__main__':
    main()
