#!/usr/bin/env python3
"""
Validate the castle hypothesis script by compiling its YAML into a ReCoN graph
and running comprehensive structural validations. Exits with non-zero status
if any validation errors are detected.
"""

import sys
import os
import json

# Ensure we can import recon_core from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core.compiler import compile_from_file


def main() -> int:
    yaml_path = os.path.join(os.path.dirname(__file__), 'castle.yaml')
    if not os.path.exists(yaml_path):
        print(f"Error: YAML script not found at {yaml_path}")
        return 2

    # Compile graph from YAML
    graph = compile_from_file(yaml_path)

    # Run comprehensive validations
    results = graph.validate_all(strict_activation=True)
    summary = graph.get_validation_summary(results)

    is_valid = graph.is_valid()

    print("Castle Hypothesis Validation Summary:")
    print(json.dumps({
        "valid": is_valid,
        "summary": summary,
        "issues": results,
        "units": len(graph.units),
        "edges": sum(len(edges) for edges in graph.out_edges.values()),
    }, indent=2))

    return 0 if is_valid else 1


if __name__ == '__main__':
    raise SystemExit(main())

