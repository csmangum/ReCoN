#!/usr/bin/env python3
"""
ReCoN Graph Validation Demonstration Script

This script demonstrates all the graph validation capabilities added to the ReCoN system,
including cycle detection, link consistency validation, activation bounds checking,
performance metrics, and custom validation rules.
"""

import sys
import os
# Add the parent directory to the path so we can import recon_core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core.graph import Graph, Unit, Edge
from recon_core.enums import UnitType, LinkType


def create_problematic_graph():
    """Create a graph with various validation issues for demonstration."""
    graph = Graph()

    # Create units with some issues
    root_script = Unit('root', UnitType.SCRIPT, a=0.9, thresh=0.6)
    intermediate_script = Unit('intermediate', UnitType.SCRIPT, a=0.7, thresh=0.5)
    terminal1 = Unit('terminal1', UnitType.TERMINAL, a=1.2, thresh=0.8)  # Invalid activation
    terminal2 = Unit('terminal2', UnitType.TERMINAL, a=0.6, thresh=0.4)
    isolated_terminal = Unit('isolated', UnitType.TERMINAL)  # No connections

    # Add units to graph
    for unit in [root_script, intermediate_script, terminal1, terminal2, isolated_terminal]:
        graph.add_unit(unit)

    # Add edges - some valid, some problematic
    graph.add_edge(Edge('terminal1', 'intermediate', LinkType.SUB, w=0.9))  # Valid SUB
    graph.add_edge(Edge('terminal2', 'intermediate', LinkType.SUB, w=0.7))  # Valid SUB
    graph.add_edge(Edge('intermediate', 'root', LinkType.SUB, w=1.0))      # Valid SUB

    graph.add_edge(Edge('root', 'intermediate', LinkType.SUR, w=0.8))     # Valid SUR
    graph.add_edge(Edge('intermediate', 'terminal1', LinkType.SUR, w=0.6)) # Valid SUR

    # Problematic edges
    graph.add_edge(Edge('terminal1', 'root', LinkType.POR, w=0.5))        # Invalid: POR from terminal
    graph.add_edge(Edge('terminal1', 'intermediate', LinkType.SUR, w=0.3)) # Invalid: SUR from terminal
    graph.add_edge(Edge('root', 'intermediate', LinkType.SUB, w=0.2))     # Invalid: SUB from script

    return graph


def demonstrate_cycle_detection():
    """Demonstrate cycle detection capabilities."""
    print("\n" + "="*60)
    print("CYCLE DETECTION DEMONSTRATION")
    print("="*60)

    # Create a graph with cycles
    graph = Graph()

    # Create units
    for i in range(5):
        graph.add_unit(Unit(f'u{i}', UnitType.SCRIPT))

    # Create cycles in different link types
    # SUB cycle: u0 -> u1 -> u2 -> u0
    graph.add_edge(Edge('u0', 'u1', LinkType.SUB))
    graph.add_edge(Edge('u1', 'u2', LinkType.SUB))
    graph.add_edge(Edge('u2', 'u0', LinkType.SUB))

    # SUR cycle: u3 -> u4 -> u3
    graph.add_edge(Edge('u3', 'u4', LinkType.SUR))
    graph.add_edge(Edge('u4', 'u3', LinkType.SUR))

    # No POR or RET cycles in this example

    # Check for cycles
    cycle_results = graph.validate_cycles()
    print(f"Cycle detection results: {len(cycle_results)} link types have cycles")

    for link_type, cycles in cycle_results.items():
        print(f"\n{link_type} cycles found: {len(cycles)}")
        for cycle in cycles:
            print(f"  - {cycle}")

    # Check specific link type
    sub_cycles = graph.validate_cycles(LinkType.SUB)
    print(f"\nSpecific SUB cycle check: {len(sub_cycles.get('SUB', []))} cycles found")


def demonstrate_validation_features():
    """Demonstrate all validation features."""
    print("\n" + "="*60)
    print("COMPREHENSIVE VALIDATION DEMONSTRATION")
    print("="*60)

    graph = create_problematic_graph()

    print("Graph created with intentional issues for demonstration:")
    print("- Invalid activation bounds (terminal1.a = 1.2)")
    print("- Invalid link types (POR from terminal, SUR from terminal)")
    print("- Isolated unit (no connections)")
    print("- Edge weight issues")

    # Run comprehensive validation
    print("\nRunning comprehensive validation...")
    validation_results = graph.validate_all(strict_activation=True)

    # Display results
    summary = graph.get_validation_summary(validation_results)
    print("\nValidation Summary:")
    print(f"- Total issues: {summary['total_issues']}")
    print(f"- Errors: {summary['errors']}")
    print(f"- Warnings: {summary['warnings']}")
    print(f"- Categories with issues: {summary['categories_with_issues']}")

    print(f"\nGraph validity: {'VALID' if graph.is_valid() else 'INVALID'}")

    # Show detailed issues
    for category, issues in validation_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"  {issue_type}:")
                for issue in issue_list:
                    print(f"    - {issue}")


def demonstrate_performance_metrics():
    """Demonstrate performance metrics analysis."""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS DEMONSTRATION")
    print("="*60)

    # Create a moderately complex graph
    graph = Graph()

    # Add many units
    for i in range(20):
        unit_type = UnitType.TERMINAL if i < 15 else UnitType.SCRIPT
        graph.add_unit(Unit(f'unit_{i:02d}', unit_type))

    # Add connections with varying complexity
    for i in range(15):  # Terminals
        # Each terminal connects to 2-3 scripts
        for j in range(2 + (i % 2)):
            target_script = f'unit_{15 + (i + j) % 5:02d}'
            graph.add_edge(Edge(f'unit_{i:02d}', target_script, LinkType.SUB, w=0.8))

    for i in range(15, 20):  # Scripts
        # Scripts connect to other scripts with SUR/POR
        for j in range(1 + (i % 3)):
            target = f'unit_{(i + j + 1) % 20:02d}'
            link_type = LinkType.SUR if j % 2 == 0 else LinkType.POR
            graph.add_edge(Edge(f'unit_{i:02d}', target, link_type, w=0.6))

    # Analyze performance
    metrics = graph.analyze_performance_metrics()

    print("Performance Metrics Analysis:")
    print("\nStructure Metrics:")
    for key, value in metrics['structure_metrics'].items():
        print(f"  {key}: {value}")

    print("\nComplexity Metrics:")
    for key, value in metrics['complexity_metrics'].items():
        if key == 'degree_distribution':
            print(f"  {key}: {dict(list(value.items())[:5])}...")  # Show first 5
        elif key == 'link_type_distribution':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    print("\nEfficiency Indicators:")
    for key, value in metrics['efficiency_indicators'].items():
        print(f"  {key}: {value}")

    if metrics['bottleneck_warnings']:
        print("\nBottleneck Warnings:")
        for warning in metrics['bottleneck_warnings']:
            print(f"  - {warning}")

    # Show overall statistics
    stats = graph.get_graph_statistics()
    print(f"\nOverall Health Score: {stats['health_score']:.3f}")
    print(f"Detailed validation summary: {stats['validation_summary']}")


def demonstrate_custom_validation_rules():
    """Demonstrate custom validation rules."""
    print("\n" + "="*60)
    print("CUSTOM VALIDATION RULES DEMONSTRATION")
    print("="*60)

    graph = create_problematic_graph()

    # Define custom validation rules
    def validate_max_fan_out(graph):
        """Custom rule: Check that no unit has more than 5 outgoing connections."""
        issues = {}
        max_fan_out = 5

        for unit_id, edges in graph.out_edges.items():
            if len(edges) > max_fan_out:
                if 'fan_out_issues' not in issues:
                    issues['fan_out_issues'] = []
                issues['fan_out_issues'].append(
                    f"Unit '{unit_id}' has {len(edges)} outgoing connections (max allowed: {max_fan_out})"
                )

        return issues

    def validate_terminal_coverage(graph):
        """Custom rule: Ensure at least 80% of terminals have evidence connections."""
        terminals = [uid for uid, u in graph.units.items() if u.kind == UnitType.TERMINAL]
        if not terminals:
            return {}

        connected_terminals = 0
        for term_id in terminals:
            if any(e.type == LinkType.SUB for e in graph.out_edges.get(term_id, [])):
                connected_terminals += 1

        coverage_ratio = connected_terminals / len(terminals)
        min_coverage = 0.8

        if coverage_ratio < min_coverage:
            return {
                'coverage_issues': [
                    f"Terminal coverage: {coverage_ratio:.1%} ({connected_terminals}/{len(terminals)}) "
                    f"below minimum {min_coverage:.1%}"
                ]
            }

        return {}

    def validate_activation_consistency(graph):
        """Custom rule: Check that activation levels are reasonable for unit states."""
        issues = {}

        for unit_id, unit in graph.units.items():
            # Terminals in TRUE state should have activation >= threshold
            if unit.kind == UnitType.TERMINAL and unit.state.name == 'TRUE':
                if unit.a < unit.thresh:
                    if 'consistency_issues' not in issues:
                        issues['consistency_issues'] = []
                    issues['consistency_issues'].append(
                        f"Terminal '{unit_id}' in TRUE state but activation {unit.a} < threshold {unit.thresh}"
                    )

        return issues

    # Add custom rules to graph
    graph.add_custom_validation_rule('max_fan_out', validate_max_fan_out)
    graph.add_custom_validation_rule('terminal_coverage', validate_terminal_coverage)
    graph.add_custom_validation_rule('activation_consistency', validate_activation_consistency)

    # Run custom validations
    custom_results = graph.validate_custom_rules()

    print("Custom Validation Rules Results:")
    if custom_results:
        for rule_name, issues in custom_results.items():
            print(f"\n{rule_name.upper().replace('_', ' ')}:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}:")
                    for issue in issue_list:
                        print(f"    - {issue}")
    else:
        print("No custom validation issues found!")


def main():
    """Run all demonstrations."""
    print("ReCoN Graph Validation Features Demonstration")
    print("This script demonstrates the comprehensive validation capabilities")
    print("added to the ReCoN graph system.\n")

    try:
        demonstrate_cycle_detection()
        demonstrate_validation_features()
        demonstrate_performance_metrics()
        demonstrate_custom_validation_rules()

        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nSummary of validation features demonstrated:")
        print("✓ Cycle detection for different link types")
        print("✓ Link consistency validation")
        print("✓ Unit relationship validation")
        print("✓ Activation bounds checking")
        print("✓ Graph integrity validation")
        print("✓ Performance metrics analysis")
        print("✓ Custom validation rules framework")
        print("✓ Health score calculation")
        print("✓ Comprehensive validation reporting")

    except (ImportError, AttributeError, ValueError, RuntimeError) as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
