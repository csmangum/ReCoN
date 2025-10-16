#!/usr/bin/env python3
"""
Test script to investigate animation accuracy issues in ReCoN network visualization.

This script tests:
1. The _get_shape_edge_point function for both circles and squares
2. Message positioning and arrow calculations
3. Node positioning relative to actual graph structure
"""

import os
import sys
import numpy as np

# Add scripts to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from manim import Circle, Square, Text
    from manim_common import NodeViz, _get_shape_edge_point
    from recon_core.enums import UnitType
except ImportError as e:
    print(f"Import error: {e}")
    print("This script requires manim to be installed.")
    sys.exit(1)


def test_edge_point_accuracy():
    """Test edge point calculation for circles and squares."""
    print("=" * 80)
    print("Testing _get_shape_edge_point accuracy")
    print("=" * 80)
    
    # Create test nodes
    circle_node = NodeViz("u_test", unit_type=UnitType.SCRIPT, radius=0.4)
    circle_node.move_to((0, 0, 0))
    
    square_node = NodeViz("t_test", unit_type=UnitType.TERMINAL, radius=0.4)
    square_node.move_to((0, 0, 0))
    
    # Test directions (right, up, diagonal, etc.)
    test_directions = [
        (1, 0, 0, "Right"),
        (0, 1, 0, "Up"),
        (-1, 0, 0, "Left"),
        (0, -1, 0, "Down"),
        (1, 1, 0, "Up-Right Diagonal"),
        (-1, 1, 0, "Up-Left Diagonal"),
        (1, -1, 0, "Down-Right Diagonal"),
        (-1, -1, 0, "Down-Left Diagonal"),
    ]
    
    print("\n--- Circle Node Tests ---")
    for dx, dy, dz, name in test_directions:
        direction = np.array([dx, dy, dz], dtype=float)
        direction = direction / np.linalg.norm(direction)  # normalize
        
        edge_point = _get_shape_edge_point(circle_node, direction)
        center = circle_node.shape.get_center()
        
        # Calculate expected point (should be center + radius * direction)
        expected = center + circle_node.radius * direction
        
        distance_from_center = np.linalg.norm(edge_point - center)
        error = np.linalg.norm(edge_point - expected)
        
        print(f"{name:25s}: Distance from center = {distance_from_center:.4f}, Error = {error:.6f}")
        if error > 0.001:
            print(f"  WARNING: Large error! Expected {expected[:2]}, got {edge_point[:2]}")
    
    print("\n--- Square Node Tests ---")
    for dx, dy, dz, name in test_directions:
        direction = np.array([dx, dy, dz], dtype=float)
        direction = direction / np.linalg.norm(direction)  # normalize
        
        edge_point = _get_shape_edge_point(square_node, direction)
        center = square_node.shape.get_center()
        
        # For a square with side_length = 2*radius, half_side = radius
        half_side = square_node.radius
        
        # Calculate expected point manually
        # The square has vertices at (+/-half_side, +/-half_side)
        # We need to find which edge the ray hits
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Determine which edge based on angle
        if abs_dx > abs_dy:
            # Hits left or right edge
            expected_x = half_side if dx > 0 else -half_side
            expected_y = expected_x * dy / dx if dx != 0 else 0
        else:
            # Hits top or bottom edge
            expected_y = half_side if dy > 0 else -half_side
            expected_x = expected_y * dx / dy if dy != 0 else 0
        
        expected = center + np.array([expected_x, expected_y, 0])
        
        distance_from_center = np.linalg.norm(edge_point[:2] - center[:2])
        error = np.linalg.norm(edge_point[:2] - expected[:2])
        
        print(f"{name:25s}: Distance from center = {distance_from_center:.4f}, Error = {error:.6f}")
        if error > 0.001:
            print(f"  WARNING: Large error! Expected {expected[:2]}, got {edge_point[:2]}")
        
        # Also check if point is actually on the square boundary
        rel_x = edge_point[0] - center[0]
        rel_y = edge_point[1] - center[1]
        on_boundary = (abs(abs(rel_x) - half_side) < 0.001 or 
                      abs(abs(rel_y) - half_side) < 0.001)
        if not on_boundary:
            print(f"  WARNING: Point not on square boundary! rel_pos = ({rel_x:.4f}, {rel_y:.4f})")


def test_message_positioning():
    """Test message animation positioning between nodes."""
    print("\n" + "=" * 80)
    print("Testing message positioning between nodes")
    print("=" * 80)
    
    # Test circle to circle
    src_circle = NodeViz("u_src", unit_type=UnitType.SCRIPT, radius=0.4)
    src_circle.move_to((-2, 0, 0))
    
    dst_circle = NodeViz("u_dst", unit_type=UnitType.SCRIPT, radius=0.4)
    dst_circle.move_to((2, 0, 0))
    
    src_center = src_circle.shape.get_center()
    dst_center = dst_circle.shape.get_center()
    direction = dst_center - src_center
    distance = np.linalg.norm(direction)
    direction = direction / distance
    
    start_point = _get_shape_edge_point(src_circle, direction)
    end_point = _get_shape_edge_point(dst_circle, -direction)
    
    print("\n--- Circle to Circle ---")
    print(f"Source center: {src_center[:2]}")
    print(f"Dest center: {dst_center[:2]}")
    print(f"Start point: {start_point[:2]}")
    print(f"End point: {end_point[:2]}")
    print(f"Expected gap: {2 * 0.4} (2 * radius)")
    print(f"Actual gap: {np.linalg.norm(end_point - start_point):.4f}")
    
    # Test square to square
    src_square = NodeViz("t_src", unit_type=UnitType.TERMINAL, radius=0.4)
    src_square.move_to((-2, 0, 0))
    
    dst_square = NodeViz("t_dst", unit_type=UnitType.TERMINAL, radius=0.4)
    dst_square.move_to((2, 0, 0))
    
    src_center = src_square.shape.get_center()
    dst_center = dst_square.shape.get_center()
    direction = dst_center - src_center
    distance = np.linalg.norm(direction)
    direction = direction / distance
    
    start_point = _get_shape_edge_point(src_square, direction)
    end_point = _get_shape_edge_point(dst_square, -direction)
    
    print("\n--- Square to Square ---")
    print(f"Source center: {src_center[:2]}")
    print(f"Dest center: {dst_center[:2]}")
    print(f"Start point: {start_point[:2]}")
    print(f"End point: {end_point[:2]}")
    print(f"Expected gap: {2 * 0.4} (2 * radius, horizontal)")
    print(f"Actual gap: {np.linalg.norm(end_point - start_point):.4f}")
    
    # Test circle to square (mixed)
    start_point = _get_shape_edge_point(src_circle, direction)
    end_point = _get_shape_edge_point(dst_square, -direction)
    
    print("\n--- Circle to Square ---")
    print(f"Start point (circle): {start_point[:2]}")
    print(f"End point (square): {end_point[:2]}")
    print(f"Actual gap: {np.linalg.norm(end_point - start_point):.4f}")


def test_diagonal_connections():
    """Test connections at various angles to identify offset issues."""
    print("\n" + "=" * 80)
    print("Testing diagonal connections (potential offset issues)")
    print("=" * 80)
    
    # Create nodes at different relative positions
    positions = [
        ((0, 0), (1, 1), "45° Up-Right"),
        ((0, 0), (1, -1), "45° Down-Right"),
        ((0, 0), (2, 1), "~26° Up-Right"),
        ((0, 0), (1, 2), "~63° Up-Right"),
    ]
    
    for src_pos, dst_pos, name in positions:
        src = NodeViz("t_src", unit_type=UnitType.TERMINAL, radius=0.4)
        src.move_to((src_pos[0], src_pos[1], 0))
        
        dst = NodeViz("t_dst", unit_type=UnitType.TERMINAL, radius=0.4)
        dst.move_to((dst_pos[0], dst_pos[1], 0))
        
        src_center = src.shape.get_center()
        dst_center = dst.shape.get_center()
        direction = dst_center - src_center
        distance = np.linalg.norm(direction)
        direction = direction / distance
        
        start_point = _get_shape_edge_point(src, direction)
        end_point = _get_shape_edge_point(dst, -direction)
        
        print(f"\n{name}:")
        print(f"  Direction: ({direction[0]:.4f}, {direction[1]:.4f})")
        print(f"  Start offset from center: {start_point - src_center}")
        print(f"  End offset from center: {end_point - dst_center}")
        
        # Check if offsets are roughly equal in magnitude
        start_mag = np.linalg.norm(start_point[:2] - src_center[:2])
        end_mag = np.linalg.norm(end_point[:2] - dst_center[:2])
        print(f"  Start magnitude: {start_mag:.4f}")
        print(f"  End magnitude: {end_mag:.4f}")
        print(f"  Magnitude difference: {abs(start_mag - end_mag):.4f}")
        
        if abs(start_mag - end_mag) > 0.01:
            print(f"  ⚠️  WARNING: Asymmetric edge points!")


if __name__ == "__main__":
    print("ReCoN Animation Accuracy Investigation")
    print("=" * 80)
    
    try:
        test_edge_point_accuracy()
        test_message_positioning()
        test_diagonal_connections()
        
        print("\n" + "=" * 80)
        print("Investigation complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
