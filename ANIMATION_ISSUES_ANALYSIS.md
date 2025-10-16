# ReCoN Animation Accuracy Issues - Investigation Report

## Summary
Investigation of animation inaccuracies in the ReCoN network visualization, particularly focusing on:
1. Offset elements during animation steps
2. Incorrect positioning of message animations between nodes
3. Mathematical issues in edge point calculations

## Issues Identified

### Issue 1: Complex and Potentially Buggy Square Edge Point Calculation

**Location**: `scripts/manim_common.py`, function `_get_shape_edge_point()` (lines 98-159)

**Problem**: The current implementation for finding where a ray intersects a square's edge uses an overly complex ray-line intersection algorithm with potential mathematical errors.

**Current Logic**:
```python
# Lines 133-151 (simplified)
denom = direction[0] * edge_unit[1] - direction[1] * edge_unit[0]
if abs(denom) > 1e-10:
    t = ((v1[0] - center[0]) * edge_unit[1] - (v1[1] - center[1]) * edge_unit[0]) / denom
    s = ((v1[0] - center[0]) * direction[1] - (v1[1] - center[1]) * direction[0]) / denom
    
    if t > 0 and 0 <= s <= 1:
        intersection_point = center + t * direction
```

**Issues**:
1. The formula computes a 2D cross product for intersection, but `s` represents a parameter along the edge vector that is NOT normalized to the edge length
2. The condition `0 <= s <= 1` assumes `s` is a unit parameter, but `edge_unit` is normalized while the calculation should use the actual edge length
3. The `projection_length` check (line 131) is redundant and potentially incorrect
4. Falls back to `center + direction * node.radius` which is wrong for squares (line 156)

**Expected Behavior**: For a square centered at origin with half-side length `h`, the intersection point with a ray in direction `(dx, dy)` should be:
- If `|dx| > |dy|`: hits vertical edge at `x = ±h`, `y = (±h) * dy/dx`  
- If `|dy| > |dx|`: hits horizontal edge at `y = ±h`, `x = (±h) * dx/dy`

### Issue 2: Incorrect Fallback for Square Edge Points

**Location**: `scripts/manim_common.py`, line 156

**Problem**: When the complex intersection algorithm fails to find a valid point, it falls back to:
```python
return center + direction * node.radius
```

This treats the square as a circle, which causes arrows/messages to be positioned incorrectly—they don't start/end at the actual edge of the square.

**Impact**: Messages appear offset from where they should be, especially at diagonal angles where the square's corners are farther from center than `radius`.

### Issue 3: Asymmetric Edge Point Distances

**Problem**: Due to the above issues, the distance from node center to edge point varies inconsistently:
- Circle nodes: Always `radius` (correct)
- Square nodes: Varies between `radius` (at face centers) and `radius * sqrt(2)` (at corners)
- When the algorithm fails: Always `radius` (incorrect)

This causes visual asymmetry in animations where arrows/messages appear to travel different distances even when the nodes are the same distance apart.

### Issue 4: Message Animation Positioning

**Location**: `scripts/manim_common.py`, functions:
- `animate_message_between_nodes()` (lines 228-266)
- `animate_message_to_root()` (lines 268-305)

**Problem**: These functions rely on `_get_shape_edge_point()` for positioning. When that function returns incorrect points:
1. Messages start at wrong positions (offset from actual node edges)
2. Messages end at wrong positions 
3. Message travel distances are inconsistent
4. Highlight shapes that appear at destination may not align properly with node shape

**Visual Impact**: Users see messages "jumping" or appearing disconnected from nodes, especially for terminal (square) nodes.

### Issue 5: Node Positioning May Not Reflect Actual Graph Topology

**Location**: Various scene classes in `scripts/manim_recon_house.py`

**Observation**: Node positions are hardcoded in each scene's `node_positions` dictionary. While this gives visual control, it means:
1. Positions might not accurately reflect the conceptual hierarchy
2. Adding/removing nodes requires manual position updates
3. Different scenes use different layouts, causing inconsistency

**Current Approach**:
```python
node_positions = {
    "u_root": (right_x, y_root, 0),
    "u_roof": (right_x - spread, y_root - 2.0, 0),
    "u_body": (right_x, y_root - 2.0, 0),
    "u_door": (right_x + spread, y_root - 2.0, 0),
    "t_horz": (right_x - spread, y_root - 4.0, 0),
    "t_mean": (right_x, y_root - 4.0, 0),
    "t_vert": (right_x + spread, y_root - 4.0, 0),
}
```

This is not inherently wrong, but could be improved with automatic layout based on graph structure.

## Root Causes

1. **Overcomplicated Geometry**: The square edge intersection uses a general ray-line intersection when a simpler approach would work better
2. **Incorrect Math**: The 2D cross product intersection formula has parameter scaling issues
3. **Poor Fallback**: When intersection fails, falling back to circle-like behavior is wrong
4. **No Validation**: No tests verify that edge points are actually on the shape boundary

## Recommended Fixes

### Fix 1: Simplify Square Edge Point Calculation (HIGH PRIORITY)

Replace the complex intersection logic with a simpler, correct approach:

```python
def _get_shape_edge_point(node: NodeViz, direction: np.ndarray) -> np.ndarray:
    """Return the point on a node's outline in the given direction vector."""
    center = node.shape.get_center()
    
    # Normalize direction
    norm = np.linalg.norm(direction[:2])
    if norm < 1e-10:
        return center
    dx, dy = direction[0] / norm, direction[1] / norm
    
    if isinstance(node.shape, Square):
        # For squares, use simple geometric calculation
        # Square has vertices at center ± (half_side, half_side)
        half_side = node.radius  # radius is half the side length
        
        # Determine which edge the direction points to
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy:
            # Hits left or right vertical edge
            edge_x = half_side if dx > 0 else -half_side
            edge_y = edge_x * dy / dx if abs_dx > 1e-10 else 0
            # Clamp y to square bounds
            edge_y = max(-half_side, min(half_side, edge_y))
        else:
            # Hits top or bottom horizontal edge
            edge_y = half_side if dy > 0 else -half_side
            edge_x = edge_y * dx / dy if abs_dy > 1e-10 else 0
            # Clamp x to square bounds
            edge_x = max(-half_side, min(half_side, edge_x))
        
        return center + np.array([edge_x, edge_y, 0.0])
    else:
        # For circles, simple radius-based calculation
        return center + direction / norm * node.radius
```

### Fix 2: Add Edge Point Validation Tests (MEDIUM PRIORITY)

Create unit tests to verify edge points are correct:
- Test all 8 major directions (cardinal + diagonals)
- Verify points are actually on shape boundary
- Verify distance from center is correct for each shape type
- Test edge cases (zero direction, nearly-zero direction)

### Fix 3: Improve Message Animation Robustness (LOW PRIORITY)

Add validation in animation functions:
```python
def animate_message_between_nodes(self, src_node, dst_node, message_text, color, duration):
    # ... existing code ...
    
    # Validate edge points
    start_dist = np.linalg.norm(start_point[:2] - src_center[:2])
    end_dist = np.linalg.norm(end_point[:2] - dst_center[:2])
    
    # Log warnings if distances seem wrong
    expected_dist = src_node.radius if isinstance(src_node.shape, Circle) else src_node.radius
    if abs(start_dist - expected_dist) > 0.1:
        print(f"Warning: Unusual start distance: {start_dist} vs expected {expected_dist}")
    
    # ... rest of animation ...
```

### Fix 4: Consider Automatic Layout (FUTURE ENHANCEMENT)

For consistency and maintainability, consider using a graph layout algorithm:
- NetworkX provides spring layout, hierarchical layout, etc.
- Would ensure consistent positioning across scenes
- Reduces manual tweaking

However, this is lower priority since manual layout gives better visual control for presentations.

## Impact Assessment

### High Impact
- Fix 1: Will immediately improve visual accuracy
- Affects all animations with square (terminal) nodes
- Should eliminate most "offset" issues user reported

### Medium Impact  
- Fix 2: Prevents regressions
- Catches future geometry bugs early

### Low Impact
- Fix 3: Defensive programming, helps debugging
- Fix 4: Nice to have, but manual layout works fine

## Testing Strategy

After implementing fixes:

1. **Visual Inspection**: Render animations and verify:
   - Messages start exactly at node edges
   - Messages end exactly at node edges  
   - No visible "jumps" or offsets
   - Symmetric behavior for equivalent nodes

2. **Automated Tests**: Run unit tests to verify:
   - Edge points at correct distances
   - Edge points on shape boundaries
   - Consistent behavior across directions

3. **Regression Check**: Re-render existing scenes:
   - `RootActivationScene`
   - `HouseWalkthrough`
   - `CastleActivationScene`
   - `HouseHypothesisOnCastle`

## Implementation Priority

1. **IMMEDIATE**: Fix 1 - Simplify square edge point calculation
2. **SOON**: Fix 2 - Add validation tests  
3. **LATER**: Fix 3 - Add animation robustness checks
4. **FUTURE**: Fix 4 - Consider automatic layout

## Conclusion

The primary issue is the overcomplicated and buggy square edge point calculation in `_get_shape_edge_point()`. This causes:
- Messages to appear offset from square (terminal) nodes
- Inconsistent animation distances
- Visual artifacts during network activation sequences

The fix is straightforward: replace the complex ray-line intersection with a simple geometric calculation based on whether the direction primarily points horizontally or vertically. This will immediately improve animation accuracy with minimal risk.
