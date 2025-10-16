# Streamlit ReCoN Visualization Issues - Investigation Report

## Date: 2025-10-16

## Summary
Investigation of animation and positioning inaccuracies in the Streamlit ReCoN network visualization (`viz/app_streamlit.py`), focusing on:
1. Message arrow positioning and offset issues
2. Terminal overlay positioning on scene images
3. Dynamic node positioning for generated terminals
4. Overall coordinate system consistency

---

## Critical Issues Identified

### 🚨 CRITICAL Issue 1: Incorrect Message Arrow Shrink Values

**Location**: `viz/app_streamlit.py`, lines 735-747

**Problem**: Message arrows use `shrinkA=25` and `shrinkB=25` which are in **data coordinates**, not pixels!

**Current Code**:
```python
ax_graph.annotate(
    "",
    xy=(end_x, end_y),
    xytext=(start_x, start_y),
    arrowprops=dict(
        arrowstyle="->",
        color=style["color"],
        linewidth=style["width"],
        linestyle=style["style"],
        alpha=style["alpha"],
        shrinkA=25,  # ❌ WAY TOO LARGE!
        shrinkB=25,  # ❌ WAY TOO LARGE!
    ),
    zorder=10,
)
```

**Context**:
- Node positions are in range: x ∈ [-1.5, 1.5], y ∈ [0.8, 3.0]
- Total coordinate space is approximately 3 units wide × 2.2 units tall
- A shrink of 25 units would move arrows MUCH farther than the entire graph!

**Impact**:
- Arrows appear completely disconnected from nodes
- Arrows may not be visible at all if they're pushed outside the plot area
- Message animations look "offset" and inaccurate

**Expected Values**: Should be approximately 0.15-0.3 (roughly matching node visual radius)

**Fix**: Change to:
```python
shrinkA=0.2,  # Shrink by ~0.2 data units from start
shrinkB=0.2,  # Shrink by ~0.2 data units from end
```

---

### ⚠️ HIGH Issue 2: Hardcoded Terminal Scene Overlay Positions

**Location**: `viz/app_streamlit.py`, lines 404-427

**Problem**: Terminal detection overlays on the scene image only show 3 hardcoded terminals, ignoring dynamically generated ones.

**Current Code**:
```python
# Add terminal detection overlays using live activations
terminal_positions = {
    "t_mean": (32, 32),
    "t_vert": (45, 32),
    "t_horz": (20, 32),
}
for term_name, (x, y) in terminal_positions.items():
    if term_name in st.session_state.sim.graph.units:
        # ... draw overlay
```

**Impact**:
- When user generates a scene with many terminals (using the perception system), only 3 show up on the scene
- No visual feedback for rich terminal activations
- Confusing UX - terminals are in the graph but not shown on scene

**Expected Behavior**:
- Should show ALL active terminals
- Should use actual terminal positions if available from perception system
- Or distribute evenly across image if positions unknown

**Fix Options**:
1. **Short-term**: Show all terminals in a grid or distributed pattern
2. **Long-term**: Extract actual terminal detection positions from perception system

---

### ⚠️ MEDIUM Issue 3: Message Arrow Offset Calculation Logic

**Location**: `viz/app_streamlit.py`, lines 717-719

**Problem**: Offset for multiple messages uses global index, not per-edge-pair index.

**Current Code**:
```python
offset = messages.index((sender, receiver, msg)) * 0.05
```

**Issue**: `messages.index()` returns the position in the entire message list, not among messages between the same sender-receiver pair. This means:
- If messages A→B and C→D happen in the same step, they both get different offsets
- But messages going the same direction should have the same base offset
- Creates inconsistent visual spacing

**Impact**: Medium - causes some visual noise but not critical

**Fix**: Calculate offset based on count of messages between the same pair:
```python
# Count how many messages between this pair so far
same_pair_count = sum(1 for s, r, m in messages[:messages.index((sender, receiver, msg))] 
                      if s == sender and r == receiver)
offset = same_pair_count * 0.05
```

---

### ⚠️ MEDIUM Issue 4: Dynamic Terminal Positioning May Cause Overlaps

**Location**: `viz/app_streamlit.py`, lines 539-561

**Problem**: When many terminals are positioned under one parent script, they might overlap.

**Current Code**:
```python
if parent_id in pos:
    parent_x, _ = pos[parent_id]
    xs = np.linspace(-0.8, 0.8, num=len(terms)) if len(terms) > 1 else [0.0]
    for dx, n in zip(xs, terms):
        pos[n] = (float(parent_x + dx), 0.9)
```

**Issues**:
1. All terminals for one parent are placed at same y-coordinate (0.9), creating overlap
2. Range [-0.8, 0.8] might not be enough for many terminals (10+)
3. Fixed y=0.9 might collide with other elements

**Impact**:
- Terminals overlap visually when many are generated
- Graph becomes cluttered and hard to read
- Node labels overlap

**Fix Options**:
1. Use multiple rows if too many terminals per parent
2. Increase vertical spacing
3. Use force-directed layout for terminals

---

### ℹ️ LOW Issue 5: Skip Message Drawing at t=1

**Location**: `viz/app_streamlit.py`, lines 652, 788

**Observation**: Code explicitly skips drawing message arrows at `timeline_idx == 1`:

```python
if st.session_state.sim.message_history and timeline_idx != 1:
```

**Comment**: This seems like a workaround for some issue at the initial step. Not necessarily wrong, but worth investigating why this special case exists. May hide another underlying issue.

---

## Visual Coordinate System Analysis

### Network Graph Coordinate System
- **X range**: Approximately -1.5 to 1.5 (3 units wide)
- **Y range**: Approximately 0.8 to 3.0 (2.2 units tall)
- **Node positions**:
  - Root: (0, 3)
  - Scripts: (-1, 2), (0, 2), (1, 2)
  - Terminals: (-1, 1), (0, 1), (1, 1) or at 0.9 for dynamic ones

### Scene Image Coordinate System
- **X range**: 0 to 64 pixels
- **Y range**: 0 to 64 pixels (with flipped y-axis)
- **Terminal hardcoded positions**: Around (32, 32) center region

### Node Visual Sizes
From state_sizes dictionary (lines 506-515):
- INACTIVE: 1800 (matplotlib scatter size units)
- ACTIVE: 2700
- CONFIRMED: 3300

These are matplotlib scatter point sizes. Converting to approximate data units:
- sqrt(1800) ≈ 42 points → radius ≈ 0.15-0.2 data units
- sqrt(3300) ≈ 57 points → radius ≈ 0.2-0.3 data units

**Conclusion**: Shrink values should be in range 0.15-0.3, not 25!

---

## Testing & Validation Strategy

### Manual Testing Steps
1. **Generate a scene** in Streamlit app
2. **Step through simulation** using Step button
3. **Observe message arrows**:
   - Do they connect to nodes properly?
   - Are they visible?
   - Do they start/end at node edges?
4. **Check terminal overlays**:
   - Are all active terminals shown on scene?
   - Do colors match network graph?
5. **Generate scene with many terminals**:
   - Do they overlap in graph?
   - Are they shown on scene image?

### Automated Testing
- Unit tests for coordinate transformations
- Visual regression tests (screenshot comparison)
- Edge case tests (many terminals, many messages)

---

## Recommended Fix Priority

### 🔴 IMMEDIATE (Critical - breaks visualization)
1. **Fix shrinkA/shrinkB values** in message arrows (Issue 1)
   - Change from 25 to 0.2
   - Test with various node pairs

### 🟡 HIGH (Significant UX issues)  
2. **Fix terminal scene overlays** (Issue 2)
   - Show all terminals, not just 3 hardcoded ones
   - Consider extracting actual positions from perception

### 🟢 MEDIUM (Polish and edge cases)
3. **Fix message offset calculation** (Issue 3)
4. **Improve dynamic terminal positioning** (Issue 4)

### ⚪ LOW (Investigation)
5. **Investigate t=1 skip logic** (Issue 5)

---

## Proposed Implementation Plan

### Phase 1: Critical Fix (shrinkA/shrinkB)
**Time estimate**: 5 minutes
**Risk**: Very low
**Impact**: High

```python
# In viz/app_streamlit.py, line 745-746
# Change from:
shrinkA=25,
shrinkB=25,

# To:
shrinkA=0.2,
shrinkB=0.2,
```

### Phase 2: Terminal Overlays
**Time estimate**: 30 minutes
**Risk**: Low
**Impact**: Medium-High

Two options:
1. **Simple**: Show all terminals distributed evenly
2. **Advanced**: Extract actual positions from perception system

Recommend starting with simple approach.

### Phase 3: Message Offset & Terminal Positioning
**Time estimate**: 1 hour
**Risk**: Medium
**Impact**: Medium

Refactor positioning logic to handle many terminals gracefully.

---

## Additional Observations

### Good Practices Observed
✅ Consistent color scheme for states across graph and scene  
✅ State sizes create visual hierarchy  
✅ Edge styling differentiates link types  
✅ Timeline scrubber for reviewing simulation history

### Code Quality Notes
- Well-structured code with clear sections
- Good use of Streamlit's layout features
- Comprehensive state management
- Could benefit from extracting positioning logic into helper functions

---

## Conclusion

The **primary issue** causing inaccurate animation is the **incorrect shrinkA/shrinkB values** (25 instead of ~0.2) in the message arrow drawing code. This causes arrows to be positioned completely wrong relative to the nodes.

**Secondary issues** include:
- Only 3 terminals shown on scene image (hardcoded)
- Potential overlaps when many terminals are generated
- Minor offset calculation inefficiency

The fix for the critical issue is trivial (2-line change) and will immediately improve the visualization accuracy. The secondary issues can be addressed in follow-up improvements.

---

## Next Steps

1. ✅ Complete investigation
2. ⬜ Implement critical fix (shrinkA/shrinkB)
3. ⬜ Test with various scenarios
4. ⬜ Implement terminal overlay improvements
5. ⬜ Add validation/testing code
6. ⬜ Document coordinate system for future maintenance
