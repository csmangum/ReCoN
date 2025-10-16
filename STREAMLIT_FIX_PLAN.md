# ReCoN Streamlit Visualization - Fix Plan

## Executive Summary

Investigation of the Streamlit frontend animation inaccuracies revealed **5 issues**, with **1 critical bug** that causes message arrows to appear completely disconnected from nodes.

**Root Cause**: Message arrow shrink parameters are set to 25 data units when they should be ~0.2 data units, given the graph's coordinate system spans only 3×2.2 units.

---

## Issues Summary

| Priority | Issue | Location | Impact | Fix Time |
|----------|-------|----------|--------|----------|
| 🔴 CRITICAL | Message arrow shrink values too large | `viz/app_streamlit.py:745-746` | Arrows offset/invisible | 2 min |
| 🟡 HIGH | Only 3 hardcoded terminals shown on scene | `viz/app_streamlit.py:404-427` | Missing visual feedback | 30 min |
| 🟢 MEDIUM | Message offset uses global index | `viz/app_streamlit.py:717-719` | Minor visual noise | 15 min |
| 🟢 MEDIUM | Dynamic terminals may overlap | `viz/app_streamlit.py:539-561` | Cluttered when many terminals | 45 min |
| ⚪ LOW | Special case skip at t=1 | `viz/app_streamlit.py:652,788` | Investigation needed | TBD |

---

## Detailed Fixes

### Fix 1: Message Arrow Shrink Values (CRITICAL)

**File**: `viz/app_streamlit.py`  
**Lines**: 745-746

**Current code**:
```python
shrinkA=25,  # Shrink from start point
shrinkB=25,  # Shrink from end point
```

**Fixed code**:
```python
shrinkA=0.2,  # Shrink from start point (~node radius)
shrinkB=0.2,  # Shrink from end point (~node radius)
```

**Rationale**: 
- Graph coordinate system: x ∈ [-1.5, 1.5], y ∈ [0.8, 3.0]
- Node visual radius ≈ 0.15-0.25 data units (based on scatter sizes 1800-3300)
- Shrink should match visual node size, not be orders of magnitude larger

**Testing**: After fix, message arrows should:
- Start at edge of sender node
- End at edge of receiver node
- Be visible and properly positioned

---

### Fix 2: Terminal Scene Overlays (HIGH)

**File**: `viz/app_streamlit.py`  
**Lines**: 404-427

**Problem**: Only shows `t_mean`, `t_vert`, `t_horz` even when many more terminals exist.

**Proposed fix**:
```python
# Add terminal detection overlays using live activations
# Show ALL active terminals, not just the 3 hardcoded ones
terminal_base_positions = {
    "t_mean": (32, 32),
    "t_vert": (45, 32),
    "t_horz": (20, 32),
}

# Get all terminal units
all_terminals = [
    (uid, u) for uid, u in st.session_state.sim.graph.units.items() 
    if u.kind == UnitType.TERMINAL
]

# Distribute terminals across scene if not in base positions
if len(all_terminals) > 3:
    # Create grid layout for additional terminals
    grid_size = int(np.ceil(np.sqrt(len(all_terminals))))
    grid_positions = []
    for i in range(len(all_terminals)):
        row = i // grid_size
        col = i % grid_size
        x = 10 + (col * 54 // grid_size)
        y = 10 + (row * 54 // grid_size)
        grid_positions.append((x, y))
    
    for idx, (term_name, term_unit) in enumerate(all_terminals):
        if term_name in terminal_base_positions:
            x, y = terminal_base_positions[term_name]
        else:
            x, y = grid_positions[idx]
        
        term_value = float(getattr(term_unit, "a", 0.0))
        # ... rest of overlay drawing code
```

**Alternative (simpler)**: Place additional terminals along edges:
```python
# Show first 3 at base positions, rest distributed along edges
additional_terms = [uid for uid in all_terminals if uid not in terminal_base_positions]
for idx, term_name in enumerate(additional_terms):
    x = 5 + (idx % 6) * 10  # Along left/right edges
    y = 5 + (idx // 6) * 10  # Multiple rows if needed
    # ... draw overlay
```

---

### Fix 3: Message Offset Calculation (MEDIUM)

**File**: `viz/app_streamlit.py`  
**Lines**: 717-719

**Current code**:
```python
offset = messages.index((sender, receiver, msg)) * 0.05
```

**Fixed code**:
```python
# Calculate offset based on how many messages between this specific pair
same_pair_messages = [
    (s, r, m) for s, r, m in messages 
    if s == sender and r == receiver
]
offset = same_pair_messages.index((sender, receiver, msg)) * 0.05
```

**Rationale**: Offset should be consistent for all messages between the same node pair, not based on position in global message list.

---

### Fix 4: Dynamic Terminal Positioning (MEDIUM)

**File**: `viz/app_streamlit.py`  
**Lines**: 549-561

**Current code**:
```python
if parent_id in pos:
    parent_x, _ = pos[parent_id]
    xs = np.linspace(-0.8, 0.8, num=len(terms)) if len(terms) > 1 else [0.0]
    for dx, n in zip(xs, terms):
        pos[n] = (float(parent_x + dx), 0.9)
```

**Issues**:
1. All at same y=0.9 causes overlaps
2. Range [-0.8, 0.8] insufficient for many terminals
3. Node labels may overlap

**Fixed code**:
```python
if parent_id in pos:
    parent_x, _ = pos[parent_id]
    
    # Use multiple rows if too many terminals
    max_per_row = 5
    num_rows = (len(terms) + max_per_row - 1) // max_per_row
    
    for idx, n in enumerate(terms):
        row = idx // max_per_row
        col = idx % max_per_row
        row_size = min(max_per_row, len(terms) - row * max_per_row)
        
        # Center each row and add vertical spacing
        xs = np.linspace(-0.8, 0.8, num=row_size)
        dx = xs[col]
        dy = 0.9 - row * 0.3  # Stack vertically with spacing
        
        pos[n] = (float(parent_x + dx), float(dy))
```

---

### Fix 5: Investigate t=1 Skip (LOW)

**File**: `viz/app_streamlit.py`  
**Lines**: 652, 788

**Investigation needed**: Why are messages skipped at `timeline_idx == 1`?

Possible reasons:
1. Workaround for initialization artifacts
2. Messages at t=1 are duplicates or invalid
3. Historical bug fix that may no longer be needed

**Action**: Add logging and test to understand this behavior before removing.

---

## Implementation Order

### Phase 1: Immediate (Today)
1. ✅ Complete investigation
2. ⬜ **Implement Fix 1** (critical shrink values) - 2 min
3. ⬜ **Test Fix 1** - 10 min

### Phase 2: High Priority (This Week)
4. ⬜ **Implement Fix 2** (terminal overlays) - 30 min
5. ⬜ **Test Fix 2** with many terminals - 15 min

### Phase 3: Medium Priority (Next Week)
6. ⬜ **Implement Fix 3** (message offset) - 15 min
7. ⬜ **Implement Fix 4** (terminal positioning) - 45 min
8. ⬜ **Test Fixes 3-4** - 20 min

### Phase 4: Investigate (Future)
9. ⬜ **Investigate Fix 5** (t=1 skip) - TBD

---

## Testing Checklist

After each fix, verify:

- [ ] **Message arrows**: Connect properly to node edges
- [ ] **Message arrows**: Visible and not offset
- [ ] **Message arrows**: Smooth animation
- [ ] **Terminal overlays**: All terminals shown on scene
- [ ] **Terminal overlays**: Colors match network graph states
- [ ] **Terminal positioning**: No overlaps in network graph
- [ ] **Terminal positioning**: Labels readable
- [ ] **Overall**: No regressions in other features
- [ ] **Performance**: No slowdowns with many nodes/messages

---

## Risk Assessment

### Fix 1 (Critical - shrink values)
- **Risk**: Very Low
- **Reasoning**: Simple constant change, well-understood coordinate system
- **Rollback**: Trivial (change back to 25)

### Fix 2 (High - terminal overlays)
- **Risk**: Low
- **Reasoning**: Additional rendering code, doesn't change existing logic
- **Rollback**: Easy (keep hardcoded positions)

### Fix 3 (Medium - message offset)
- **Risk**: Low
- **Reasoning**: Logic change but isolated to one calculation
- **Rollback**: Easy (revert to global index)

### Fix 4 (Medium - terminal positioning)
- **Risk**: Medium
- **Reasoning**: Complex layout logic, edge cases with many terminals
- **Rollback**: Easy but may need iteration

### Fix 5 (Low - t=1 skip)
- **Risk**: Unknown
- **Reasoning**: Removing safeguard without understanding original issue
- **Rollback**: Trivial (keep the skip)

---

## Success Metrics

After implementing all fixes:

1. **Accuracy**: Message arrows visually connect nodes properly (0 offset)
2. **Completeness**: All active terminals shown on scene and graph
3. **Clarity**: No overlapping nodes or labels
4. **Performance**: No degradation in render time
5. **User Feedback**: Users report improved visualization quality

---

## Additional Recommendations

### Documentation
- Add comments explaining coordinate system ranges
- Document why specific values (0.2 for shrink) were chosen
- Add diagram showing coordinate systems (scene vs. graph)

### Code Quality
- Extract positioning logic into helper functions
- Add type hints for coordinate values
- Consider dataclass for position tuples

### Testing
- Add unit tests for coordinate transformations
- Add visual regression tests (screenshot comparison)
- Add edge case tests (1 terminal, 100 terminals, etc.)

### Future Enhancements
- Extract actual terminal positions from perception system
- Implement zoom/pan for large graphs
- Add animation speed control
- Highlight active message flows with color gradients

---

## References

- **Investigation Report**: `STREAMLIT_ANIMATION_ISSUES_ANALYSIS.md`
- **Main Code**: `viz/app_streamlit.py`
- **Related**: `recon_core/engine.py` (message system)
- **Related**: `perception/terminals.py` (terminal generation)
