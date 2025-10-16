# ReCoN Streamlit Animation Investigation - Summary

**Date**: 2025-10-16  
**Investigator**: AI Assistant  
**Status**: ✅ Investigation Complete

---

## What Was Investigated

The Streamlit frontend application (`viz/app_streamlit.py`) was showing inaccurate animations of the ReCoN network, specifically:
- Message arrows appearing offset from nodes
- Extra elements not positioned where they should be
- General visual inaccuracies during simulation steps

---

## Root Cause Found

**🚨 CRITICAL BUG**: Message arrow positioning uses incorrect shrink values.

**Location**: `viz/app_streamlit.py`, lines 745-746

**The Problem**:
```python
shrinkA=25,  # ❌ WRONG - should be ~0.2
shrinkB=25,  # ❌ WRONG - should be ~0.2
```

**Why This Is Wrong**:
- The graph coordinate system is only ~3 units wide × 2.2 units tall
- Node positions are like: (0, 3), (-1, 2), (1, 2), etc.
- A shrink of 25 units pushes arrows completely off the graph!
- Should be ~0.2 units (matching visual node radius)

**Impact**: Message arrows appear nowhere near the nodes they're supposed to connect.

---

## Additional Issues Found

### 🟡 HIGH Priority
**Terminal Overlays Limited**: Only 3 hardcoded terminals (`t_mean`, `t_vert`, `t_horz`) show up on the scene image, even when many more terminals are active in the network.

### 🟢 MEDIUM Priority
**Message Offset Calculation**: Uses global message index instead of per-edge-pair index, causing inconsistent visual spacing.

**Dynamic Terminal Overlap**: When many terminals are generated, they may overlap in the network graph due to all being placed at the same y-coordinate.

### ⚪ LOW Priority
**Timeline Skip at t=1**: Code intentionally skips drawing messages at `timeline_idx == 1`. Reason unclear and needs investigation.

---

## The Fix Plan

### Immediate Fix (2 minutes)
Change lines 745-746 in `viz/app_streamlit.py`:
```python
# From:
shrinkA=25,
shrinkB=25,

# To:
shrinkA=0.2,
shrinkB=0.2,
```

This will immediately make message arrows appear correctly positioned.

### Follow-up Fixes (Optional)
1. Show all active terminals on scene, not just 3
2. Fix message offset to be per-edge-pair
3. Improve terminal layout to prevent overlaps
4. Investigate the t=1 skip logic

---

## Documents Created

1. **`STREAMLIT_ANIMATION_ISSUES_ANALYSIS.md`**
   - Detailed technical analysis of all 5 issues
   - Coordinate system breakdown
   - Visual impact assessment
   - Testing strategy

2. **`STREAMLIT_FIX_PLAN.md`**
   - Step-by-step implementation guide
   - Code snippets for all fixes
   - Priority order and time estimates
   - Risk assessment
   - Testing checklist

3. **`INVESTIGATION_SUMMARY.md`** (this file)
   - High-level overview for quick reference

---

## Quick Start: Apply the Critical Fix

```bash
# Edit the file
vim viz/app_streamlit.py

# Or using sed (backup created automatically)
sed -i.bak 's/shrinkA=25,/shrinkA=0.2,/' viz/app_streamlit.py
sed -i.bak 's/shrinkB=25,/shrinkB=0.2,/' viz/app_streamlit.py

# Test the fix
streamlit run viz/app_streamlit.py
```

Then in the app:
1. Click "🎲 Generate" to create a scene
2. Click "⏭️ Step" a few times
3. Verify message arrows now connect properly to nodes

---

## Visual Comparison

### Before Fix (Current State)
```
Node A          Node B
  ●               ●
                    
        ??????→      
                    
  (Arrow way off, might be invisible)
```

### After Fix (Expected)
```
Node A ────────→ Node B
  ●               ●
  
  (Arrow connects edge-to-edge)
```

---

## Key Takeaways

1. **Root cause**: Mathematical error in coordinate scaling (25 vs 0.2)
2. **Impact**: High - completely breaks message visualization
3. **Fix**: Trivial - 2-line change
4. **Risk**: Very low - simple constant change
5. **Additional issues**: 4 more issues identified, all lower priority

---

## Recommendations

### Immediate Action
✅ Apply the critical fix (shrinkA/shrinkB = 0.2)

### This Week
- Implement terminal overlay improvements
- Test with various scenarios

### Next Week
- Address medium-priority positioning issues
- Add validation tests

### Future
- Document coordinate systems
- Add visual regression tests
- Consider extracting helper functions

---

## Conclusion

The investigation successfully identified the root cause of the animation inaccuracies: **incorrect arrow shrink parameters**. A simple 2-line fix will resolve the critical visual issue. Additional improvements are documented for follow-up work.

The investigation also revealed the need for better coordinate system documentation and testing to prevent similar issues in the future.

---

## Questions?

Refer to the detailed documents:
- Technical details → `STREAMLIT_ANIMATION_ISSUES_ANALYSIS.md`
- Implementation guide → `STREAMLIT_FIX_PLAN.md`
