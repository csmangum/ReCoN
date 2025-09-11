## Challenge Option 5: Request Confirmation Networks

Build a working implementation and dynamic visualization of a Request Confirmation Network (ReCoN) as described in this paper: [https://ceur-ws.org/Vol-1583/CoCoNIPS\_2015\_paper\_6.pdf](https://ceur-ws.org/Vol-1583/CoCoNIPS_2015_paper_6.pdf)   
Demonstrate the use of a ReCoN to solve an interesting representation problem. Bonus for implementing additional components not covered in the paper, such as learning new representation, conversion into different formats, etc.  
What we want to see: Ability to understand and implement a novel theoretical idea, translating between representation and user interface.

# What ReCoN is (in one paragraph)

A Request Confirmation Network is a spreading-activation graph whose nodes are **stateful units** (script or terminal) connected by **typed links**: `sub/sur` (part–whole) and `por/ret` (successor–predecessor). Each unit is a tiny finite-state machine (`inactive, requested, active, suppressed, waiting, true, confirmed, failed`) that exchanges a small alphabet of messages—*request, wait, confirm, inhibit-request, inhibit-confirm*—so a **script executes itself** without a central controller. Activations can also carry real-valued signals (e.g., feature strengths), letting you combine bottom-up cues with top-down hypotheses during request/confirmation phases.&#x20;

---

# Your demo concept: “Active Parts-of Confirmation in 2D Scenes”

**Representation problem:** Given a 2D scene with simple parts (edges/corners/colored blobs), **represent composite objects as scripts** (hierarchies + sequences). Use ReCoN to *actively* verify a hypothesis (“there is a ‘house’ here”) by requesting child features and confirming them in the right order and configuration.

* **Terminal nodes (sensors):** local feature detectors (SIFT-lite, or tiny CNN/autoencoder features).
* **Script nodes:** OR/AND of parts (roof, body, door), plus sequential checks (e.g., “scan left→right for edges, then verify roof above body”).
* **Why it’s a good fit:** It mirrors the paper’s “active perception” (with request/confirm phases) but on a clean synthetic dataset you control; it clearly demonstrates *translation between representation and UI*.&#x20;

**Stretch (optional):** swap the 2D scene for your **AgentFarm gridworld**—terminal nodes = resource/sensor signals; scripts = multi-step behaviors (“approach, harvest, deposit”), confirmed by observed state changes.

---

# Minimal ReCoN core (Python)

* **Data model**

  * `Unit` (id, type: script|terminal, state, activation vector)
  * `Edge` (u→v, type: sub|sur|por|ret, weight vector)
* **States & messages** (enum): as per paper’s table; step function emits/consumes messages.&#x20;
* **Update cycle (discrete time)**

  1. **Propagation:** `z = W · a` grouped by link type.
  2. **Node update:** apply per-link/node rules implementing request/confirm and inhibition (compact arithmetic form).
     (Paper shows a compact definition using per-gate functions for `gen/por/ret/sub/sur`—perfect to reproduce.)&#x20;
* **Learning (bonus)**

  * Learn `sur` weights as a linear combiner of child evidence during confirmation (logistic/least-squares); backprop through unrolled steps if you want flair. Paper okays real-valued activations for associative tasks.&#x20;

---

# Dynamic visualization (your “wow”)

**Two panes, one timeline scrubber:**

1. **Graph pane (D3/Canvas):**

   * Nodes colored by state (inactive gray, requested blue, waiting amber, true green, confirmed teal, failed red, suppressed purple).
   * Animated message arrows: request (thin up), wait (up to parent), confirm (thick up), inhibit-request (lateral), inhibit-confirm (reverse lateral).
   * Hover shows activations and child contributions.

2. **Scene pane:**

   * Current fovea/patch (or gridworld cell) with scan path; overlays for detected terminals; per-part confidence bars.

**Tech choices:**

* Fast path: **Streamlit** + `networkx/pyvis` (or `bokeh`) for the graph and `numpy/opencv` for scene; keeps everything in Python.
* Fancy path: **React + D3** frontend; Python backend via **FastAPI**; share a `/step` endpoint streaming states via websockets.

---

# Dataset & scripts

* **Synthetic scenes:** render 64×64 canvases with primitive shapes; vary noise, size, occlusion.
* **Terminals:** tiny patch encoder (simple denoising autoencoder or hand-engineered filters) → feature activations. (The paper used a denoising AE over Minecraft projections; we’ll keep it lightweight.)&#x20;
* **Scripts:** YAML or JSON:

  ```yaml
  object: house
  children:
    - name: roof
      type: AND
      parts: [triangle]
      relation: above: body
    - name: body
      type: AND
      parts: [rectangle]
    - name: door
      type: OR
      parts: [dark-rect, arch]
      relation: inside: body
  sequence:
    - scan_edges_left_to_right
    - verify_roof_above_body
    - verify_door_inside_body
  ```

  Converted into ReCoN nodes/edges: `sub/sur` for hierarchy; `por/ret` for ordered steps.

---

# Success criteria to show in your write-up

* **Correctness:** For scenes with a house, root node ends *confirmed*; for foils, ends *failed*—with clear causal trace (which child failed / which inhibition fired).&#x20;
* **Active perception:** Show scan sequence driven by *requests*; not all terminals fire—only those asked for.
* **Representational power:** Swap scripts at runtime (e.g., “barn” vs “house”) without retraining terminals; demonstrate reuse of parts.
* **Learning add-on:** After a few examples, `sur` weights shift to favor more diagnostic parts.

---

# Repo scaffold

```
recon-demo/
  README.md
  /recon_core/
    __init__.py
    enums.py            # states, messages, link types
    graph.py            # Unit, Edge, Graph
    engine.py           # step(), propagate(), apply_rules()
    learn.py            # optional weight updates
    tests/
  /perception/
    dataset.py          # synthetic scenes
    terminals.py        # AE/features or filters
  /scripts/
    house.yaml
    ...
  /viz/
    app_streamlit.py    # fast path UI
    # or /web (React) + backend/
```

---

# 7-day sprint (part-time)

* **Day 1:** Parse paper, lock API (Unit/Edge/Engine). Build state machine + message passing, unit tests for state transitions and inhibitions.&#x20;
* **Day 2:** Implement compact arithmetic update (propagation + fnode/fgate rules). Golden tests vs. hand-crafted micro-graphs.&#x20;
* **Day 3:** Build synthetic scene generator + simple terminals (filters or tiny AE).
* **Day 4:** Script compiler (YAML→ReCoN graph). House/barn examples.
* **Day 5:** Streamlit UI: graph states, message animations, scene fovea path; “step/run/pause/reset” controls.
* **Day 6:** Metrics (precision/recall, steps to confirm, #terminal queries); optional learning for `sur` weights.
* **Day 7:** Polish docs, record 2–3 min demo, prepare submission zip + repo.

---

# Write-up structure (what CIMC wants to read)

1. **Problem & intuition:** Why self-executing scripts matter for bridging symbolic structure and neural signals.
2. **ReCoN recap:** States, messages, link types, and compact update (with one figure from your UI).&#x20;
3. **Design choices:** Terminal feature design; why `por/ret` for sequencing; how you encode OR/AND via structure.&#x20;
4. **Interface:** How the visualization reveals flow of control and evidence.
5. **Results:** qualitative traces + quantitative metrics.
6. **Limitations & next:** temporal motifs, noise robustness, learning `por/ret` structure from data; swapping in AgentFarm sensors.
7. **Related work:** Place ReCoN in neuro-symbolic landscape (e.g., KBANN lineage, lifted relational NNs).&#x20;

---

# Small but meaningful bonuses

* **Converter:** JSON↔ReCoN graph (and GraphML export); import/export earns bonus points (conversion between formats).
* **Learning:** simple online update for `sur` weights during confirmation.
* **AgentFarm hook:** a second demo where terminals are grid signals and the script validates a two-step behavior.
