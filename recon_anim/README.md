## ReCoN + Manim Integration

This package integrates `recon_core` with Manim to render technically accurate, stepwise animations of ReCoN graph activations and states.

## Directory layout

- recon_anim/
  - models/: events protocol, graph spec conversion, graph state diffs
  - adapters/: live engine stepper and JSONL event source
  - script/: event→step compiler (timestamp-aware)
  - utils/: layout, normalization, mobject helpers
  - scenes/: base mixin and activation graph scene
  - runner/: programmatic/CLI runner
  - examples/: toy assets

## Event protocol

- GraphDeclared(graph, seed=0, units?)
- StepStart(step_index, t?)
- NodeActivation(node_id, value, t?, value_min?, value_max?, units?)
- NodeState(node_id, state, t?)
- EdgeFlow(edge_id, value, t?, value_min?, value_max?, units?)
- NodeLabel(node_id, text)
- EdgeWeight(edge_id, value)
- StepEnd(step_index, t?)
- RunMetadata(key, value)

SceneStep(idx, duration, events) batches events between StepStart and StepEnd. Duration prefers timestamp deltas when present.

## Adapters

- EngineStepper(g, EngineConfig): emits per-step truthful activations/states using Engine.t
- JsonlEventSource(path): replays events from a JSONL log (each line has a type field)

## Scenes

- ReconSceneMixin: builds graph, consumes compiled steps, applies updates, plays with per-step run_time
- ActivationGraphScene: renders from an attached EngineStepper (live) or an event source (offline)

## Usage (programmatic)

Call the runner helper from Python to render the activation scene for a YAML graph:

- render_scene(ActivationGraphScene, "recon_anim/examples/toy_graph.yaml", quality="ql", preview=True, time_scale=1.0)

To play back a JSONL event log instead of simulating the engine, use the CLI runner (after extending with --events):

- python -m recon_anim.runner.render --scene ActivationGraphScene --events recon_anim/examples/toy_run.jsonl --quality ql --preview --time-scale 1.0

## Accuracy guardrails

- Use raw engine values; normalize only in utils/normalization.py
- Prefer event timestamps for timing; fall back to default step duration
- Deterministic layouts with fixed seed (fallback circle layout if NetworkX is unavailable)

## Extensibility

- Add scenes in scenes/ using the same event protocol and mixin
- Add adapters without changing scenes
- Extend apply_step for richer transitions, legends, overlays