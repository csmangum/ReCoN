# Request Confirmation Network (ReCoN) Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csmangum/ReCoN/blob/main/ReCoN_Key_Features_Demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/csmangum/ReCoN/HEAD?labpath=ReCoN_Key_Features_Demo.ipynb)

A faithful, comprehensive **Request Confirmation Network** (ReCoN) implementation based on the CoCoNIPS 2015 paper. Features active perception, hierarchical recognition, and temporal sequencing for intelligent object recognition.

## 🚀 Key Features

- **Active Perception**: Networks request only relevant information
- **Hierarchical Recognition**: Multi-level part-based object recognition
- **Temporal Sequencing**: Ordered execution via precedence links
- **Interactive Visualization**: Real-time network dynamics
- **Graph Validation**: Comprehensive validation of network structure and integrity
- **Extensible Architecture**: Clean separation for custom applications
- **Comprehensive Test Suite**: Broad tests across core, compiler, perception, metrics

## 📖 Documentation

- **[Complete Documentation](RECON_DOCUMENTATION.md)**: Comprehensive guide covering theory, implementation, and usage
- **[Original Paper](CoCoNIPS_2015_paper_6.pdf)**: Request Confirmation Networks for Active Object Recognition

## 🚀 Quickstart

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run interactive demo
streamlit run viz/app_streamlit.py
```

### CLI

```bash
# Show version and help
PYTHONPATH=. python3 scripts/recon_cli.py --version
PYTHONPATH=. python3 scripts/recon_cli.py -h

# Discover bundled sample scenes
PYTHONPATH=. python3 scripts/recon_cli.py --list-scenes

# Compile only (no stepping)
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --dry-run

# Run a YAML script for N steps and print JSON snapshot summary
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --steps 5 --deterministic --ret-feedback

# Override gate strengths and confirmation ratio
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --sur 0.25 --por 0.6 --confirm-ratio 0.7

# Dump to a file
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --steps 10 --out snapshot.json

# Validate compiled graph (non-zero exit on errors). Add --strict-activation to treat bounds as errors
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --validate --strict-activation

# Print comprehensive statistics and health score
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --stats

# Export GraphML for external tools (Gephi, yEd, NetworkX)
PYTHONPATH=. python3 scripts/recon_cli.py scripts/house.yaml --export-graphml house.graphml --dry-run

# Run comprehensive graph validation demo
python3 scripts/graph_validation_demo.py
```

Then click **Generate Scene** and **Run** to watch the active perception in action!

## Repo layout

```
recon_core/
  enums.py        # state/message/link enums
  graph.py        # Unit, Edge, Graph model
  engine.py       # propagation + state update rules
  learn.py        # (optional) tiny learning helpers for sur weights
  metrics.py      # (Day 6) metrics helpers and engine.stats accessors
perception/
  dataset.py      # synthetic 2D scenes with variety (houses, barns, occlusion)
  terminals.py    # comprehensive terminals (filters + SIFT-like + autoencoder + engineered [+ optional TinyCNN])
scripts/
  house.yaml                    # script → recon graph compiler input
  graph_validation_demo.py      # comprehensive graph validation demonstration
viz/
  app_streamlit.py    # interactive visualization
tests/
  test_engine.py      # smoke tests for engine transitions
```

## 🧠 Core Concepts

### Network Components
- **Script Units**: Orchestrate recognition activities and coordinate children
- **Terminal Units**: Detect basic features and provide sensory evidence
- **8-State FSM**: Each unit has states like INACTIVE, REQUESTED, CONFIRMED, FAILED
- **Typed Links**: SUB (evidence), SUR (requests), POR (sequencing), RET (feedback)

### Algorithm Phases
1. **Propagation**: Calculate activation flow using gate functions
2. **State Updates**: Process messages and update unit states
3. **Message Delivery**: Move messages between unit queues

### Active Perception
Unlike passive classifiers, ReCoN **actively requests** only relevant information, making recognition more efficient and selective.

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

If your environment cannot install heavy test dependencies, use the lightweight runner:

```bash
python3 run_tests.py
```

### Training flags (feature extractors)

- Set `RECON_TRAIN_AE=1` to enable autoencoder training for AE-based terminals (disabled by default).
- Set `RECON_TRAIN_CNN=1` to enable TinyCNN training for CNN-based terminals (disabled by default).

### Engine configuration

- Use `EngineConfig` to tune gate strengths and behavior:

```python
from recon_core.config import EngineConfig
from recon_core.engine import Engine

cfg = EngineConfig(
    sur_positive=0.25,
    por_positive=0.6,
    confirmation_ratio=0.7,
    deterministic_order=True,
    ret_feedback_enabled=True,
)
engine = Engine(graph, config=cfg)
```

**Metrics**

- Engine records metrics in `engine.snapshot()['stats']`:
  - `terminal_request_count`: total SUR requests to TERMINALs
  - `terminal_request_counts_by_id`: per-terminal request counts
  - `first_request_step`, `first_active_step`: script timing
  - `first_true_step`, `first_confirm_step`: key terminal/script events

- Convenience helpers in `recon_core.metrics`:
  - `binary_precision_recall(y_true, y_pred)`
  - `steps_to_first_true(engine, unit_id)`
  - `steps_to_first_confirm(engine, unit_id)`
  - `total_terminal_requests(engine)`
  - `terminal_request_counts_by_id(engine)`

Example:

```python
from recon_core.metrics import (
    binary_precision_recall,
    steps_to_first_true,
    steps_to_first_confirm,
    total_terminal_requests,
)

snap = engine.step(5)
print(snap['stats']['terminal_request_count'])
print(steps_to_first_confirm(engine, 'u_root'))
print(binary_precision_recall([1,0,1],[1,1,0]))
```

Configuration notes:

- Gate constants (SUB/SUR/POR/RET) default to `(+1/-1, +0.3/-0.3, +0.5/-0.5, +0.2/-0.5)` and are configurable via `EngineConfig`.
- Optional per-link minimal source activation thresholds can suppress propagation unless the source activation exceeds a chosen value: `sub_min_source_activation`, `sur_min_source_activation`, `por_min_source_activation`, `ret_min_source_activation` (all default to 0.0 to preserve baseline behavior).

Tests cover:
- Engine transitions, compact arithmetic, message passing, sequencing
- Script compiler (YAML→graph), POR sequencing, SUR/SUB wiring
- Perception pipeline (filters, SIFT-like, autoencoder, engineered terminals, optional TinyCNN)
- Metrics recording and convenience helpers
- Graph validation (cycles, link consistency, activation bounds, integrity, performance)

### Graph Validation Features

The ReCoN implementation includes comprehensive graph validation capabilities to ensure network integrity and performance:

```python
from recon_core.graph import Graph, Unit, Edge
from recon_core.enums import UnitType, LinkType

# Create and validate a graph
graph = Graph()
# ... add units and edges ...

# Quick validity check
if graph.is_valid():
    print("Graph structure is valid!")
else:
    print("Graph has validation issues")

# Comprehensive validation with detailed report
validation_results = graph.validate_all()
summary = graph.get_validation_summary(validation_results)
print(f"Found {summary['total_issues']} issues: {summary['errors']} errors, {summary['warnings']} warnings")

# Performance analysis
performance = graph.analyze_performance_metrics()
print(f"Health score: {graph.get_graph_statistics()['health_score']:.3f}")
```

**Validation Capabilities:**
- **Cycle Detection**: Identifies problematic cycles in SUB/SUR relationships
- **Link Consistency**: Validates proper link type usage (SUB from terminals, POR between scripts, etc.)
- **Unit Relationships**: Ensures terminals and scripts follow proper hierarchical patterns
- **Activation Bounds**: Checks activation levels and thresholds are within valid ranges
- **Graph Integrity**: Detects orphaned units, connectivity issues, and structural problems
- **Performance Metrics**: Analyzes complexity, efficiency, and identifies bottlenecks
- **Custom Validation**: Extensible framework for domain-specific validation rules

## 💡 Usage Examples

### Basic Network Creation
```python
from recon_core.graph import Graph, Unit, Edge
from recon_core.enums import UnitType, LinkType
from recon_core.engine import Engine

# Create a simple recognition network
g = Graph()
g.add_unit(Unit('detector', UnitType.TERMINAL, thresh=0.5))
g.add_unit(Unit('recognizer', UnitType.SCRIPT))

g.add_edge(Edge('detector', 'recognizer', LinkType.SUB))
g.add_edge(Edge('recognizer', 'detector', LinkType.SUR))

engine = Engine(g)
```

### Perception Pipeline
```python
from perception.dataset import make_varied_scene
from perception.terminals import comprehensive_terminals_from_image

# Generate diverse synthetic scenes
house_scene = make_varied_scene('house', size=64, noise=0.1)
barn_scene = make_varied_scene('barn', size=64) 
occluded_scene = make_varied_scene('occluded', size=64)


# Extract comprehensive features (21 terminals: 12 advanced + 4 AE + 5 engineered)
features = comprehensive_terminals_from_image(house_scene)
print(f"Basic: mean={features['t_mean']:.3f}, edges={features['t_vert']:.3f}")
print(f"SIFT: corners={features['t_corners']:.3f}, grad_mag={features['t_edges']:.3f}")
print(f"AE: {features['t_ae_0']:.3f}, {features['t_ae_1']:.3f}")
print(f"Extra: vsym={features['t_vsym']:.3f}, rect={features['t_rect']:.3f}")
```

## 🔧 Architecture Notes

- **Clean Implementation**: Faithful to the paper's algorithmic specifications
- **Extensible Design**: Easy to add custom terminal units or learning mechanisms
- **Modular Structure**: Clear separation between core algorithm and applications
- **Well-Tested**: Comprehensive test coverage ensures reliability

For detailed implementation notes, see [RECON_DOCUMENTATION.md](RECON_DOCUMENTATION.md).
