# Request Confirmation Network (ReCoN) Implementation

A faithful, comprehensive **Request Confirmation Network** (ReCoN) implementation based on the CoCoNIPS 2015 paper. Features active perception, hierarchical recognition, and temporal sequencing for intelligent object recognition.

## 🚀 Key Features

- **Active Perception**: Networks request only relevant information
- **Hierarchical Recognition**: Multi-level part-based object recognition
- **Temporal Sequencing**: Ordered execution via precedence links
- **Interactive Visualization**: Real-time network dynamics
- **Extensible Architecture**: Clean separation for custom applications
- **Comprehensive Testing**: Full test coverage for reliability

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
  terminals.py    # comprehensive terminal features (filters + SIFT + autoencoder)
scripts/
  house.yaml      # script → recon graph compiler input
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

**Metrics (Day 6)**

- Engine now records metrics in `engine.snapshot()['stats']`:
  - `terminal_request_count`: total SUR requests to TERMINALs
  - `terminal_request_counts_by_id`: per-terminal request counts
  - `first_true_step`, `first_confirm_step`: timing of key events

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

**44+ tests** covering:
- **18 synthetic scene tests**: Drawing primitives, house/barn generation, occlusion, variations
- **26 terminal feature tests**: Basic filters, SIFT-like features, autoencoder, integration
- **Original core tests**: State transitions, message passing, inhibition, temporal sequencing

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

# Extract comprehensive features (16 terminals)
features = comprehensive_terminals_from_image(house_scene)
print(f"Basic: mean={features['t_mean']:.3f}, edges={features['t_vert']:.3f}")
print(f"SIFT: corners={features['t_corners']:.3f}, grad_mag={features['t_edges']:.3f}")
print(f"Autoencoder: {features['t_ae_0']:.3f}, {features['t_ae_1']:.3f}")
```

## 🔧 Architecture Notes

- **Clean Implementation**: Faithful to the paper's algorithmic specifications
- **Extensible Design**: Easy to add custom terminal units or learning mechanisms
- **Modular Structure**: Clear separation between core algorithm and applications
- **Well-Tested**: Comprehensive test coverage ensures reliability

For detailed implementation notes, see [RECON_DOCUMENTATION.md](RECON_DOCUMENTATION.md).
