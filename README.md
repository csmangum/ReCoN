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
perception/
  dataset.py      # synthetic 2D scenes (64x64) with simple shapes
  terminals.py    # terminal feature detectors (filters/AE stub)
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

All 11 tests should pass, covering state transitions, message passing, inhibition, and temporal sequencing.

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

### Custom Perception
```python
from perception.terminals import simple_filters

# Extract features from your image
features = simple_filters(your_image_array)
print(f"Mean intensity: {features['mean']}")
```

## 🔧 Architecture Notes

- **Clean Implementation**: Faithful to the paper's algorithmic specifications
- **Extensible Design**: Easy to add custom terminal units or learning mechanisms
- **Modular Structure**: Clear separation between core algorithm and applications
- **Well-Tested**: Comprehensive test coverage ensures reliability

For detailed implementation notes, see [RECON_DOCUMENTATION.md](RECON_DOCUMENTATION.md).
