# Request Confirmation Network (ReCoN) Documentation

## Overview

This document provides a comprehensive guide to the Request Confirmation Network (ReCoN) implementation, based on the algorithm described in the CoCoNIPS 2015 paper "Request Confirmation Networks for Active Object Recognition".

## What is ReCoN?

ReCoN is a **spreading-activation graph** whose nodes are **stateful units** (script or terminal) connected by **typed links**. Each unit is a tiny finite-state machine that exchanges messages to **actively coordinate recognition** without a central controller.

### Key Innovation: Active Perception

Unlike traditional passive classifiers, ReCoN implements **active perception**:
- The network **requests** specific information when needed
- Only relevant features are computed and processed
- Recognition proceeds through **hierarchical confirmation** and **temporal sequencing**

## Core Components

### Units

ReCoN networks consist of two types of computational units:

#### Script Units
- **Purpose**: Orchestrate and coordinate recognition activities
- **Function**: Manage child units, handle temporal sequencing, aggregate evidence
- **Example**: A "house" script that coordinates recognition of roof, body, and door parts

#### Terminal Units
- **Purpose**: Interface with sensory/perceptual inputs
- **Function**: Detect basic features and provide evidence to parent scripts
- **Example**: Edge detectors, intensity sensors, or feature extractors

### Unit States

Each unit maintains a **finite state machine** with 8 possible states:

| State | Description |
|-------|-------------|
| `INACTIVE` | Unit is dormant, no activity |
| `REQUESTED` | Unit has received a request for activation |
| `WAITING` | Unit is waiting for dependencies or external conditions |
| `ACTIVE` | Unit is actively processing/confirming |
| `TRUE` | Terminal unit has detected its feature/pattern (activation ≥ threshold) |
| `CONFIRMED` | Script unit has sufficient confirmation from children |
| `FAILED` | Unit has failed validation or encountered error |
| `SUPPRESSED` | Unit has been inhibited by conflicting information |

### Link Types

Units are connected by **directed, typed links** that determine information flow:

#### SUB (Subordinate) Links
- **Direction**: Child → Parent
- **Purpose**: Evidence propagation (bottom-up)
- **Examples**:
  - Terminal provides confirmation evidence to its parent script
  - Child script provides aggregated evidence to its parent script

#### SUR (Superior) Links
- **Direction**: Parent → Child
- **Purpose**: Request propagation (top-down)
- **Example**: Parent script requests activation from child units

#### POR (Precedence) Links
- **Direction**: Predecessor → Successor
- **Purpose**: Temporal sequencing
- **Example**: "Roof recognition" must complete before "door recognition"

#### RET (Return) Links
- **Direction**: Successor → Predecessor
- **Purpose**: Temporal feedback
- **Example**: Successor provides completion feedback to predecessor

### Messages

Units communicate via an **asynchronous message system**:

| Message | Purpose |
|---------|---------|
| `REQUEST` | Request a unit to activate |
| `CONFIRM` | Confirm successful completion/activation |
| `WAIT` | Signal to wait for further instructions |
| `INHIBIT_REQUEST` | Block/suppress a request |
| `INHIBIT_CONFIRM` | Invalidate a confirmation |

## Algorithm Overview

### Four-Phase Update Cycle

Each simulation time step consists of four phases:

#### 1. Propagation Phase
```python
# Calculate activation deltas using gate functions (compact arithmetic)
delta = self._propagate()
```

The **compact arithmetic propagation** computes how activation flows through each link type using **per-gate functions**. This implementation chooses concrete defaults (configurable) that reflect the qualitative behavior described in the paper:

- **SUB Gate**: `TRUE/CONFIRMED` → +1.0, `FAILED` → -1.0
- **SUR Gate**: `REQUESTED/ACTIVE` → +0.3, `FAILED` → -0.3
- **POR Gate**: `CONFIRMED` → +0.5, `FAILED` → -0.5
- **RET Gate**: `FAILED` → -0.5, `CONFIRMED` → +0.2

All values are configurable via `EngineConfig` to support experiments.
In addition, optional minimal source activation thresholds per link type can suppress gate output when the source unit's activation is below a configured value (defaults are 0.0 for all link types).

#### 2. State Update Phase
```python
# Process messages and update unit states with soft activation integration
self._update_states(delta)
```

- **Message Processing**: Handle incoming messages (REQUEST, CONFIRM, etc.)
- **Soft Activation Update**: Apply propagation deltas with configurable gain: `a = clip(a + config.activation_gain * delta, 0, 1)` (default gain = 0.8)
- **State Transitions**: Update unit states based on current activation and message history

#### 3. Message Delivery Phase
```python
# Deliver messages from outboxes to inboxes
self._deliver_messages()
```

#### 4. Second Message Processing
```python
# Process newly delivered messages in the same step
for uid in self.g.units:
    self._process_messages(uid)
```

## Implementation Architecture

### Core Modules

```
recon_core/
├── __init__.py
├── enums.py          # State, Message, LinkType, UnitType definitions
├── graph.py          # Unit, Edge, Graph data structures
├── engine.py         # Core ReCoN simulation engine
└── learn.py          # Optional learning utilities

perception/
├── __init__.py
├── dataset.py        # Synthetic scene generation
└── terminals.py      # Feature extraction for terminals

viz/
├── __init__.py
└── app_streamlit.py  # Interactive visualization

tests/
├── __init__.py
└── test_engine.py    # Comprehensive test suite
```

### Metrics (Day 6)

The engine records runtime metrics to evaluate active perception efficiency and timing. These are exposed via `Engine.snapshot()` under the `stats` key and via convenience helpers in `recon_core.metrics`.

Recorded fields in `engine.stats`:

- `terminal_request_count`: total number of SUR requests sent to TERMINAL units
- `terminal_request_counts_by_id`: mapping `terminal_id -> count`
- `first_request_step`: first time step a unit entered `REQUESTED`
- `first_active_step`: first time step a unit entered `ACTIVE`
- `first_true_step`: first time step a TERMINAL entered `TRUE`
- `first_confirm_step`: first time step a SCRIPT entered `CONFIRMED`

Convenience API (`recon_core.metrics`):

```python
from recon_core.metrics import (
    binary_precision_recall,
    steps_to_first_true,
    steps_to_first_confirm,
    total_terminal_requests,
    terminal_request_counts_by_id,
)

snap = engine.step(5)
print(snap['stats']['terminal_request_count'])
print(steps_to_first_confirm(engine, 'u_root'))

# Evaluate predictions
m = binary_precision_recall([1,0,1], [1,1,0])
print(m['precision'], m['recall'])
```

Implementation notes:

- SUR requests to children are issued once per parent script activation episode to avoid over-counting.
- Timing fields record the first step at which the state transition occurred and remain stable thereafter.

### Optional Propagation Thresholds

For additional control over when gates emit signals, `EngineConfig` exposes per-link minimal source activation thresholds. When set above 0, a gate output is suppressed unless the source unit's activation `a` meets the threshold:

- `sub_min_source_activation`
- `sur_min_source_activation`
- `por_min_source_activation`
- `ret_min_source_activation`

Defaults are 0.0 to preserve paper-faithful behavior; set higher to require stronger source activation before propagating.

### Key Classes

#### Unit Class
```python
@dataclass
class Unit:
    id: str                           # Unique identifier
    kind: UnitType                   # SCRIPT or TERMINAL
    state: State = State.INACTIVE    # Current state
    a: float = 0.0                   # Activation level (0.0-1.0)
    thresh: float = 0.5             # Confirmation threshold
    meta: dict = field(default_factory=dict)  # Additional metadata

    # Message queues for asynchronous communication
    inbox: List[Tuple[str, Message]] = field(default_factory=list)
    outbox: List[Tuple[str, Message]] = field(default_factory=list)
```

#### Graph Class
```python
class Graph:
    def __init__(self):
        self.units: Dict[str, Unit] = {}
        self.out_edges: Dict[str, List[Edge]] = {}
        self.in_edges: Dict[str, List[Edge]] = {}

    def add_unit(self, u: Unit) -> None:
        """Add a unit to the network"""

    def add_edge(self, e: Edge) -> None:
        """Add a directed edge between units"""

    def sub_children(self, parent_id: str) -> List[str]:
        """Get child units connected via SUB links"""

    def por_successors(self, u_id: str) -> List[str]:
        """Get successor units connected via POR links"""

    def to_networkx(self) -> "nx.DiGraph":
        """Convert ReCoN graph to NetworkX DiGraph for analysis/export"""

    def export_graphml(self, filepath: str) -> None:
        """Export graph to GraphML format for external tools (Gephi, yEd, etc.)"""
```

#### Engine Class
```python
class Engine:
    def __init__(self, g: Graph, config: EngineConfig | None = None):
        self.g = g
        self.t = 0

    def step(self, n: int = 1) -> dict:
        """Advance simulation by n time steps"""

    def reset(self) -> None:
        """Reset network to initial state"""

    def snapshot(self) -> dict:
        """Create current network state snapshot"""
```

## Graph Validation System

The ReCoN implementation includes a comprehensive graph validation system to ensure network integrity, detect structural issues, and optimize performance. This system provides both automated validation and detailed performance analysis.

### Validation Overview

The validation system consists of multiple validation modules that can be run individually or collectively:

```python
from recon_core.graph import Graph, Unit, Edge
from recon_core.enums import UnitType, LinkType

# Create a graph with potential issues
graph = Graph()
# ... add units and edges ...

# Run all validations
validation_results = graph.validate_all(strict_activation=True)

# Get summary
summary = graph.get_validation_summary(validation_results)
print(f"Issues found: {summary['total_issues']} total, {summary['errors']} errors, {summary['warnings']} warnings")

# Check overall validity
is_valid = graph.is_valid()
print(f"Graph is {'valid' if is_valid else 'invalid'}")
```

### Cycle Detection

Detects problematic cycles in the network that could cause infinite loops or unexpected behavior:

```python
# Check for cycles in all link types
all_cycles = graph.validate_cycles()

# Check specific link type
sub_cycles = graph.validate_cycles(LinkType.SUB)
sur_cycles = graph.validate_cycles(LinkType.SUR)
por_cycles = graph.validate_cycles(LinkType.POR)
ret_cycles = graph.validate_cycles(LinkType.RET)
```

**Cycle Detection Rules:**
- **SUB cycles**: Generally problematic (infinite evidence propagation)
- **SUR cycles**: Generally problematic (infinite request propagation)
- **POR cycles**: May be acceptable for circular temporal processes
- **RET cycles**: Expected and normal for feedback loops

### Link Consistency Validation

Validates that link types are used appropriately based on unit types:

```python
link_issues = graph.validate_link_consistency()
```

**Link Type Rules:**
- **SUB links**: Should target scripts (evidence flows from terminals or child scripts → scripts)
- **SUR links**: Should connect scripts → children (request flow)
- **POR links**: Should only connect scripts ↔ scripts (temporal precedence)
- **RET links**: Should only connect scripts ↔ scripts (temporal feedback)
- **Edge weights**: Recommended range [-2.0, 2.0] to support inhibitory and learned connections

### Unit Relationship Validation

Ensures units follow proper hierarchical and structural patterns:

```python
relationship_issues = graph.validate_unit_relationships()
```

**Unit Relationship Rules:**
- **Terminals**: Should only participate in SUB (evidence) and SUR (request) relationships
- **Scripts**: Should form proper hierarchical structures without direct SUB links
- **Root Scripts**: Should exist (scripts with no incoming SUB links)
- **Reachability**: All units should be reachable from root scripts

### Activation Bounds Validation

Validates activation levels and thresholds are within proper bounds:

```python
activation_issues = graph.validate_activation_bounds(strict=True)
```

**Activation Validation Rules:**
- Activation levels must be within [0.0, 1.0]
- Threshold values must be within [0.0, 1.0]
- Consistency checks for threshold/activation relationships
- Warning for unusual threshold/activation combinations

### Graph Integrity Validation

Comprehensive structural validation of the network:

```python
integrity_issues = graph.validate_graph_integrity()
```

**Integrity Checks:**
- **Orphaned Units**: Units with no connections
- **Invalid Edges**: Edges referencing non-existent units
- **Connectivity Issues**: Disconnected graph components
- **Structure Warnings**: Terminals without parent scripts

### Performance Metrics Analysis

Analyzes graph complexity, efficiency, and potential bottlenecks:

```python
performance = graph.analyze_performance_metrics()
```

**Performance Metrics:**
- **Structure Metrics**: Unit/edge counts, terminal ratio, connectivity
- **Complexity Metrics**: Degree distribution, link type distribution, max degree
- **Efficiency Indicators**: Connected components, isolated units, hierarchy depth
- **Bottleneck Warnings**: High degree nodes, deep hierarchies, cycles

### Custom Validation Rules

Extensible framework for domain-specific validation:

```python
def validate_max_fan_out(graph):
    """Custom rule: Check that no unit has more than 5 outgoing connections."""
    issues = {}
    max_fan_out = 5

    for unit_id, edges in graph.out_edges.items():
        if len(edges) > max_fan_out:
            if 'fan_out_issues' not in issues:
                issues['fan_out_issues'] = []
            issues['fan_out_issues'].append(
                f"Unit '{unit_id}' has {len(edges)} outgoing connections (max allowed: {max_fan_out})"
            )
    return issues

# Add custom rule
graph.add_custom_validation_rule('max_fan_out', validate_max_fan_out)

# Run custom validations
custom_results = graph.validate_custom_rules()
```

### Graph Statistics and Health Scoring

Comprehensive graph analysis with health scoring:

```python
stats = graph.get_graph_statistics()

print(f"Basic Stats: {stats['basic_stats']}")
print(f"Validation Summary: {stats['validation_summary']}")
print(f"Performance Metrics: {stats['performance_metrics']}")
print(f"Overall Health Score: {stats['health_score']:.3f}")
```

**Health Score Components:**
- **Error Penalty**: Heavily penalizes validation errors
- **Warning Penalty**: Moderately penalizes warnings
- **Performance Penalty**: Penalizes high complexity and connectivity issues
- **Range**: 0.0 (very unhealthy) to 1.0 (perfect health)

### Validation Demo

Run the comprehensive validation demonstration:

```bash
python scripts/graph_validation_demo.py
```

This demo showcases:
- Cycle detection across all link types
- Link consistency validation with rule violations
- Unit relationship validation with structural issues
- Activation bounds checking with out-of-range values
- Graph integrity validation with orphaned units
- Performance metrics analysis with complexity indicators
- Custom validation rules with domain-specific constraints

## GraphML Export

ReCoN graphs can be exported to GraphML format for analysis in external graph visualization and analysis tools.

### Export Methods

```python
from recon_core.compiler import compile_from_file

# Compile a ReCoN graph
graph = compile_from_file('scripts/house.yaml')

# Export to GraphML format
graph.export_graphml('house_network.graphml')

# Or get NetworkX DiGraph for programmatic analysis
nx_graph = graph.to_networkx()
```

### Exported Attributes

**Node Attributes:**
- `kind`: Unit type ("SCRIPT" or "TERMINAL")
- `state`: Current state ("INACTIVE", "REQUESTED", "ACTIVE", etc.)
- `activation`: Current activation level (0.0-1.0)
- `threshold`: Confirmation threshold
- `meta_*`: Any custom metadata attributes (prefixed)

**Edge Attributes:**
- `type`: Link type ("SUB", "SUR", "POR", "RET")
- `weight`: Connection strength

### Compatible Tools

GraphML files can be imported into:
- **Gephi**: Advanced graph visualization and analysis
- **yEd**: Graph editor with layout algorithms
- **NetworkX**: For further programmatic analysis
- **Other tools**: Any GraphML-compatible software

### Example Usage

```python
import networkx as nx

# Export and re-import with NetworkX
graph.export_graphml('network.graphml')
imported = nx.read_graphml('network.graphml')

# Analyze with NetworkX
print(f"Nodes: {imported.number_of_nodes()}")
print(f"Edges: {imported.number_of_edges()}")

# Access ReCoN-specific attributes
for node, attrs in imported.nodes(data=True):
    print(f"Unit {node}: {attrs['kind']} in state {attrs['state']}")
```

## House Recognition Example

### Network Topology

The demo implements a house recognition network with this hierarchy:

```
u_root (SCRIPT)
├── u_roof (SCRIPT) → t_horz (TERMINAL)
├── u_body (SCRIPT) → t_mean (TERMINAL)
└── u_door (SCRIPT) → t_vert (TERMINAL) + t_mean (TERMINAL)
```

### Link Configuration

```python
# Evidence flow (SUB links)
g.add_edge(Edge('t_horz', 'u_roof', LinkType.SUB, w=1.0))
g.add_edge(Edge('t_mean', 'u_body', LinkType.SUB, w=1.0))
g.add_edge(Edge('t_vert', 'u_door', LinkType.SUB, w=1.0))
g.add_edge(Edge('t_mean', 'u_door', LinkType.SUB, w=0.6))  # OR relationship

# Request flow (SUR links)
g.add_edge(Edge('u_root', 'u_roof', LinkType.SUR, w=1.0))
g.add_edge(Edge('u_root', 'u_body', LinkType.SUR, w=1.0))
g.add_edge(Edge('u_root', 'u_door', LinkType.SUR, w=1.0))

# Temporal sequencing (POR links)
g.add_edge(Edge('u_roof', 'u_body', LinkType.POR, w=1.0))
g.add_edge(Edge('u_body', 'u_door', LinkType.POR, w=1.0))
```

### Recognition Flow

1. **Initialization**: Root script activated, sends REQUEST to all children
2. **Parallel Recognition**: Roof, body, and door scripts request their terminals
3. **Terminal Detection**: Terminals detect features and send CONFIRM when above threshold
4. **Sequential Confirmation**: Roof confirms first, enables body, which enables door
5. **Final Confirmation**: Root confirms when sufficient children are confirmed

## Usage Guide

### Basic Usage

```python
from recon_core.enums import UnitType, LinkType, State
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine

# Create network
g = Graph()

# Add units
g.add_unit(Unit('root', UnitType.SCRIPT))
g.add_unit(Unit('terminal', UnitType.TERMINAL, thresh=0.5))

# Add connections
g.add_edge(Edge('terminal', 'root', LinkType.SUB))
g.add_edge(Edge('root', 'terminal', LinkType.SUR))

# Create engine and run
engine = Engine(g)

# Activate root to start recognition
g.units['root'].a = 1.0
g.units['root'].state = State.ACTIVE

# Run simulation
snapshot = engine.step(10)
print(f"Root state: {snapshot['units']['root']['state']}")
```

### Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive visualization
streamlit run viz/app_streamlit.py
```

The Streamlit interface provides:
- **Scene Generation**: Create synthetic house scenes
- **Step-by-Step Execution**: Watch network dynamics unfold
- **Real-time Visualization**: See state changes and message flow
- **Network Graph**: Interactive view of the ReCoN topology

### Perception Pipeline

The implementation includes a comprehensive synthetic perception pipeline with diverse scene generation and rich feature extraction:

#### Synthetic Scene Generation

```python
from perception.dataset import *

# Generate diverse scene types
house = make_house_scene(size=64, noise=0.05)
barn = make_barn_scene(size=64, scale_factor=1.2) 
occluded = make_occluded_scene(size=64, occlusion_type='tree')

# Random variations for training diversity
varied_house = make_varied_scene('house', size=64, 
                                scale_range=(0.7, 1.3), 
                                position_variance=0.2)
```

**Scene Types:**
- **Houses**: Traditional house with triangular roof, rectangular body, door
- **Barns**: Wider structures with arched roofs and large door openings
- **Occluded scenes**: Houses partially hidden by trees, clouds, or boxes
- **Varied scenes**: Random scaling, positioning, and noise for diversity

#### Comprehensive Terminal Features

```python
from perception.terminals import comprehensive_terminals_from_image

# Extract 21 terminal features (12 advanced + 4 autoencoder + 5 engineered)

# Basic features (3 terminals)
print(f"Mean intensity: {features['t_mean']:.3f}")
print(f"Vertical edges: {features['t_vert']:.3f}")  
print(f"Horizontal edges: {features['t_horz']:.3f}")

# SIFT-like features (3 terminals)  
print(f"Corner strength: {features['t_corners']:.3f}")
print(f"Edge magnitude: {features['t_edges']:.3f}")
print(f"Orientation variance: {features['t_orient_var']:.3f}")

# Blob detection (3 terminals)
print(f"Blob density: {features['t_blobs']:.3f}")
print(f"Local texture: {features['t_texture']:.3f}")
print(f"Global contrast: {features['t_contrast']:.3f}")

# Geometric features (3 terminals)
print(f"Shape count: {features['t_n_shapes']:.3f}")
print(f"Compactness: {features['t_compact']:.3f}") 
print(f"Aspect ratio: {features['t_aspect']:.3f}")

# Autoencoder features (4 terminals)
print(f"Latent features: {[features[f't_ae_{i}'] for i in range(4)]}")

# Extra engineered features (5 terminals)
print(f"Vertical symmetry: {features['t_vsym']:.3f}")
print(f"Line anisotropy: {features['t_line_aniso']:.3f}")
print(f"Triangularity: {features['t_triangle']:.3f}")
print(f"Rectangularity: {features['t_rect']:.3f}")
print(f"Door brightness: {features['t_door_bright']:.3f}")
```

**Feature Categories:**
1. **Basic Filters**: Mean intensity, edge detection using simple convolution
2. **SIFT-like**: Harris corner detection, gradient analysis, orientation statistics  
3. **Blob Detection**: Difference of Gaussians, texture analysis, contrast measurement
4. **Geometric Analysis**: Connected components, shape properties, spatial relationships
5. **Autoencoder**: Learned patch representations via denoising autoencoder

#### Autoencoder Terminal Details

The system includes a lightweight denoising autoencoder for learned feature extraction:

```python
from perception.terminals import SimpleAutoencoder, get_autoencoder

# Get the global autoencoder (training gated by env var RECON_TRAIN_AE)
ae = get_autoencoder()

# Extract autoencoder features directly
ae_features = ae.encode_patches(your_image, n_patches=8)

# Or use through the terminal interface
ae_terminals = autoencoder_terminals_from_image(your_image)
```

**Autoencoder Architecture:**
- **Input**: 8×8 image patches (64 dimensions)
- **Hidden Layer**: 8 neurons with ReLU activation
- **Latent**: 4 dimensions with sigmoid activation  
- **Decoder**: Symmetric reconstruction path
- **Training**: Denoising reconstruction on diverse synthetic scenes (enable with env `RECON_TRAIN_AE=1` or pass `retrain=True`)
- **Features**: Compressed patch representations averaged across image

## Advanced Features

### Learning (Optional)

The `learn.py` module provides online utilities for adapting link weights:

```python
from recon_core.learn import online_sur_update, online_generic_update

# Reinforce SUR links to helpful children when parent confirms
online_sur_update(graph, parent_id='u_house', lr=0.05)

# Generic per-edge update across SUB/SUR/POR/RET with simple heuristics
online_generic_update(graph, src_id='u_roof', dst_id='u_body', lr=0.05)
```

### Custom Terminal Units

The system provides multiple levels of terminal feature extraction that you can extend:

```python
# Use different feature extraction levels
from perception.terminals import (
    terminals_from_image,           # Basic 3 features
    advanced_terminals_from_image,  # Advanced 12 features  
    comprehensive_terminals_from_image  # All 21 features (12 + 4 + 5)
)

# Create custom terminal extractors
def custom_terminals_from_image(img):
    # Start with existing comprehensive features
    features = comprehensive_terminals_from_image(img)
    
    # Add your custom features
    features.update({
        't_custom_feature': your_feature_detector(img),
        't_domain_specific': domain_specific_analysis(img)
    })
    return features

# Or extend the autoencoder approach
from perception.terminals import SimpleAutoencoder

# Train specialized autoencoder for your domain
custom_ae = SimpleAutoencoder(patch_size=16, latent_dim=8)
custom_ae.train(your_training_images, n_epochs=100)

# Use in ReCoN network
def specialized_ae_terminals(img):
    latent_features = custom_ae.encode_patches(img)
    return {f't_spec_ae_{i}': feat for i, feat in enumerate(latent_features)}
```

**Extension Points:**
- **Basic Level**: Add simple filter-based features
- **Advanced Level**: Implement sophisticated computer vision algorithms  
- **Autoencoder Level**: Train domain-specific learned representations
- **Hybrid Approach**: Combine multiple feature types for maximum discrimination

## Testing

The implementation includes comprehensive unit tests across engine, graph, learning, and perception components:

```bash
# Run all tests (if pytest is available)
python -m pytest tests/ -v

# Lightweight runner (no pytest needed)
python run_tests.py
```

Selected pytest modules:
- `tests/test_engine.py`, `tests/test_state_machine.py`: Core engine behavior
- `tests/test_compact_arithmetic.py`: Gate arithmetic and propagation
- `tests/test_graph.py`, `tests/test_message_system.py`, `tests/test_learn.py`: Infrastructure and learning
- `tests/test_terminals.py`, `tests/test_synthetic_scenes.py`: Perception pipeline

## Performance Characteristics

### Complexity
- **Time Complexity**: O(E) per propagation step, where E is number of edges
- **Space Complexity**: O(N + E) for graph storage, where N is number of units
- **Message Complexity**: O(M) for message delivery, where M is number of messages

### Scalability
- Efficient for networks with hundreds to thousands of units
- Asynchronous message passing enables parallel processing
- Soft activation updates provide numerical stability

## Extensions and Applications

### Potential Enhancements

1. **Advanced Perception**: Replace simple filters with CNN-based feature extractors
2. **Learning**: Implement full gradient-based weight learning
3. **Multi-Modal**: Extend to handle multiple sensory modalities
4. **Real-Time**: Optimize for real-time processing constraints

### Alternative Applications

1. **Robotics**: Sequential task execution with sensory feedback
2. **Natural Language**: Syntactic/semantic parsing with confirmation
3. **Planning**: Hierarchical plan execution with failure recovery
4. **Diagnosis**: Fault diagnosis with selective testing

## References

- **Original Paper**: "Request Confirmation Networks for Active Object Recognition" (CoCoNIPS 2015)
- **Key Concepts**: Active perception, spreading activation, hierarchical recognition
- **Related Work**: Neural-symbolic integration, active vision systems

## Contributing

The codebase is designed for clarity and extensibility:
- Clean separation between core algorithm and application-specific code
- Comprehensive test coverage ensures reliability
- Modular architecture enables easy customization

For questions or contributions, refer to the implementation comments and test cases for detailed behavior specifications.
