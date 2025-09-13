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
| `TRUE` | Terminal unit has detected its feature/pattern |
| `CONFIRMED` | Script unit has sufficient confirmation from children |
| `FAILED` | Unit has failed validation or encountered error |
| `SUPPRESSED` | Unit has been inhibited by conflicting information |

### Link Types

Units are connected by **directed, typed links** that determine information flow:

#### SUB (Subordinate) Links
- **Direction**: Child → Parent
- **Purpose**: Evidence propagation (bottom-up)
- **Example**: Terminal provides confirmation evidence to its parent script

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

### Three-Phase Update Cycle

Each simulation time step consists of three phases:

#### 1. Propagation Phase
```python
# Calculate activation deltas using gate functions
delta = self._propagate()
```

The **compact arithmetic propagation** computes how activation flows through each link type using **per-gate functions**:

- **SUB Gate**: `TRUE/CONFIRMED` = +1.0 (positive evidence), `FAILED` = -1.0 (negative evidence)
- **SUR Gate**: `REQUESTED/ACTIVE` = +0.3 (request signal), `FAILED` = -0.3 (inhibition)
- **POR Gate**: `CONFIRMED` = +0.5 (enable successor), `FAILED` = -0.5 (inhibit successor)
- **RET Gate**: `FAILED` = -0.5 (failure feedback), `CONFIRMED` = +0.2 (success feedback)

#### 2. State Update Phase
```python
# Process messages and update unit states
self._update_states(delta)
```

- **Message Processing**: Handle incoming messages (REQUEST, CONFIRM, etc.)
- **Soft Activation Update**: Apply propagation deltas with damping: `a = clip(a + 0.2*delta, 0, 1)`
- **State Transitions**: Update unit states based on current activation and message history

#### 3. Message Delivery Phase
```python
# Deliver messages from outboxes to inboxes
self._deliver_messages()
```

Messages are moved from sender outboxes to receiver inboxes, enabling asynchronous communication.

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
```

#### Engine Class
```python
class Engine:
    def __init__(self, g: Graph):
        self.g = g
        self.t = 0

    def step(self, n: int = 1) -> dict:
        """Advance simulation by n time steps"""

    def reset(self) -> None:
        """Reset network to initial state"""

    def snapshot(self) -> dict:
        """Create current network state snapshot"""
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

# Extract all 16 terminal features
features = comprehensive_terminals_from_image(scene)

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

# Get the global autoencoder (automatically trains on diverse scenes)
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
- **Training**: Denoising reconstruction on diverse synthetic scenes
- **Features**: Compressed patch representations averaged across image

## Advanced Features

### Learning (Optional)

The `learn.py` module provides utilities for learning link weights:

```python
from recon_core.learn import update_weights

# Update SUR link weights based on confirmation patterns
update_weights(engine, learning_rate=0.01)
```

### Custom Terminal Units

The system provides multiple levels of terminal feature extraction that you can extend:

```python
# Use different feature extraction levels
from perception.terminals import (
    terminals_from_image,           # Basic 3 features
    advanced_terminals_from_image,  # Advanced 12 features  
    comprehensive_terminals_from_image  # All 16 features
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

The implementation includes comprehensive test coverage with **44 total tests**:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_synthetic_scenes.py -v  # 18 scene generation tests
python -m pytest tests/test_terminals.py -v        # 26 terminal feature tests
python -m pytest tests/test_engine.py -v           # Core ReCoN engine tests
```

### Test Coverage

#### Synthetic Scene Generation Tests (18 tests)
- **Basic Drawing**: Canvas creation, rectangle/triangle primitives
- **House Scenes**: Basic generation, scaling, positioning, noise robustness
- **Barn Scenes**: Structure validation, house/barn discrimination
- **Occluded Scenes**: Tree/cloud/box occlusion types, vs clean comparison
- **Varied Scenes**: Randomization, scale ranges, unknown type handling

#### Terminal Feature Tests (26 tests) 
- **Basic Filters**: Edge detection, uniform images, blank canvas handling
- **Advanced Features**: SIFT-like, blob detection, geometric analysis
- **Autoencoder**: Training, encoding, save/load, patch extraction
- **Integration**: Comprehensive features, convenience functions
- **Discrimination**: House vs barn, noise robustness, scale invariance

#### Core ReCoN Tests (Original)
- **State Transitions**: All 8 state changes, edge cases
- **Message Passing**: Request/confirm/inhibit message flow
- **Inhibition**: Lateral and feedback inhibition mechanisms  
- **Temporal Sequencing**: POR/RET link behavior
- **Failure Propagation**: Error handling and recovery

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
