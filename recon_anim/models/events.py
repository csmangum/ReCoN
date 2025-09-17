from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Dict, List

# Edge identifiers can be string or a tuple of (src, dst)
EdgeId = Union[str, Tuple[str, str]]


@dataclass(frozen=True)
class GraphDeclared:
    graph: Dict[str, Any]
    seed: int = 0
    units: Optional[str] = None


@dataclass(frozen=True)
class StepStart:
    step_index: int
    t: Optional[float] = None


@dataclass(frozen=True)
class StepEnd:
    step_index: int
    t: Optional[float] = None


@dataclass(frozen=True)
class NodeActivation:
    node_id: str
    value: float
    t: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class EdgeFlow:
    edge_id: EdgeId
    value: float
    t: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class NodeState:
    node_id: str
    state: str
    t: Optional[float] = None


@dataclass(frozen=True)
class NodeLabel:
    node_id: str
    text: str


@dataclass(frozen=True)
class EdgeWeight:
    edge_id: EdgeId
    value: float


@dataclass(frozen=True)
class RunMetadata:
    key: str
    value: Any


Event = Union[
    GraphDeclared,
    StepStart,
    StepEnd,
    NodeActivation,
    EdgeFlow,
    NodeState,
    NodeLabel,
    EdgeWeight,
    RunMetadata,
]


@dataclass(frozen=True)
class SceneStep:
    idx: int
    duration: float
    events: List[Event]

