from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class NodeKind(str, Enum):
    terminal = "terminal"
    script = "script"


class NodeState(str, Enum):
    IDLE = "IDLE"
    REQUESTED = "REQUESTED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    INHIBITED = "INHIBITED"


class LinkType(str, Enum):
    POR = "POR"
    RET = "RET"
    SUB = "SUB"
    SUR = "SUR"
    INH = "INH"


@dataclass
class ReconNode:
    id: str
    label: str
    kind: NodeKind
    score: Optional[float] = None


@dataclass
class ReconEdge:
    id: str
    src: str
    dst: str
    link: LinkType
    weight: Optional[float] = None


@dataclass
class ReconGraph:
    nodes: List[ReconNode]
    edges: List[ReconEdge]


@dataclass
class ReconState:
    step: int = 0
    nodeStates: Dict[str, NodeState] = field(default_factory=dict)
    edgeStates: Dict[str, str] = field(default_factory=dict)  # ACTIVE/INACTIVE
    explanations: List[str] = field(default_factory=list)


class InMemoryReconEngine:
    """
    Minimal, deterministic stub of a ReCoN-like engine.

    - Maintains a fixed graph and simple state transitions
    - step(): advances one tick and updates node/edge states
    - run(): background loop that steps until a pseudo fixpoint
    - pause(): halts the loop
    - reset(): clears state
    - subscribe(): returns an asyncio.Queue receiving state updates
    """

    def __init__(self) -> None:
        self._graph = self._build_default_graph()
        self._state = ReconState(
            step=0,
            nodeStates={n.id: NodeState.IDLE for n in self._graph.nodes},
            edgeStates={e.id: "INACTIVE" for e in self._graph.edges},
            explanations=[],
        )
        self._lock = asyncio.Lock()
        self._running = False
        self._runner_task: Optional[asyncio.Task] = None
        self._subscribers: Set[asyncio.Queue] = set()

    # --- graph -------------------------------------------------------------
    @staticmethod
    def _build_default_graph() -> ReconGraph:
        nodes = [
            ReconNode(id="n1", label="User report", kind=NodeKind.terminal),
            ReconNode(id="n2", label="Parse log", kind=NodeKind.script),
            ReconNode(id="n3", label="Detect root cause", kind=NodeKind.script),
            ReconNode(id="n4", label="Remediate", kind=NodeKind.script),
            ReconNode(id="n5", label="Resolved", kind=NodeKind.terminal),
        ]
        edges = [
            ReconEdge(id="e1", src="n1", dst="n2", link=LinkType.POR),
            ReconEdge(id="e2", src="n2", dst="n3", link=LinkType.RET),
            ReconEdge(id="e3", src="n3", dst="n4", link=LinkType.POR),
            ReconEdge(id="e4", src="n4", dst="n5", link=LinkType.SUR),
            ReconEdge(id="e5", src="n3", dst="n2", link=LinkType.INH),
        ]
        return ReconGraph(nodes=nodes, edges=edges)

    def graph(self) -> ReconGraph:
        return self._graph

    def state(self) -> ReconState:
        return self._state

    # --- pubsub ------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _broadcast(self) -> None:
        # Non-blocking fan-out; drop if queues are full
        for q in list(self._subscribers):
            try:
                q.put_nowait(self._state)
            except asyncio.QueueFull:
                # best-effort; ignore
                pass

    # --- lifecycle ---------------------------------------------------------
    async def reset(self) -> None:
        async with self._lock:
            self._state = ReconState(
                step=0,
                nodeStates={n.id: NodeState.IDLE for n in self._graph.nodes},
                edgeStates={e.id: "INACTIVE" for e in self._graph.edges},
                explanations=["reset"],
            )
        await self._broadcast()

    async def pause(self) -> None:
        self._running = False
        if self._runner_task and not self._runner_task.done():
            # Let the loop observe _running = False
            await asyncio.sleep(0)

    async def run(self, interval_ms: int = 400) -> None:
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            try:
                while self._running:
                    done = await self.step()
                    if done:
                        self._running = False
                        break
                    await asyncio.sleep(interval_ms / 1000)
            finally:
                self._running = False

        self._runner_task = asyncio.create_task(_loop())

    async def step(self) -> bool:
        """Advance a simple deterministic state machine.

        Returns True if a fixpoint ("done") is reached.
        """
        async with self._lock:
            s = self._state
            s.step += 1
            s.explanations = []

            # Trivial pipeline progression n1 -> n2 -> n3 -> n4 -> n5
            def set_state(node_id: str, new: NodeState, why: str) -> None:
                prev = s.nodeStates.get(node_id, NodeState.IDLE)
                if prev != new:
                    s.nodeStates[node_id] = new
                    s.explanations.append(f"{node_id} -> {new} ({why})")

            if s.step == 1:
                set_state("n1", NodeState.CONFIRMED, "initial trigger")
                s.edgeStates["e1"] = "ACTIVE"
            elif s.step == 2:
                set_state("n2", NodeState.REQUESTED, "POR from n1")
            elif s.step == 3:
                set_state("n2", NodeState.CONFIRMED, "executed")
                s.edgeStates["e2"] = "ACTIVE"
            elif s.step == 4:
                set_state("n3", NodeState.REQUESTED, "RET from n2")
            elif s.step == 5:
                set_state("n3", NodeState.CONFIRMED, "analysis complete")
                s.edgeStates["e3"] = "ACTIVE"
            elif s.step == 6:
                set_state("n4", NodeState.CONFIRMED, "remediation started")
                s.edgeStates["e4"] = "ACTIVE"
            elif s.step == 7:
                set_state("n5", NodeState.CONFIRMED, "resolved")
            else:
                # fixpoint: no further changes
                pass

        await self._broadcast()

        # done after we confirm n5
        return self._state.nodeStates.get("n5") == NodeState.CONFIRMED

    # --- helpers -----------------------------------------------------------
    def is_done(self) -> bool:
        return self._state.nodeStates.get("n5") == NodeState.CONFIRMED

