from __future__ import annotations

import json
from typing import Iterator

from recon_anim.adapters.base import ReconEventSource
from recon_anim.models.events import Event, GraphDeclared, StepStart, StepEnd, NodeActivation, EdgeFlow, NodeState, NodeLabel, EdgeWeight, RunMetadata


_TYPE_MAP = {
    "GraphDeclared": GraphDeclared,
    "StepStart": StepStart,
    "StepEnd": StepEnd,
    "NodeActivation": NodeActivation,
    "EdgeFlow": EdgeFlow,
    "NodeState": NodeState,
    "NodeLabel": NodeLabel,
    "EdgeWeight": EdgeWeight,
    "RunMetadata": RunMetadata,
}


class JsonlEventSource(ReconEventSource):
    def __init__(self, path: str):
        self.path = path

    def stream_events(self) -> Iterator[Event]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                typ = obj.pop("type", None)
                cls = _TYPE_MAP.get(typ)
                if cls is None:
                    continue
                yield cls(**obj)

