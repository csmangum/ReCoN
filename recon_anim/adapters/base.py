from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from recon_anim.models.events import Event


class ReconEventSource(ABC):
    @abstractmethod
    def stream_events(self) -> Iterator[Event]:
        ...


class ReconStepper(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def step(self, n: int = 1) -> None:
        ...

    @abstractmethod
    def stream_events(self) -> Iterator[Event]:
        ...

