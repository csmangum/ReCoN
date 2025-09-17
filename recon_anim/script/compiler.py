from __future__ import annotations

from typing import Iterable, Iterator, List

from recon_anim.models.events import Event, SceneStep, StepStart, StepEnd


def compile_events_to_steps(events: Iterable[Event], default_step_duration: float = 0.5) -> List[SceneStep]:
    steps: List[SceneStep] = []
    current_events: List[Event] = []
    current_idx = None
    current_start_t = None
    last_t = None

    for ev in events:
        if isinstance(ev, StepStart):
            if current_events:
                # close previous (shouldn't happen if events are well-formed)
                duration = (
                    (last_t - current_start_t) if (last_t is not None and current_start_t is not None) else default_step_duration
                )
                steps.append(SceneStep(idx=current_idx if current_idx is not None else len(steps), duration=float(max(0.0, duration)), events=current_events))
                current_events = []
            current_idx = ev.step_index
            current_start_t = ev.t
            last_t = ev.t
            current_events.append(ev)
        elif isinstance(ev, StepEnd):
            current_events.append(ev)
            # compute duration
            end_t = ev.t
            duration = (
                (end_t - current_start_t) if (end_t is not None and current_start_t is not None) else default_step_duration
            )
            steps.append(SceneStep(idx=current_idx if current_idx is not None else len(steps), duration=float(max(0.0, duration)), events=current_events))
            current_events = []
            current_idx = None
            current_start_t = None
            last_t = end_t
        else:
            current_events.append(ev)
            # best-effort track of last timestamp
            t = getattr(ev, "t", None)
            if t is not None:
                last_t = t

    if current_events:
        # flush tail
        duration = default_step_duration
        steps.append(SceneStep(idx=current_idx if current_idx is not None else len(steps), duration=float(max(0.0, duration)), events=current_events))

    return steps

