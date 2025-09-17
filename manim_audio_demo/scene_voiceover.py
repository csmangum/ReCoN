"""
Alternative approach using manim-voiceover for auto-timed narration.

This example uses the offline PyTTSX3 service (Linux typically needs espeak or
espeak-ng and ffmpeg installed). You can swap services, e.g. GTTS, if desired.

Render:
  manim -pqh scene_voiceover.py NarratedVoiceover
"""

from __future__ import annotations

from manim import BLUE, MathTex, Text, Circle, UP, DOWN, Write, Create
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.pyttsx3 import PyTTSX3Service


class NarratedVoiceover(VoiceoverScene):
    def construct(self) -> None:
        self.set_speech_service(PyTTSX3Service(rate=180))

        title = Text("Area of a Circle").to_edge(UP)
        with self.voiceover(text="Welcome. We'll explore the area of a circle.") as tracker:
            self.play(Write(title), run_time=tracker.duration)

        circle = Circle(radius=1.7, color=BLUE)
        with self.voiceover(text="Here's a circle. The distance from the center to the edge is the radius, r.") as tracker:
            self.play(Create(circle), run_time=tracker.duration)

        formula = MathTex("A = \\pi r^2").to_edge(DOWN)
        with self.voiceover(text="The area equals pi times r squared.") as tracker:
            self.play(Write(formula), run_time=tracker.duration)

        self.wait(0.5)

