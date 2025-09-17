"""
Demonstrates injecting a pre-recorded narration track into a Manim animation.

How to use
---------
1) Put your narration audio at: audio/narration.mp3 (48 kHz WAV/MP3 recommended)
2) Optionally tweak the waits and run_times to better sync visuals to narration
3) Render (preview, high quality):
   manim -pqh scene_add_sound.py NarratedAddSound

Tip: You can quickly iterate by using low quality preview:
   manim -pql scene_add_sound.py NarratedAddSound
"""

from __future__ import annotations

import pathlib
from manim import (
    BLUE,
    GREEN,
    YELLOW,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    Circle,
    Line,
    MathTex,
    Scene,
    Square,
    Text,
    Write,
    Create,
    FadeIn,
)


class NarratedAddSound(Scene):
    """Basic example using add_sound to attach a pre-recorded narration.

    Place your audio file at audio/narration.mp3 and keep the total scene length
    equal or longer than the audio length. Adjust waits and run_time values to
    align visual beats with spoken phrases.
    """

    def construct(self) -> None:
        project_root = pathlib.Path(__file__).parent
        narration_file = project_root / "audio" / "narration.mp3"

        if not narration_file.exists():
            raise FileNotFoundError(
                f"Narration file not found at: {narration_file}. "
                "Place your recorded narration there and re-render."
            )

        title = Text("Area of a Circle").to_edge(UP)
        self.play(Write(title), run_time=1.0)

        circle = Circle(radius=1.7, color=BLUE).shift(2.0 * LEFT)
        square = Square(side_length=3.4, color=GREEN).shift(2.0 * RIGHT)
        self.play(Create(circle), Create(square), run_time=1.2)

        # Start the narration at scene time t=0
        self.add_sound(str(narration_file), time_offset=0)

        # Approximate syncing with narration beats. Adjust as needed.
        # 0.0s - Title & shapes appear
        self.wait(0.8)

        # 1.0s - Emphasize circle
        self.play(circle.animate.set_stroke(width=8), run_time=0.5)
        self.play(circle.animate.set_stroke(width=2), run_time=0.3)

        # 2.0s - Show radius r
        self.wait(0.7)
        radius_line = Line(circle.get_center(), circle.get_right(), color=YELLOW)
        radius_label = MathTex("r").next_to(radius_line, 0.5 * UP)
        self.play(Create(radius_line), FadeIn(radius_label), run_time=0.9)

        # 3.5s - Show formula A = pi r^2
        self.wait(0.8)
        formula = MathTex("A = \\pi r^2").to_edge(DOWN)
        self.play(Write(formula), run_time=1.0)

        # 5.5s - Emphasize square for comparison
        self.wait(1.0)
        self.play(square.animate.set_stroke(width=8), run_time=0.5)
        self.play(square.animate.set_stroke(width=2), run_time=0.3)

        # Pad to the end of narration
        self.wait(2.0)

