# Manim Audio Narration Demo

This project shows two ways to add narration to a Manim animation:

- Pre-recorded audio using `add_sound` (recommended for human voiceovers)
- Auto-timed narration using `manim-voiceover` (offline TTS via `pyttsx3`)

## Quick Start

1. Ensure `ffmpeg` is installed and on your PATH.
2. (Optional) Create and activate a virtualenv.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Put your recorded narration at `audio/narration.mp3`.
5. Render one of the scenes:

```bash
# Pre-recorded narration
manim -pqh scene_add_sound.py NarratedAddSound

# Auto voiceover via manim-voiceover + pyttsx3
manim -pqh scene_voiceover.py NarratedVoiceover
```

Use `-pql` for quicker previews.

## Narrator Script (example)

You can record this as `audio/narration.mp3` to match the `add_sound` scene.

1. "Welcome! In this short animation, we'll explore the area of a circle."
2. "This is a circle. The distance from the center to the edge is the radius, r."
3. "If we square the radius and multiply by pi, we get the area."
4. "Compared to a square, the circle's area depends only on its radius."
5. "So remember: A equals pi r squared."

Tip: Aim for ~10–12 seconds total. Then tweak waits/run_times in `scene_add_sound.py` to align visuals to your pacing.

## Notes

- `add_sound` attaches the audio track to the scene timeline; keep your scene long enough to include the entire narration.
- `manim-voiceover` will synthesize audio and automatically return durations to sync animations.
- On Linux, `pyttsx3` typically requires `espeak` or `espeak-ng`. Install via your package manager if needed.

