#!/usr/bin/env python3
"""
Script to generate TTS audio file using Google Text-to-Speech.
"""

from gtts import gTTS
import os

def generate_tts_audio(text, filename):
    """
    Generate an MP3 file using Google Text-to-Speech.

    Args:
        text (str): The text to convert to speech
        filename (str): The output filename (should end with .mp3)
    """
    print(f"Generating TTS audio for: '{text}'")
    print(f"Saving to: {filename}")

    # Create gTTS object
    tts = gTTS(text=text, lang='en', slow=False)

    # Save the audio file
    tts.save(filename)

    print(f"Audio file saved successfully: {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")

if __name__ == "__main__":
    # Generate the engage_active_perception.mp3 file
    text = "engage active perception"
    filename = "engage_active_perception.mp3"

    generate_tts_audio(text, filename)
