# Syllable Template Learning App

A tkinter application that helps you understand how syllable/phoneme templates work in the ReCoN (Request Confirmation Network) system. This interactive tool allows you to explore how different syllables activate their template features, helping you get better intuition on how to make terminal scripts more accurate.

## Features

- **Interactive Syllable Buttons**: Click on syllable buttons to see how they activate template features
- **Real-time Template Visualization**: See which audio features are activated for each syllable
- **Network State Display**: View the current state of all network units
- **Learning Tips**: Built-in guidance on understanding templates and making them more accurate
- **Step-by-step Simulation**: Run full phrase recognition simulations

## Installation

The app requires Python 3 and several dependencies. Install them with:

```bash
# Install Python packages
pip3 install numpy pyyaml

# Install tkinter (GUI library)
sudo apt-get install python3-tk
```

## Usage

### Quick Start

```bash
cd /workspace
python3 run_syllable_app.py
```

### Manual Launch

```bash
cd /workspace
python3 syllable_learning_app.py
```

## How to Use the App

### 1. Understanding the Interface

The app has several main areas:

- **Left Panel**: Syllable buttons - click these to activate specific syllables
- **Right Panel**: Template visualization with three tabs:
  - **Template Features**: Shows which audio features are activated
  - **Network State**: Shows the current state of all network units
  - **Learning Tips**: Built-in guidance and explanations

### 2. Exploring Syllables

1. **Click a syllable button** (e.g., "en", "gei", "dj") to activate it
2. **Watch the template features** update in real-time
3. **Notice the color coding**:
   - Green = CONFIRMED (fully activated)
   - Yellow = ACTIVE (partially activated)
   - Orange = REQUESTED (waiting for evidence)
   - Gray = INACTIVE (not activated)

### 3. Understanding Template Features

Each syllable template is made up of audio features:

- **t_mfcc_low**: Low frequency features (good for vowels)
- **t_pitch_high**: High frequency features (good for consonants)
- **t_formant**: Vowel formant structure
- **t_rhythm**: Timing and rhythm patterns
- **t_spectrogram**: Overall frequency content
- **t_noise_level**: Background noise (inhibits recognition when high)

### 4. Learning Tips

The "Learning Tips" tab provides guidance on:

- How templates work
- What each audio feature represents
- How to make terminals more accurate
- Understanding activation patterns
- Experimentation strategies

## Making Terminal Scripts More Accurate

### 1. Understand Feature Patterns

- **Vowels** (like /a/, /e/, /o/) need strong MFCC and formant features
- **Consonants** (like /s/, /t/, /k/) need strong pitch and spectrogram features
- **Similar syllables** will have similar feature patterns

### 2. Adjust Thresholds

- Lower thresholds for easier activation
- Higher thresholds for more selective activation
- Consider noise levels in real environments

### 3. Test Different Combinations

- Try clicking different syllables to see their unique patterns
- Notice how similar syllables have similar feature patterns
- Use this to understand why some syllables are confused

### 4. Experiment with Weights

- Adjust terminal weights based on what you observe
- Consider the hierarchical structure (Phrase → Words → Phonemes → Terminals)
- Test with different syllable combinations

## Troubleshooting

### Common Issues

1. **"No module named 'tkinter'"**
   - Install tkinter: `sudo apt-get install python3-tk`

2. **"No module named 'numpy'"**
   - Install numpy: `pip3 install numpy`

3. **App won't start**
   - Check that all dependencies are installed
   - Make sure you're in the `/workspace` directory
   - Check the test script: `python3 test_syllable_app.py`

### Getting Help

- Run the test script to verify everything is working: `python3 test_syllable_app.py`
- Check the console output for error messages
- Make sure the network file exists: `/workspace/scripts/engage_active_perception.yaml`

## Technical Details

The app uses the ReCoN (Request Confirmation Network) system with:

- **9 syllables/phonemes** from the "engage active perception" phrase
- **6 terminal features** for audio processing
- **Hierarchical structure**: Phrase → Words → Phonemes → Terminals
- **Real-time simulation** of network activation

## Files

- `syllable_learning_app.py`: Main application
- `run_syllable_app.py`: Launcher script
- `test_syllable_app.py`: Test script
- `SYLLABLE_APP_README.md`: This documentation

## Example Workflow

1. **Start the app**: `python3 run_syllable_app.py`
2. **Click "en" syllable**: See how it activates MFCC and formant features
3. **Click "gei" syllable**: Compare the feature pattern with "en"
4. **Click "Run Full Simulation"**: See the complete phrase recognition process
5. **Check "Learning Tips"**: Read about making terminals more accurate
6. **Experiment**: Try different syllables and observe their patterns

This interactive approach helps you develop intuition about how syllable templates work and how to improve their accuracy in real applications.