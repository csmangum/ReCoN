# ReCoN Real-time Audio Recognition System

This system provides real-time audio recognition using the Request Confirmation Network (ReCoN) for the "Engage Active Perception" hypothesis. It includes multiple interfaces and can work with or without audio hardware.

## 🎯 Features

- **Real-time Audio Processing**: Continuous microphone input with feature extraction
- **ReCoN Network Visualization**: Interactive network state visualization
- **Multiple Interfaces**: CLI, GUI, and Web-based interfaces
- **Simulated Audio**: Works without audio hardware using simulated data
- **Hierarchical Recognition**: Word-level to syllable-level hypothesis activation
- **Temporal Sequencing**: Ordered execution of recognition activities

## 📁 Files

### Core Components
- `audio_capture.py` - Real-time audio capture using pygame and sounddevice
- `realtime_recon_gui.py` - Tkinter GUI for real-time visualization
- `realtime_recon_demo.py` - Simplified GUI version with simulated audio
- `realtime_recon_cli.py` - Command-line interface
- `realtime_recon_streamlit.py` - Web-based interface using Streamlit
- `run_realtime_recon.py` - Launcher script for all modes

### Test and Documentation
- `test_realtime_system.py` - System test script
- `REALTIME_RECON_README.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install pygame librosa scipy sounddevice matplotlib networkx streamlit plotly pandas

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-tk portaudio19-dev
```

### 2. Run the System

#### Command Line Interface (Recommended for headless environments)
```bash
# Run simulation with 20 steps, 1 second delay
python3 run_realtime_recon.py cli --steps 20 --delay 1.0

# Run interactive mode
python3 run_realtime_recon.py interactive

# Run with custom parameters
python3 run_realtime_recon.py cli --steps 10 --delay 0.5
```

#### Web Interface (Streamlit)
```bash
# Start web server
python3 run_realtime_recon.py streamlit

# Or directly
streamlit run realtime_recon_streamlit.py --server.port 8501
```

#### GUI Interface (Requires display)
```bash
# Run GUI version
python3 run_realtime_recon.py gui

# Or directly
python3 realtime_recon_demo.py
```

#### Test Audio Capture
```bash
# Test audio functionality
python3 run_realtime_recon.py audio
```

## 🎵 How It Works

### 1. Audio Processing
The system continuously captures audio from the microphone and extracts features:
- **MFCC Low**: Low-frequency spectral features
- **Pitch High**: High-frequency pitch detection
- **Rhythm**: Tempo and beat detection
- **Noise Level**: Signal quality assessment
- **Formant**: Vocal tract characteristics
- **Spectrogram**: Time-frequency representation

### 2. ReCoN Network
The "Engage Active Perception" hypothesis is represented as a hierarchical network:

```
u_phrase (root hypothesis)
├── u_engage (word level)
│   ├── u_en_phoneme
│   └── u_gei_phoneme
├── u_active (word level)
│   ├── u_dj_phoneme
│   └── u_ak_phoneme
└── u_perception (word level)
    ├── u_ti_phoneme
    ├── u_v_phoneme
    ├── u_per_phoneme
    ├── u_sep_phoneme
    └── u_shun_phoneme
```

### 3. Link Types
- **SUB (Subordinate)**: Evidence propagation (bottom-up)
- **SUR (Superior)**: Request propagation (top-down)
- **POR (Precedence)**: Temporal sequencing
- **RET (Return)**: Temporal feedback

### 4. Unit States
- **INACTIVE**: Not processing
- **REQUESTED**: Waiting for input
- **ACTIVE**: Currently processing
- **TRUE**: Terminal confirmed by input
- **CONFIRMED**: Script confirmed by evidence

## 🎛️ Interface Modes

### CLI Mode
- **Simulation**: Run automated simulation with configurable steps and delay
- **Interactive**: Manual step-by-step control
- **Real-time**: Continuous processing (if audio hardware available)

### Web Mode (Streamlit)
- **Network Visualization**: Interactive Plotly network graph
- **Audio Features**: Real-time bar chart of extracted features
- **Activation Timeline**: Line chart showing unit activations over time
- **Controls**: Start/stop simulation, reset network, single step

### GUI Mode (Tkinter)
- **Multi-tab Interface**: Network, timeline, and audio features
- **Real-time Updates**: Live visualization of network state
- **Interactive Controls**: Sliders for sensitivity and thresholds

## 🔧 Configuration

### Audio Settings
- **Sample Rate**: 22050 Hz (configurable)
- **Chunk Size**: 1024 samples (configurable)
- **Channels**: Mono (1 channel)

### ReCoN Engine Settings
- **SUR Positive**: 0.4 (superior link strength)
- **POR Positive**: 0.6 (precedence link strength)
- **RET Positive**: 0.2 (return link strength)
- **Confirmation Ratio**: 0.75 (threshold for confirmation)

### Simulation Settings
- **Speed**: 0.1x to 5.0x (web/GUI)
- **Intensity**: 0.0 to 1.0 (feature scaling)
- **Steps**: 1 to 1000 (CLI)

## 🐛 Troubleshooting

### Audio Issues
```bash
# Test audio capture
python3 audio_capture.py

# Check audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Display Issues
```bash
# For GUI mode, ensure display is available
export DISPLAY=:0

# Or use web mode instead
python3 run_realtime_recon.py streamlit
```

### Dependencies
```bash
# Reinstall audio libraries
pip install --force-reinstall sounddevice pygame

# Install system audio libraries
sudo apt-get install portaudio19-dev libasound2-dev
```

## 📊 Example Output

### CLI Simulation
```
Step 1 - Network State
Audio Features:
  mfcc_low       : 0.102
  pitch_high     : 0.515
  rhythm         : 0.334
  noise_level    : 0.255
  formant        : 0.429
  spectrogram    : 0.198

Unit States:
  Script Units:
    u_phrase             | ACTIVE       | a=1.000  | thresh=0.800
    u_engage             | ACTIVE       | a=1.000  | thresh=0.700
    u_active             | REQUESTED    | a=1.000  | thresh=0.650
    u_perception         | REQUESTED    | a=1.000  | thresh=0.750
  Terminal Units:
    t_mfcc_low           | REQUESTED    | a=0.300  | thresh=0.400
    t_pitch_high         | TRUE         | a=0.515  | thresh=0.500
    t_rhythm             | TRUE         | a=0.526  | thresh=0.300

Activation Summary:
  Active units: 14
  Confirmed units: 9
```

## 🎯 Use Cases

1. **Speech Recognition Research**: Study hierarchical recognition patterns
2. **Audio Processing Education**: Learn about feature extraction and network dynamics
3. **ReCoN Algorithm Development**: Test and visualize network behavior
4. **Real-time Audio Analysis**: Monitor audio features and network responses
5. **Interactive Demonstrations**: Show ReCoN concepts in action

## 🔬 Technical Details

### Audio Feature Extraction
- Uses librosa for professional audio analysis
- Extracts MFCC, spectral, and temporal features
- Normalizes features to [0, 1] range
- Implements speech activity detection

### Network Visualization
- NetworkX for graph representation
- Matplotlib/Plotly for rendering
- State-based node coloring
- Link type differentiation

### Real-time Processing
- Multi-threaded audio capture
- Asynchronous GUI updates
- Configurable processing rates
- Error handling and recovery

## 📝 License

This project is part of the ReCoN (Request Confirmation Network) system. See the main project documentation for licensing information.

## 🤝 Contributing

To contribute to the real-time audio recognition system:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different audio inputs
5. Submit a pull request

## 📞 Support

For issues with the real-time system:
1. Check the troubleshooting section
2. Run the test script: `python3 test_realtime_system.py`
3. Check system audio configuration
4. Verify all dependencies are installed

---

**Note**: This system is designed to work with or without audio hardware. When no microphone is available, it uses simulated audio data to demonstrate the ReCoN network behavior.