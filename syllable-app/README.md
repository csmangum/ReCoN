# Syllable Template Learning App

A modern, interactive React application for exploring syllable/phoneme templates in the ReCoN (Request Confirmation Network) system. This app provides a clean, minimalist interface for understanding how syllable templates work and how to make terminal scripts more accurate.

## 🚀 Features

- **Interactive Syllable Buttons** - Click to activate specific syllables and see their template patterns
- **Real-time Network Visualization** - Visual representation of network state and activations
- **Terminal Feature Analysis** - Detailed view of audio features and their activations
- **Step-by-step Simulation** - Control the simulation with step, run, and reset controls
- **Clean, Modern UI** - Minimalist design with smooth animations and transitions
- **Hot Reload Development** - Instant updates as you develop

## 🛠️ Tech Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Radix UI** for accessible components

### Backend
- **FastAPI** for the Python API
- **ReCoN Core** for network simulation
- **NumPy** for audio feature processing

## 📦 Installation

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- pip3

### Quick Start

1. **Clone and setup**:
   ```bash
   cd /workspace/syllable-app
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Install backend dependencies**:
   ```bash
   cd backend
   pip3 install fastapi uvicorn pydantic
   cd ..
   ```

4. **Start the application**:
   ```bash
   ./start.sh
   ```

   Or manually:
   ```bash
   # Terminal 1 - Backend
   cd backend && python3 app.py
   
   # Terminal 2 - Frontend  
   npm run dev
   ```

## 🎯 Usage

### Getting Started

1. **Open the app** at `http://localhost:5175` (or the port shown in terminal)
2. **Click syllable buttons** to see how they activate template features
3. **Use controls** to step through simulations or run full simulations
4. **Explore different tabs** to see network visualization and terminal features

### Understanding the Interface

#### Syllable Buttons
- **Gray** = INACTIVE (not activated)
- **Orange** = REQUESTED (waiting for evidence)  
- **Yellow** = ACTIVE (partially activated)
- **Green** = CONFIRMED (fully activated)

#### Terminal Features
- **MFCC Low** - Low frequency features (good for vowels)
- **Pitch High** - High frequency features (good for consonants)
- **Formant** - Vowel formant structure
- **Rhythm** - Timing and rhythm patterns
- **Spectrogram** - Overall frequency content
- **Noise Level** - Background noise (inhibitory)

### Controls

- **Reset** - Reset the network to initial state
- **Step** - Run one simulation step
- **Run** - Run multiple steps with visualization
- **Full Simulation** - Run complete phrase recognition

## 🔧 Development

### Project Structure
```
syllable-app/
├── src/
│   ├── components/          # React components
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # App entry point
│   └── index.css           # Global styles
├── backend/
│   ├── app.py              # FastAPI backend
│   └── requirements.txt    # Python dependencies
├── public/                 # Static assets
└── package.json           # Node.js dependencies
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run backend` - Start Python backend
- `npm run start` - Start both frontend and backend

### API Endpoints

The backend provides these endpoints:

- `GET /network/info` - Get network information
- `GET /network/state` - Get current network state
- `POST /network/reset` - Reset network
- `POST /syllable/activate` - Activate a syllable
- `POST /simulation/run` - Run simulation
- `GET /audio/features` - Get audio features

## 🎓 Learning Objectives

This app helps you understand:

1. **How syllable templates work** - See which audio features each syllable activates
2. **Template activation patterns** - Understand why some syllables are confused
3. **Audio feature relationships** - Learn which features are important for different sounds
4. **Network dynamics** - Watch how activations propagate through the network
5. **Making terminals more accurate** - Get practical tips for improving recognition

## 🐛 Troubleshooting

### Common Issues

1. **"Connection Error"** - Make sure the backend is running on port 8000
2. **Port conflicts** - The app will automatically find available ports
3. **Module not found** - Run `npm install` to install dependencies
4. **Python errors** - Make sure ReCoN modules are in the Python path

### Debug Mode

Check the browser console for detailed error messages and network requests.

## 🚀 Deployment

### Production Build

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Backend Deployment

The FastAPI backend can be deployed using:
- Docker
- Gunicorn + Uvicorn
- Cloud platforms (Heroku, Railway, etc.)

## 📚 Related Documentation

- [ReCoN Documentation](../RECON_DOCUMENTATION.md)
- [Audio Demo Report](../AUDIO_DEMO_REPORT.md)
- [Use Cases](../USE_CASE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the ReCoN (Request Confirmation Network) system.

---

**Happy Learning!** 🎉

This interactive app makes it much easier to understand how syllable templates work and how to improve their accuracy. The real-time updates and clean interface provide an excellent learning experience for developers working with speech recognition systems.