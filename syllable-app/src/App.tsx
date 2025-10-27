import React, { useState, useEffect } from 'react';
import { Play, Square, RotateCcw, Zap } from 'lucide-react';

// Types
interface UnitState {
  id: string;
  state: string;
  activation: number;
  kind: string;
}

interface NetworkState {
  units: UnitState[];
  step: number;
  terminals: Record<string, number>;
}

interface Syllable {
  id: string;
  display_name: string;
}

// Mock data for now
const mockSyllables: Syllable[] = [
  { id: 'u_en_phoneme', display_name: 'en' },
  { id: 'u_gei_phoneme', display_name: 'gei' },
  { id: 'u_dj_phoneme', display_name: 'dj' },
  { id: 'u_ak_phoneme', display_name: 'ak' },
  { id: 'u_ti_phoneme', display_name: 'ti' },
  { id: 'u_v_phoneme', display_name: 'v' },
  { id: 'u_per_phoneme', display_name: 'per' },
  { id: 'u_sep_phoneme', display_name: 'sep' },
  { id: 'u_shun_phoneme', display_name: 'shun' },
];

const mockNetworkState: NetworkState = {
  units: mockSyllables.map(s => ({
    id: s.id,
    state: 'INACTIVE',
    activation: 0,
    kind: 'SCRIPT'
  })),
  step: 0,
  terminals: {
    't_mfcc_low': 0.6,
    't_pitch_high': 0.4,
    't_rhythm': 0.6,
    't_noise_level': 0.1,
    't_formant': 0.7,
    't_spectrogram': 0.6,
  }
};

function App() {
  const [networkState, setNetworkState] = useState<NetworkState>(mockNetworkState);
  const [selectedSyllable, setSelectedSyllable] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleSyllableClick = (syllableId: string) => {
    setSelectedSyllable(syllableId);
    
    // Simulate activation
    setNetworkState(prev => ({
      ...prev,
      units: prev.units.map(unit => 
        unit.id === syllableId 
          ? { ...unit, state: 'ACTIVE', activation: 1.0 }
          : unit
      )
    }));
  };

  const handleReset = () => {
    setSelectedSyllable(null);
    setNetworkState(mockNetworkState);
  };

  const handleStep = () => {
    setNetworkState(prev => ({
      ...prev,
      step: prev.step + 1
    }));
  };

  const handleRun = () => {
    setIsRunning(true);
    // Simulate running
    setTimeout(() => setIsRunning(false), 2000);
  };

  const getSyllableState = (syllableId: string) => {
    const unit = networkState.units.find(u => u.id === syllableId);
    return unit ? { state: unit.state, activation: unit.activation } : { state: 'INACTIVE', activation: 0 };
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'CONFIRMED':
        return 'bg-green-500 hover:bg-green-600 text-white';
      case 'ACTIVE':
        return 'bg-yellow-500 hover:bg-yellow-600 text-white';
      case 'REQUESTED':
        return 'bg-orange-500 hover:bg-orange-600 text-white';
      default:
        return 'bg-gray-200 hover:bg-gray-300 text-gray-700';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900">
              Syllable Template Learning
            </h1>
            <p className="mt-2 text-gray-600">
              Interactive exploration of ReCoN syllable templates and audio features
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Control Panel */}
        <div className="mb-8">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-600">Step:</span>
                  <span className="text-lg font-bold text-gray-800">{networkState.step}</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleReset}
                    className="flex items-center space-x-1 px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                  
                  <button
                    onClick={handleStep}
                    disabled={isRunning}
                    className="flex items-center space-x-1 px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play className="w-4 h-4" />
                    <span>Step</span>
                  </button>
                  
                  <button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="flex items-center space-x-1 px-3 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play className="w-4 h-4" />
                    <span>Run</span>
                  </button>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => {}}
                  className="flex items-center space-x-1 px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors"
                >
                  <Zap className="w-4 h-4" />
                  <span>Full Simulation</span>
                </button>
              </div>
            </div>
            
            {isRunning && (
              <div className="mt-3 flex items-center space-x-2 text-sm text-blue-600">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span>Simulation running...</span>
              </div>
            )}
          </div>
        </div>

        {/* Syllables Grid */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Click a syllable to see its template activation
          </h2>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-9 gap-4">
            {mockSyllables.map((syllable) => {
              const { state, activation } = getSyllableState(syllable.id);
              return (
                <button
                  key={syllable.id}
                  onClick={() => handleSyllableClick(syllable.id)}
                  className={`
                    relative flex flex-col items-center justify-center
                    w-20 h-20 rounded-lg border-2 transition-all duration-200
                    ${getStateColor(state)}
                    ${selectedSyllable === syllable.id ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
                    hover:scale-105 active:scale-95
                    shadow-md hover:shadow-lg
                  `}
                >
                  <div className="text-lg font-semibold">
                    {syllable.display_name}
                  </div>
                  
                  <div className="text-xs opacity-80">
                    {activation.toFixed(2)}
                  </div>
                  
                  {state !== 'INACTIVE' && (
                    <div className="absolute bottom-1 left-1 right-1 h-1 bg-white/30 rounded-full">
                      <div 
                        className="h-full bg-white/60 rounded-full transition-all duration-300"
                        style={{ width: `${activation * 100}%` }}
                      />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Terminal Features */}
        <div className="mt-8 bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Terminal Features
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(networkState.terminals).map(([terminalId, value]) => (
              <div key={terminalId} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    {terminalId.replace('t_', '')}
                  </span>
                  <span className="text-xs text-gray-500">
                    {value.toFixed(3)}
                  </span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="h-2 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-300"
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;