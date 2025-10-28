import React from 'react';
import { Progress } from '@radix-ui/react-progress';

interface TerminalFeaturesProps {
  terminals: Record<string, number>;
  audioFeatures: Record<string, number>;
}

const terminalDescriptions: Record<string, string> = {
  't_mfcc_low': 'Low frequency features (vowels)',
  't_pitch_high': 'High frequency features (consonants)',
  't_formant': 'Vowel formant structure',
  't_rhythm': 'Timing and rhythm patterns',
  't_spectrogram': 'Overall frequency content',
  't_noise_level': 'Background noise (inhibitory)',
};

export const TerminalFeatures: React.FC<TerminalFeaturesProps> = ({
  terminals,
  audioFeatures,
}) => {
  return (
    <div className="space-y-6">
      {/* Current Terminal Activations */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Terminal Activations
        </h3>
        <div className="space-y-3">
          {Object.entries(terminals).map(([terminalId, activation]) => (
            <div key={terminalId} className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-700">
                  {terminalId.replace('t_', '')}
                </span>
                <span className="text-sm text-gray-500">
                  {activation.toFixed(3)}
                </span>
              </div>
              
              <div className="mb-2">
                <Progress 
                  value={activation * 100} 
                  className="w-full h-2 bg-gray-200 rounded-full overflow-hidden"
                />
                <div 
                  className="h-2 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-300"
                  style={{ width: `${activation * 100}%` }}
                />
              </div>
              
              <p className="text-xs text-gray-500">
                {terminalDescriptions[terminalId] || 'Audio feature'}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Synthetic Audio Features */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Synthetic Audio Features
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(audioFeatures).map(([featureId, value]) => (
            <div key={featureId} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-600">
                  {featureId.replace('t_', '')}
                </span>
                <span className="text-xs text-gray-500">
                  {value.toFixed(3)}
                </span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div 
                  className="h-1.5 bg-gradient-to-r from-green-400 to-green-600 rounded-full transition-all duration-300"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Learning Tips */}
      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
        <h4 className="font-semibold text-blue-800 mb-2">💡 Learning Tips</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• <strong>Vowels</strong> need strong MFCC and formant features</li>
          <li>• <strong>Consonants</strong> need strong pitch and spectrogram features</li>
          <li>• <strong>Noise level</strong> inhibits recognition when high</li>
          <li>• <strong>Rhythm</strong> helps with word boundaries</li>
        </ul>
      </div>
    </div>
  );
};