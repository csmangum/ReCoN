import React from 'react';
import { UnitState } from '../types';

interface NetworkVisualizationProps {
  units: UnitState[];
  selectedSyllable?: string;
}

const getUnitColor = (state: string) => {
  switch (state) {
    case 'CONFIRMED':
      return '#10b981'; // green-500
    case 'ACTIVE':
      return '#f59e0b'; // yellow-500
    case 'REQUESTED':
      return '#f97316'; // orange-500
    default:
      return '#6b7280'; // gray-500
  }
};

const getUnitSize = (activation: number, kind: string) => {
  const baseSize = kind === 'SCRIPT' ? 20 : 15;
  return baseSize + (activation * 15);
};

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  units,
  selectedSyllable,
}) => {
  const syllables = units.filter(unit => unit.id.includes('phoneme'));
  const words = units.filter(unit => 
    unit.kind === 'SCRIPT' && 
    !unit.id.includes('phoneme') && 
    unit.id !== 'u_phrase'
  );
  const phrase = units.find(unit => unit.id === 'u_phrase');
  const terminals = units.filter(unit => unit.kind === 'TERMINAL');

  return (
    <div className="w-full h-96 bg-gray-50 rounded-lg border-2 border-gray-200 relative overflow-hidden">
      {/* Phrase (top center) */}
      {phrase && (
        <div
          className="absolute flex items-center justify-center rounded-full text-white font-bold text-sm"
          style={{
            left: '50%',
            top: '20px',
            transform: 'translateX(-50%)',
            width: getUnitSize(phrase.activation, phrase.kind),
            height: getUnitSize(phrase.activation, phrase.kind),
            backgroundColor: getUnitColor(phrase.state),
          }}
        >
          PHRASE
        </div>
      )}

      {/* Words (middle row) */}
      <div className="absolute top-24 left-0 right-0 flex justify-center space-x-8">
        {words.map((word, index) => (
          <div
            key={word.id}
            className="flex items-center justify-center rounded-full text-white font-semibold text-xs"
            style={{
              width: getUnitSize(word.activation, word.kind),
              height: getUnitSize(word.activation, word.kind),
              backgroundColor: getUnitColor(word.state),
            }}
          >
            {word.id.replace('u_', '')}
          </div>
        ))}
      </div>

      {/* Syllables (bottom row) */}
      <div className="absolute bottom-8 left-0 right-0 flex justify-center space-x-4">
        {syllables.map((syllable, index) => (
          <div
            key={syllable.id}
            className={`
              flex items-center justify-center rounded-full text-white font-semibold text-xs
              ${selectedSyllable === syllable.id ? 'ring-2 ring-blue-500' : ''}
            `}
            style={{
              width: getUnitSize(syllable.activation, syllable.kind),
              height: getUnitSize(syllable.activation, syllable.kind),
              backgroundColor: getUnitColor(syllable.state),
            }}
          >
            {syllable.id.replace('u_', '').replace('_phoneme', '')}
          </div>
        ))}
      </div>

      {/* Terminals (right side) */}
      <div className="absolute right-4 top-1/2 transform -translate-y-1/2 space-y-2">
        {terminals.map((terminal, index) => (
          <div
            key={terminal.id}
            className="flex items-center space-x-2"
          >
            <div
              className="w-3 h-3 rounded-full"
              style={{
                backgroundColor: getUnitColor(terminal.state),
              }}
            />
            <span className="text-xs text-gray-600">
              {terminal.id.replace('t_', '')}
            </span>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 text-xs text-gray-500">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>Confirmed</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Active</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-full bg-orange-500"></div>
            <span>Requested</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-full bg-gray-500"></div>
            <span>Inactive</span>
          </div>
        </div>
      </div>
    </div>
  );
};