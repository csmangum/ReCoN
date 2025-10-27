import React from 'react';
import { Play, Square, RotateCcw, Zap } from 'lucide-react';

interface ControlPanelProps {
  currentStep: number;
  isRunning: boolean;
  onReset: () => void;
  onStep: () => void;
  onRun: () => void;
  onStop: () => void;
  onRunFullSimulation: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  currentStep,
  isRunning,
  onReset,
  onStep,
  onRun,
  onStop,
  onRunFullSimulation,
}) => {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-600">Step:</span>
            <span className="text-lg font-bold text-gray-800">{currentStep}</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={onReset}
              className="flex items-center space-x-1 px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
            
            <button
              onClick={onStep}
              disabled={isRunning}
              className="flex items-center space-x-1 px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4" />
              <span>Step</span>
            </button>
            
            {isRunning ? (
              <button
                onClick={onStop}
                className="flex items-center space-x-1 px-3 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors"
              >
                <Square className="w-4 h-4" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                onClick={onRun}
                className="flex items-center space-x-1 px-3 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Run</span>
              </button>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={onRunFullSimulation}
            disabled={isRunning}
            className="flex items-center space-x-1 px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
  );
};