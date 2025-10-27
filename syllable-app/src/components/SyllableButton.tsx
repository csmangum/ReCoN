import React from 'react';
import { Play, Volume2 } from 'lucide-react';

interface SyllableButtonProps {
  syllable: {
    id: string;
    display_name: string;
  };
  state: string;
  activation: number;
  onClick: () => void;
  isActive?: boolean;
}

const getStateColor = (state: string, activation: number) => {
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

const getStateIcon = (state: string) => {
  switch (state) {
    case 'CONFIRMED':
      return <Volume2 className="w-4 h-4" />;
    case 'ACTIVE':
      return <Play className="w-4 h-4" />;
    default:
      return null;
  }
};

export const SyllableButton: React.FC<SyllableButtonProps> = ({
  syllable,
  state,
  activation,
  onClick,
  isActive = false,
}) => {
  const colorClass = getStateColor(state, activation);
  const icon = getStateIcon(state);

  return (
    <button
      onClick={onClick}
      className={`
        relative flex flex-col items-center justify-center
        w-20 h-20 rounded-lg border-2 transition-all duration-200
        ${colorClass}
        ${isActive ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
        hover:scale-105 active:scale-95
        shadow-md hover:shadow-lg
      `}
    >
      {icon && (
        <div className="absolute top-1 right-1">
          {icon}
        </div>
      )}
      
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
};