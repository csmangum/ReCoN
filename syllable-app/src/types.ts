export interface UnitState {
  id: string;
  state: string;
  activation: number;
  kind: string;
}

export interface NetworkState {
  units: UnitState[];
  step: number;
  terminals: Record<string, number>;
}

export interface Syllable {
  id: string;
  display_name: string;
}

export interface NetworkInfo {
  syllables: string[];
  terminals: string[];
  words: string[];
  total_units: number;
}

export interface AudioFeatures {
  features: Record<string, number>;
}