import axios from 'axios';
import { NetworkState, NetworkInfo, Syllable, AudioFeatures } from './types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const syllableApi = {
  // Get network information
  getNetworkInfo: async (): Promise<NetworkInfo> => {
    const response = await api.get('/network/info');
    return response.data;
  },

  // Get current network state
  getNetworkState: async (): Promise<NetworkState> => {
    const response = await api.get('/network/state');
    return response.data;
  },

  // Reset network
  resetNetwork: async (): Promise<NetworkState> => {
    const response = await api.post('/network/reset');
    return response.data;
  },

  // Activate a syllable
  activateSyllable: async (syllableId: string, steps: number = 3): Promise<NetworkState> => {
    const response = await api.post('/syllable/activate', {
      syllable_id: syllableId,
      steps,
    });
    return response.data;
  },

  // Run simulation
  runSimulation: async (steps: number = 10, reset: boolean = true): Promise<NetworkState> => {
    const response = await api.post('/simulation/run', {
      steps,
      reset,
    });
    return response.data;
  },

  // Get audio features
  getAudioFeatures: async (): Promise<AudioFeatures> => {
    const response = await api.get('/audio/features');
    return response.data;
  },

  // Get syllables list
  getSyllables: async (): Promise<{ syllables: Syllable[] }> => {
    const response = await api.get('/syllables');
    return response.data;
  },
};