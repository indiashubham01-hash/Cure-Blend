import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 20000,
});

export const assessHealth = (payload) => api.post('/api/v1/assess', payload);
export const getHistory = (limit = 10, skip = 0) =>
  api.get('/api/v1/history', {
    params: { limit, skip },
  });
export const getHealthStatus = () => api.get('/api/v1/health');
