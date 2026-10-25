import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,  // 2 min for training
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

// ── Endpoints ──────────────────────────────────────────────────────────────

export const getHealth = () => api.get('/health')
export const getDashboard = () => api.get('/dashboard')
export const getEvaluation = () => api.get('/evaluation')
export const getInsights = () => api.get('/insights')
export const getDatasetStatus = () => api.get('/dataset/status')
export const resetDataset = () => api.post('/dataset/reset')

export const uploadDataset = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/upload', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export const trainModels = (payload) => api.post('/train', payload, { timeout: 600000 })

export const predictSingle = (features, model_name) =>
  api.post('/predict', { features, model_name })

export const predictBatch = (file, model_name) => {
  const fd = new FormData()
  fd.append('file', file)
  if (model_name) fd.append('model_name', model_name)
  return api.post('/predict/batch', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export default api
