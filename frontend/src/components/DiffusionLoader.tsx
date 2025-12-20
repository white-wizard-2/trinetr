import { useState } from 'react'
import axios from 'axios'
import './DiffusionLoader.css'

interface DiffusionLoaderProps {
  onModelLoaded: (modelId: string, modelName: string) => void
}

const DIFFUSION_MODELS = [
  { id: 'stable-diffusion-v1-4', name: 'Stable Diffusion v1.4', description: 'Classic text-to-image model' },
  { id: 'stable-diffusion-v1-5', name: 'Stable Diffusion v1.5', description: 'Improved version with better quality' },
  { id: 'stable-diffusion-2-1', name: 'Stable Diffusion 2.1', description: 'Latest stable version' },
]

function DiffusionLoader({ onModelLoaded }: DiffusionLoaderProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState('stable-diffusion-v1-4')
  const [showInfo, setShowInfo] = useState(false)

  const loadModel = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/diffusion/load', null, {
        params: { 
          model_name: selectedModel
        }
      })
      
      onModelLoaded(response.data.model_id, selectedModel)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load diffusion model')
    } finally {
      setLoading(false)
    }
  }

  const currentModelInfo = DIFFUSION_MODELS.find(m => m.id === selectedModel)

  return (
    <div className="diffusion-loader">
      <select 
        value={selectedModel} 
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={loading}
      >
        {DIFFUSION_MODELS.map(model => (
          <option key={model.id} value={model.id}>{model.name}</option>
        ))}
      </select>
      
      <button onClick={loadModel} disabled={loading} className="load-btn">
        {loading ? '...' : 'Load'}
      </button>
      
      <button 
        className="info-btn"
        onClick={() => setShowInfo(!showInfo)}
        title="Model info"
      >
        ℹ️
      </button>
      
      {error && <div className="error-tooltip">{error}</div>}
      
      {showInfo && currentModelInfo && (
        <div className="model-info-tooltip">
          <strong>{currentModelInfo.name}</strong>
          <p>{currentModelInfo.description}</p>
        </div>
      )}
    </div>
  )
}

export default DiffusionLoader

