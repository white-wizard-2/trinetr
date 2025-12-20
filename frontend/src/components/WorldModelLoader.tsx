import { useState } from 'react'
import axios from 'axios'
import './WorldModelLoader.css'

interface WorldModelLoaderProps {
  onModelLoaded: (modelId: string, modelName: string) => void
}

const WORLD_MODELS = [
  { id: 'world-model-v1', name: 'World Model v1', description: 'VAE encoder, RNN memory, MLP controller' },
]

function WorldModelLoader({ onModelLoaded }: WorldModelLoaderProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState('world-model-v1')
  const [showInfo, setShowInfo] = useState(false)
  const [success, setSuccess] = useState(false)

  const loadModel = async () => {
    setLoading(true)
    setError(null)
    setSuccess(false)
    
    try {
      console.log('Loading world model:', selectedModel)
      const response = await axios.post('http://localhost:8000/world-models/load', null, {
        params: { 
          model_name: selectedModel
        }
      })
      
      console.log('Model loaded successfully:', response.data)
      if (response.data && response.data.model_id) {
        setSuccess(true)
        onModelLoaded(response.data.model_id, selectedModel)
        // Clear success message after 2 seconds
        setTimeout(() => setSuccess(false), 2000)
      } else {
        throw new Error('Invalid response from server')
      }
    } catch (err: any) {
      console.error('Error loading world model:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to load world model. Make sure the backend server is running on port 8000.'
      setError(errorMessage)
      console.error('Error details:', errorMessage)
      // Keep error visible for 5 seconds
      setTimeout(() => setError(null), 5000)
    } finally {
      setLoading(false)
    }
  }

  const currentModelInfo = WORLD_MODELS.find(m => m.id === selectedModel)

  return (
    <div className="world-model-loader">
      <select 
        value={selectedModel} 
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={loading}
      >
        {WORLD_MODELS.map(model => (
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
      {success && <div className="success-tooltip">✓ Model loaded successfully!</div>}
      
      {showInfo && currentModelInfo && (
        <div className="model-info-tooltip">
          <strong>{currentModelInfo.name}</strong>
          <p>{currentModelInfo.description}</p>
        </div>
      )}
    </div>
  )
}

export default WorldModelLoader

