import { useState } from 'react'
import axios from 'axios'
import './StateSpaceLoader.css'

interface StateSpaceLoaderProps {
  onModelLoaded: (modelId: string, modelName: string) => void
}

const STATE_SPACE_MODELS = [
  { id: 'mamba-130m', name: 'Mamba 130M', description: 'Small efficient state space model' },
  { id: 'mamba-370m', name: 'Mamba 370M', description: 'Medium-sized state space model' },
  { id: 'mamba-790m', name: 'Mamba 790M', description: 'Large state space model' },
]

function StateSpaceLoader({ onModelLoaded }: StateSpaceLoaderProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState('mamba-130m')
  const [showInfo, setShowInfo] = useState(false)

  const loadModel = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/state-space/load', null, {
        params: { 
          model_name: selectedModel
        }
      })
      
      onModelLoaded(response.data.model_id, selectedModel)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load state space model')
    } finally {
      setLoading(false)
    }
  }

  const currentModelInfo = STATE_SPACE_MODELS.find(m => m.id === selectedModel)

  return (
    <div className="state-space-loader">
      <select 
        value={selectedModel} 
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={loading}
      >
        {STATE_SPACE_MODELS.map(model => (
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

export default StateSpaceLoader

