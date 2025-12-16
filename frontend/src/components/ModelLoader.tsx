import { useState } from 'react'
import axios from 'axios'
import ModelInfoModal from './ModelInfoModal'
import './ModelLoader.css'

interface ModelLoaderProps {
  onModelLoaded: (modelId: string) => void
}

function ModelLoader({ onModelLoaded }: ModelLoaderProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelName, setModelName] = useState('resnet18')
  const [showInfo, setShowInfo] = useState(false)

  const loadModel = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/models/load', null, {
        params: { model_name: modelName }
      })
      
      onModelLoaded(response.data.model_id)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load model')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="model-loader">
        <div className="model-loader-header">
          <h3>Load Model</h3>
          <button 
            className="info-button"
            onClick={() => setShowInfo(true)}
            title="Learn about this model"
          >
            📖 Info
          </button>
        </div>
        <select 
          value={modelName} 
          onChange={(e) => setModelName(e.target.value)}
          disabled={loading}
        >
          <option value="resnet18">ResNet-18</option>
          <option value="resnet50">ResNet-50</option>
          <option value="vgg16">VGG-16</option>
        </select>
        <button onClick={loadModel} disabled={loading}>
          {loading ? 'Loading...' : 'Load Model'}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      
      {showInfo && (
        <ModelInfoModal 
          modelName={modelName}
          onClose={() => setShowInfo(false)}
        />
      )}
    </>
  )
}

export default ModelLoader

