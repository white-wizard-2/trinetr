import { useState } from 'react'
import axios from 'axios'
import './TransformerLoader.css'

interface TransformerLoaderProps {
  onModelLoaded: (modelId: string, modelName: string) => void
  onTypeChange: (type: 'text' | 'image') => void
  transformerType: 'text' | 'image'
}

const TEXT_MODELS = [
  { id: 'bert-base', name: 'BERT Base', description: 'Bidirectional encoder, good for understanding' },
  { id: 'distilbert', name: 'DistilBERT', description: 'Smaller, faster BERT' },
  { id: 'gpt2', name: 'GPT-2', description: 'Autoregressive text generation' },
  { id: 'roberta', name: 'RoBERTa', description: 'Robustly optimized BERT' },
]

const IMAGE_MODELS = [
  { id: 'vit-base', name: 'ViT Base', description: 'Vision Transformer for image classification' },
  { id: 'deit-small', name: 'DeiT Small', description: 'Data-efficient Image Transformer' },
  { id: 'swin-tiny', name: 'Swin Tiny', description: 'Shifted Window Transformer' },
  { id: 'clip-vit', name: 'CLIP ViT', description: 'Contrastive Language-Image Pre-training' },
]

function TransformerLoader({ onModelLoaded, onTypeChange, transformerType }: TransformerLoaderProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState(transformerType === 'text' ? 'bert-base' : 'vit-base')
  const [showInfo, setShowInfo] = useState(false)

  const models = transformerType === 'text' ? TEXT_MODELS : IMAGE_MODELS

  const handleTypeChange = (type: 'text' | 'image') => {
    onTypeChange(type)
    setSelectedModel(type === 'text' ? 'bert-base' : 'vit-base')
    setError(null)
  }

  const loadModel = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/transformers/load', null, {
        params: { 
          model_name: selectedModel,
          model_type: transformerType
        }
      })
      
      onModelLoaded(response.data.model_id, selectedModel)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load transformer model')
    } finally {
      setLoading(false)
    }
  }

  const currentModelInfo = models.find(m => m.id === selectedModel)

  return (
    <div className="transformer-loader">
      <div className="transformer-type-toggle">
        <button 
          className={`toggle-btn ${transformerType === 'text' ? 'active' : ''}`}
          onClick={() => handleTypeChange('text')}
        >
          📝 Text
        </button>
        <button 
          className={`toggle-btn ${transformerType === 'image' ? 'active' : ''}`}
          onClick={() => handleTypeChange('image')}
        >
          🖼️ Image
        </button>
      </div>
      
      <select 
        value={selectedModel} 
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={loading}
      >
        {models.map(model => (
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

export default TransformerLoader

