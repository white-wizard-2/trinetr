import { useState, useEffect } from 'react'
import axios from 'axios'
import './LayerSelector.css'

interface LayerSelectorProps {
  modelId: string
  onLayerSelect: (layerName: string) => void
}

interface Layer {
  name: string
  type: string
  parameters: number
}

function LayerSelector({ modelId, onLayerSelect }: LayerSelectorProps) {
  const [layers, setLayers] = useState<Layer[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchLayers = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const response = await axios.get(`http://localhost:8000/models/${modelId}/layers`)
        setLayers(response.data.layers)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load layers')
      } finally {
        setLoading(false)
      }
    }

    if (modelId) {
      fetchLayers()
    }
  }, [modelId])

  if (loading) {
    return (
      <div className="layer-selector">
        <h2>Layers</h2>
        <div className="loading">Loading layers...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="layer-selector">
        <h2>Layers</h2>
        <div className="error">{error}</div>
      </div>
    )
  }

  return (
    <div className="layer-selector">
      <h2>Select Layer</h2>
      <div className="layer-list">
        {layers.map((layer) => (
          <div
            key={layer.name}
            className="layer-item"
            onClick={() => onLayerSelect(layer.name)}
          >
            <div className="layer-name">{layer.name || '(root)'}</div>
            <div className="layer-type">{layer.type}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default LayerSelector

