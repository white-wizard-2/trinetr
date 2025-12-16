import { useState, useEffect } from 'react'
import axios from 'axios'
import './ModelArchitecture.css'

interface Layer {
  name: string
  type: string
  description?: string
  next?: string
}

interface ModelArchitectureProps {
  modelId: string
  selectedLayer: string | null
  onLayerSelect: (layerName: string) => void
}

function ModelArchitecture({ modelId, selectedLayer, onLayerSelect }: ModelArchitectureProps) {
  const [layers, setLayers] = useState<Layer[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  useEffect(() => {
    const fetchLayers = async () => {
      setLoading(true)
      try {
        const response = await axios.get(`http://localhost:8000/models/${modelId}/layers`)
        setLayers(response.data.layers)
        
        // Auto-expand first few groups
        const groups = new Set<string>()
        response.data.layers.slice(0, 5).forEach((layer: Layer) => {
          const group = layer.name.split('.')[0]
          if (group) groups.add(group)
        })
        setExpandedGroups(groups)
      } catch (err) {
        console.error('Failed to load layers', err)
      } finally {
        setLoading(false)
      }
    }

    if (modelId) {
      fetchLayers()
    }
  }, [modelId])

  const toggleGroup = (groupName: string) => {
    const newExpanded = new Set(expandedGroups)
    if (newExpanded.has(groupName)) {
      newExpanded.delete(groupName)
    } else {
      newExpanded.add(groupName)
    }
    setExpandedGroups(newExpanded)
  }

  const getLayerGroup = (layerName: string): string => {
    const parts = layerName.split('.')
    return parts.length > 1 ? parts[0] : 'root'
  }

  const groupLayers = () => {
    const grouped: { [key: string]: Layer[] } = {}
    layers.forEach(layer => {
      const group = getLayerGroup(layer.name)
      if (!grouped[group]) {
        grouped[group] = []
      }
      grouped[group].push(layer)
    })
    return grouped
  }

  const getLayerIcon = (type: string): string => {
    if (type.includes('Conv')) return '🔲'
    if (type.includes('BatchNorm')) return '📊'
    if (type.includes('ReLU')) return '⚡'
    if (type.includes('Pool')) return '⬇️'
    if (type.includes('Linear')) return '🔗'
    return '⚙️'
  }

  const getLayerColor = (type: string): string => {
    if (type.includes('Conv')) return '#667eea'
    if (type.includes('BatchNorm')) return '#764ba2'
    if (type.includes('ReLU')) return '#f093fb'
    if (type.includes('Pool')) return '#4facfe'
    if (type.includes('Linear')) return '#43e97b'
    return '#888'
  }

  if (loading) {
    return <div className="model-architecture loading">Loading architecture...</div>
  }

  const grouped = groupLayers()

  return (
    <div className="model-architecture">
      <h2>Model Architecture</h2>
      <div className="architecture-flow">
        <div className="flow-info">
          <p>📥 Input Image (224×224×3)</p>
        </div>
        
        {Object.entries(grouped).map(([groupName, groupLayers]) => {
          const isExpanded = expandedGroups.has(groupName)
          return (
            <div key={groupName} className="layer-group">
              <div 
                className="group-header"
                onClick={() => toggleGroup(groupName)}
              >
                <span className="group-toggle">{isExpanded ? '▼' : '▶'}</span>
                <span className="group-name">{groupName}</span>
                <span className="group-count">({groupLayers.length} layers)</span>
              </div>
              
              {isExpanded && (
                <div className="group-layers">
                  {groupLayers.map((layer, idx) => {
                    const isSelected = selectedLayer === layer.name
                    return (
                      <div
                        key={layer.name}
                        className={`layer-node ${isSelected ? 'selected' : ''}`}
                        onClick={() => onLayerSelect(layer.name)}
                        style={{ borderLeftColor: getLayerColor(layer.type) }}
                      >
                        <div className="layer-header">
                          <span className="layer-icon">{getLayerIcon(layer.type)}</span>
                          <span className="layer-name">{layer.name.split('.').slice(-1)[0] || layer.name}</span>
                          <span className="layer-type">{layer.type}</span>
                        </div>
                        {layer.description && (
                          <div className="layer-description">{layer.description}</div>
                        )}
                        {layer.next && idx < groupLayers.length - 1 && (
                          <div className="layer-arrow">↓</div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
              
              {isExpanded && groupLayers.length > 0 && (
                <div className="flow-connector">↓</div>
              )}
            </div>
          )
        })}
        
        <div className="flow-info">
          <p>📤 Output (1000 classes)</p>
        </div>
      </div>
    </div>
  )
}

export default ModelArchitecture

