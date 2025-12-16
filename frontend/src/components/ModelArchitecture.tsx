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
    const newExpanded = new Set<string>(expandedGroups)
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

  const getRootDescription = () => {
    return "Root layers are layers without a module prefix (e.g., 'fc' instead of 'layer1.conv1'). " +
           "⚠️ IMPORTANT: Data flows from TOP to BOTTOM in this view. The FC layer (1000 outputs) is the FINAL layer, " +
           "not the first! Early layers (like layer1, layer2) come BEFORE the FC layer in the data flow. " +
           "The FC layer takes the final feature vector and outputs 1000 class probabilities."
  }

  const getRootOverview = (rootLayers: Layer[]) => {
    // Sort root layers by their position in the layers array (which is already sorted by execution order)
    const sortedRoot = [...rootLayers].sort((a, b) => {
      const idxA = layers.findIndex(l => l.name === a.name)
      const idxB = layers.findIndex(l => l.name === b.name)
      return idxA - idxB
    })

    let overview = "📊 Data Flow Through Root Layers (in execution order):\n\n"
    
    sortedRoot.forEach((layer, idx) => {
      const isInitial = idx < sortedRoot.length - 2 // Last 2 are typically final layers
      const prefix = isInitial ? "🔹" : "🔸"
      const stage = isInitial ? "INITIAL" : "FINAL"
      
      overview += `${prefix} ${stage}: ${layer.name} (${layer.type})\n`
      if (layer.description) {
        overview += `   ${layer.description}\n`
      }
      
      if (idx < sortedRoot.length - 1) {
        overview += `   ↓\n`
      }
    })
    
    overview += "\n💡 The root section contains both the FIRST layers (initial feature extraction) and the LAST layers (final classification) of the network."
    
    return overview
  }

  const groupLayers = () => {
    const grouped: { [key: string]: Layer[] } = {}
    
    // Separate root layers into initial and final
    const initialRootLayers: Layer[] = []
    const finalRootLayers: Layer[] = []
    
    layers.forEach(layer => {
      const group = getLayerGroup(layer.name)
      if (group === 'root') {
        const name = layer.name.toLowerCase()
        // Check if this is VGG (has features or classifier in other layers)
        const isVGG = layers.some(l => l.name.includes('features') || l.name.includes('classifier'))
        // For VGG, root avgpool comes between features and classifier, not at the end
        const isFinal = (!isVGG && (name.includes('avgpool') || name.includes('adaptive'))) || name.includes('fc') || layer.type === 'Linear'
        if (isFinal) {
          finalRootLayers.push(layer)
        } else {
          initialRootLayers.push(layer)
        }
      } else {
        if (!grouped[group]) {
          grouped[group] = []
        }
        grouped[group].push(layer)
      }
    })
    
    // Add root layers twice if needed (initial and final)
    if (initialRootLayers.length > 0) {
      grouped['root_initial'] = initialRootLayers
    }
    if (finalRootLayers.length > 0) {
      grouped['root_final'] = finalRootLayers
    }
    
    // Sort each group by the original layer order
    Object.keys(grouped).forEach(group => {
      grouped[group].sort((a, b) => {
        const idxA = layers.findIndex(l => l.name === a.name)
        const idxB = layers.findIndex(l => l.name === b.name)
        return idxA - idxB
      })
    })
    
    // Sort groups by execution order
    const sortedGroupEntries = Object.entries(grouped).sort(([nameA], [nameB]) => {
      // root_initial should come first
      if (nameA === 'root_initial') return -1
      if (nameB === 'root_initial') return 1
      
      // root_final should come last
      if (nameA === 'root_final') return 1
      if (nameB === 'root_final') return -1
      
      // For other groups (layer1, layer2, etc.), sort by finding the first layer in each group
      const firstLayerA = grouped[nameA]?.[0]
      const firstLayerB = grouped[nameB]?.[0]
      
      if (firstLayerA && firstLayerB) {
        const idxA = layers.findIndex(l => l.name === firstLayerA.name)
        const idxB = layers.findIndex(l => l.name === firstLayerB.name)
        return idxA - idxB
      }
      
      return nameA.localeCompare(nameB)
    })
    
    // Convert back to object with ordered entries
    const orderedGrouped: { [key: string]: Layer[] } = {}
    sortedGroupEntries.forEach(([name, layers]) => {
      orderedGrouped[name] = layers
    })
    
    return orderedGrouped
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
        <div className="flow-info flow-start">
          <p>📥 Input Image (224×224×3)</p>
          <p className="flow-direction">↓ Data flows DOWN ↓</p>
        </div>
        
        {Object.entries(grouped).map(([groupName, groupLayers]) => {
          const isExpanded = expandedGroups.has(groupName)
          const isRootGroup = groupName.startsWith('root')
          const displayName = groupName === 'root_initial' ? 'root (initial)' : 
                             groupName === 'root_final' ? 'root (final)' : 
                             groupName
          return (
            <div key={groupName} className="layer-group">
              <div 
                className="group-header"
                onClick={() => toggleGroup(groupName)}
              >
                <span className="group-toggle">{isExpanded ? '▼' : '▶'}</span>
                <span className="group-name">{displayName}</span>
                <span className="group-count">({groupLayers.length} layers)</span>
                {isRootGroup && (
                  <span className="group-info-icon" title={getRootDescription()}>ℹ️</span>
                )}
              </div>
              {isRootGroup && isExpanded && (
                <div className="root-explanation">
                  <p>{getRootDescription()}</p>
                  <div className="root-overview">
                    <h4>📋 {displayName.toUpperCase()} Overview:</h4>
                    <pre>{getRootOverview(groupLayers)}</pre>
                  </div>
                </div>
              )}
              
              {isExpanded && (
                <div className="group-layers">
                  {groupLayers.map((layer, idx) => {
                    const isSelected = selectedLayer === layer.name
                    const isFinalLayer = layer.type === 'Linear' || layer.name.toLowerCase().includes('fc')
                    return (
                      <div
                        key={layer.name}
                        className={`layer-node ${isSelected ? 'selected' : ''} ${isFinalLayer ? 'final-layer' : ''}`}
                        onClick={() => onLayerSelect(layer.name)}
                        style={{ borderLeftColor: getLayerColor(layer.type) }}
                      >
                        {isFinalLayer && (
                          <div className="final-layer-badge">FINAL LAYER</div>
                        )}
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
        
        <div className="flow-info flow-end">
          <p>📤 Output (1000 classes)</p>
          <p className="flow-note">Final FC layer produces class probabilities</p>
        </div>
      </div>
    </div>
  )
}

export default ModelArchitecture

