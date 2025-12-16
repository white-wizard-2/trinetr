import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import AllChannelsModal from './AllChannelsModal'
import LinearLayerVisualizer from './LinearLayerVisualizer'
import './ActivationVisualizer.css'

// Colormap functions
function applyColormap(value: number, colormap: string): [number, number, number] {
  // value is 0-255
  const t = value / 255
  
  switch (colormap) {
    case 'jet': {
      // Blue -> Cyan -> Green -> Yellow -> Red
      if (t < 0.25) {
        const r = 0
        const g = 4 * t
        const b = 1
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
      } else if (t < 0.5) {
        const r = 0
        const g = 1
        const b = 1 - 4 * (t - 0.25)
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
      } else if (t < 0.75) {
        const r = 4 * (t - 0.5)
        const g = 1
        const b = 0
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
      } else {
        const r = 1
        const g = 1 - 4 * (t - 0.75)
        const b = 0
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
      }
    }
    case 'viridis': {
      // Purple -> Blue -> Green -> Yellow
      const r = Math.round(255 * (0.267 + 0.005 * Math.sin(t * Math.PI * 4)))
      const g = Math.round(255 * (0.005 + 0.99 * t))
      const b = Math.round(255 * (0.329 + 0.671 * (1 - t)))
      return [r, g, b]
    }
    case 'hot': {
      // Black -> Red -> Yellow -> White
      if (t < 0.33) {
        const r = Math.round(255 * (t * 3))
        return [r, 0, 0]
      } else if (t < 0.66) {
        const r = 255
        const g = Math.round(255 * ((t - 0.33) * 3))
        return [r, g, 0]
      } else {
        const r = 255
        const g = 255
        const b = Math.round(255 * ((t - 0.66) * 3))
        return [r, g, b]
      }
    }
    case 'cool': {
      // Cyan -> Magenta
      const r = Math.round(255 * t)
      const g = Math.round(255 * (1 - t))
      const b = 255
      return [r, g, b]
    }
    default: // grayscale
      return [value, value, value]
  }
}

interface ChannelCanvasProps {
  data: number[][]
  colormap?: string
}

function ChannelCanvas({ data, colormap = 'grayscale' }: ChannelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (canvasRef.current && data) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (ctx && Array.isArray(data)) {
        const height = data.length
        const width = Array.isArray(data[0]) ? data[0].length : 0
        
        if (width > 0 && height > 0) {
          const displayScale = 3
          canvas.width = width * displayScale
          canvas.height = height * displayScale
          
          // Clear canvas
          ctx.fillStyle = '#000'
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          
          // Find min/max for normalization
          let minVal = Infinity
          let maxVal = -Infinity
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const val = data[y][x]
              if (val < minVal) minVal = val
              if (val > maxVal) maxVal = val
            }
          }
          
          const range = maxVal - minVal || 1
          const imageData = ctx.createImageData(canvas.width, canvas.height)
          
          for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
              const srcX = Math.floor(x / displayScale)
              const srcY = Math.floor(y / displayScale)
              
              if (srcY < height && srcX < width) {
                const idx = (y * canvas.width + x) * 4
                const rawValue = data[srcY][srcX]
                // Normalize to 0-255 range
                const normalized = Math.round(((rawValue - minVal) / range) * 255)
                const value = Math.max(0, Math.min(255, normalized))
                
                const [r, g, b] = applyColormap(value, colormap)
                imageData.data[idx] = r
                imageData.data[idx + 1] = g
                imageData.data[idx + 2] = b
                imageData.data[idx + 3] = 255
              }
            }
          }
          ctx.putImageData(imageData, 0, 0)
        }
      }
    }
  }, [data, colormap])

  return <canvas ref={canvasRef} className="channel-canvas" />
}

interface ActivationVisualizerProps {
  modelId: string
  layerName: string
  imageFile: File
  onActivationsLoaded?: (activations: any) => void
}

function ActivationVisualizer({
  modelId,
  layerName,
  imageFile,
  onActivationsLoaded
}: ActivationVisualizerProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activations, setActivations] = useState<any>(null)
  const [showAllChannels, setShowAllChannels] = useState(false)
  const [colormap, setColormap] = useState<string>('grayscale')
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const fetchActivations = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const formData = new FormData()
        formData.append('file', imageFile)
        
        const response = await axios.post(
          `http://localhost:8000/models/${modelId}/visualize/activations`,
          formData,
          {
            params: { layer_name: layerName },
            headers: { 'Content-Type': 'multipart/form-data' }
          }
        )
        
        setActivations(response.data)
        onActivationsLoaded?.(response.data)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load activations')
      } finally {
        setLoading(false)
      }
    }

    if (modelId && layerName && imageFile) {
      fetchActivations()
    }
  }, [modelId, layerName, imageFile, onActivationsLoaded])

  // Separate effect for drawing canvas when activations change (skip for linear layers)
  useEffect(() => {
    if (activations?.is_linear) return // Skip canvas drawing for linear layers
    
    if (canvasRef.current && activations && activations.activations) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (ctx && Array.isArray(activations.activations)) {
        const data = activations.activations
        const height = data.length
        const width = Array.isArray(data[0]) ? data[0].length : 0
        
        if (width > 0 && height > 0) {
          // Use actual dimensions, scale up for visibility
          const displayScale = 4
          canvas.width = width * displayScale
          canvas.height = height * displayScale
          
          // Clear canvas
          ctx.fillStyle = '#000'
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          
          const imageData = ctx.createImageData(canvas.width, canvas.height)
          
          // Find min/max for normalization
          let minVal = Infinity
          let maxVal = -Infinity
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const val = data[y][x]
              if (val < minVal) minVal = val
              if (val > maxVal) maxVal = val
            }
          }
          
          const range = maxVal - minVal || 1
          
          for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
              const srcX = Math.floor(x / displayScale)
              const srcY = Math.floor(y / displayScale)
              
              if (srcY < height && srcX < width) {
                const idx = (y * canvas.width + x) * 4
                const rawValue = data[srcY][srcX]
                // Normalize to 0-255 range
                const normalized = Math.round(((rawValue - minVal) / range) * 255)
                const value = Math.max(0, Math.min(255, normalized))
                
                const [r, g, b] = applyColormap(value, colormap)
                imageData.data[idx] = r
                imageData.data[idx + 1] = g
                imageData.data[idx + 2] = b
                imageData.data[idx + 3] = 255
              }
            }
          }
          ctx.putImageData(imageData, 0, 0)
        }
      }
    }
  }, [activations, colormap])

  if (loading) {
    return (
      <div className="activation-visualizer">
        <div className="loading">Loading activations...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="activation-visualizer">
        <div className="error">{error}</div>
      </div>
    )
  }

  return (
    <div className="activation-visualizer">
      <h2>Layer: {layerName}</h2>
      {activations && (
        <>
          <div className="activation-stats">
            {activations.previous_layer && (
              <div className="stat">
                <span className="stat-label">Previous Layer:</span>
                <span className="stat-value">{activations.previous_layer}</span>
              </div>
            )}
            {activations.input_shape && (
              <div className="stat">
                <span className="stat-label">Input Shape:</span>
                <span className="stat-value">{activations.input_shape}</span>
              </div>
            )}
            <div className="stat">
              <span className="stat-label">Output Shape:</span>
              <span className="stat-value">{activations.shape?.join(' × ')}</span>
            </div>
            {activations.num_channels && (
              <div className="stat">
                <span className="stat-label">Channels:</span>
                <span className="stat-value">{activations.num_channels}</span>
              </div>
            )}
            <div className="stat">
              <span className="stat-label">Min:</span>
              <span className="stat-value">{activations.min?.toFixed(4)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Max:</span>
              <span className="stat-value">{activations.max?.toFixed(4)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Mean:</span>
              <span className="stat-value">{activations.mean?.toFixed(4)}</span>
            </div>
          </div>
          
          <div className="visualization">
            {activations.is_linear ? (
              <LinearLayerVisualizer
                activations={activations.activations}
                topActivations={activations.top_activations}
                layerName={layerName}
              />
            ) : (
              <>
                <div className="viz-section">
                  <div className="section-header">
                    <div>
                      <h3>Mean Activation Map</h3>
                      <p className="viz-description">Average across all channels - shows overall feature response</p>
                    </div>
                    <div className="colormap-selector">
                      <label>Colormap:</label>
                      <select 
                        value={colormap} 
                        onChange={(e) => setColormap(e.target.value)}
                        className="colormap-select"
                      >
                        <option value="grayscale">Grayscale</option>
                        <option value="jet">Jet</option>
                        <option value="viridis">Viridis</option>
                        <option value="hot">Hot</option>
                        <option value="cool">Cool</option>
                      </select>
                    </div>
                  </div>
                  <canvas ref={canvasRef} className="activation-canvas" />
                </div>
              </>
            )}
            
           
            
            {!activations.is_linear && activations.channels && activations.channels.length > 0 && (
              <div className="viz-section">
                <div className="section-header">
                  <div>
                    <h3>Top Channels</h3>
                    <p className="viz-description">Individual feature maps with highest activation</p>
                  </div>
                  {activations.all_channels && activations.all_channels.length > 0 && (
                    <button 
                      className="show-all-button"
                      onClick={() => setShowAllChannels(true)}
                    >
                      Show All ({activations.num_channels})
                    </button>
                  )}
                </div>
                <div className="channel-grid">
                  {activations.channels.map((channel: any, idx: number) => (
                    <div key={idx} className="channel-item">
                      <div className="channel-header">
                        Channel {channel.channel_index} (mean: {channel.mean_activation.toFixed(4)})
                      </div>
                      <ChannelCanvas data={channel.data} colormap={colormap} />
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {showAllChannels && activations.all_channels && (
              <AllChannelsModal
                channels={activations.all_channels}
                layerName={layerName}
                numChannels={activations.num_channels}
                colormap={colormap}
                modelId={modelId}
                onClose={() => setShowAllChannels(false)}
              />
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default ActivationVisualizer

