import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import './AllChannelsModal.css'

interface Channel {
  channel_index: number
  mean_activation: number
  data: number[][]
}

interface AllChannelsModalProps {
  channels: Channel[]
  layerName: string
  numChannels: number
  colormap?: string
  modelId: string
  onClose: () => void
}

// Colormap function (same as in ActivationVisualizer)
function applyColormap(value: number, colormap: string): [number, number, number] {
  const t = value / 255
  
  switch (colormap) {
    case 'jet': {
      if (t < 0.25) {
        return [0, Math.round(4 * t * 255), 255]
      } else if (t < 0.5) {
        return [0, 255, Math.round((1 - 4 * (t - 0.25)) * 255)]
      } else if (t < 0.75) {
        return [Math.round(4 * (t - 0.5) * 255), 255, 0]
      } else {
        return [255, Math.round((1 - 4 * (t - 0.75)) * 255), 0]
      }
    }
    case 'viridis': {
      const r = Math.round(255 * (0.267 + 0.005 * Math.sin(t * Math.PI * 4)))
      const g = Math.round(255 * (0.005 + 0.99 * t))
      const b = Math.round(255 * (0.329 + 0.671 * (1 - t)))
      return [r, g, b]
    }
    case 'hot': {
      if (t < 0.33) {
        return [Math.round(255 * (t * 3)), 0, 0]
      } else if (t < 0.66) {
        return [255, Math.round(255 * ((t - 0.33) * 3)), 0]
      } else {
        return [255, 255, Math.round(255 * ((t - 0.66) * 3))]
      }
    }
    case 'cool': {
      return [Math.round(255 * t), Math.round(255 * (1 - t)), 255]
    }
    default:
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
          // Scale to fit modal grid
          const maxSize = 150
          const scale = Math.min(maxSize / width, maxSize / height, 1)
          canvas.width = width * scale
          canvas.height = height * scale
          
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
              const srcX = Math.floor(x / scale)
              const srcY = Math.floor(y / scale)
              
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

  return <canvas ref={canvasRef} className="modal-channel-canvas" />
}

interface WeightFilter {
  filter_index: number
  weights: number[][][]
  raw_weights: number[][][]
  raw_min: number
  raw_max: number
  raw_mean: number
  raw_std: number
}

interface WeightData {
  layer_name: string
  layer_type: string
  shape: number[]
  out_channels: number
  in_channels: number
  kernel_size: number[]
  filters: WeightFilter[]
  weight_stats: {
    min: number
    max: number
    mean: number
    std: number
  }
}

function AllChannelsModal({ channels, layerName, numChannels, colormap = 'grayscale', modelId, onClose }: AllChannelsModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)
  const [gridColumns, setGridColumns] = useState(6)
  const [showInfo, setShowInfo] = useState(false)
  const [showWeights, setShowWeights] = useState(false)
  const [weights, setWeights] = useState<WeightData | null>(null)
  const [loadingWeights, setLoadingWeights] = useState(false)
  const [selectedFilter, setSelectedFilter] = useState<number | null>(null)
  const [weightScale, setWeightScale] = useState(1.0)
  const [weightShift, setWeightShift] = useState(0.0)

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    const handleResize = () => {
      if (window.innerWidth > 1920) setGridColumns(8)
      else if (window.innerWidth > 1440) setGridColumns(6)
      else if (window.innerWidth > 1024) setGridColumns(5)
      else setGridColumns(4)
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    document.addEventListener('keydown', handleEscape)
    document.body.style.overflow = 'hidden'

    return () => {
      window.removeEventListener('resize', handleResize)
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [onClose])

  const loadWeights = async () => {
    setLoadingWeights(true)
    try {
      const response = await axios.get(
        `http://localhost:8000/models/${modelId}/weights/${layerName}`
      )
      setWeights(response.data)
      setShowWeights(true)
    } catch (err: any) {
      console.error('Failed to load weights:', err)
      alert(err.response?.data?.detail || 'Failed to load weights')
    } finally {
      setLoadingWeights(false)
    }
  }

  const updateWeights = async (filterIndex: number) => {
    if (!weights) return
    
    try {
      const requestBody = {
        filter_index: filterIndex,
        weight_updates: {
          scale: weightScale,
          shift: weightShift
        }
      }
      
      console.log('Updating weights:', requestBody)
      
      const response = await axios.post(
        `http://localhost:8000/models/${modelId}/weights/${layerName}/update`,
        requestBody,
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )
      
      console.log('Response:', response.data)
      
      // Safely extract response data
      const data = response.data || {}
      const message = typeof data.message === 'string' ? data.message : `Updated filter ${filterIndex}!`
      const newStats = data.new_stats || {}
      
      // Build stats string safely
      let statsStr = ''
      if (newStats && typeof newStats === 'object') {
        const min = typeof newStats.min === 'number' ? newStats.min.toFixed(4) : 'N/A'
        const max = typeof newStats.max === 'number' ? newStats.max.toFixed(4) : 'N/A'
        const mean = typeof newStats.mean === 'number' ? newStats.mean.toFixed(4) : 'N/A'
        statsStr = `\n\nNew Statistics:\n  Min: ${min}\n  Max: ${max}\n  Mean: ${mean}`
      }
      
      alert(`${message}${statsStr}\n\nReload the image to see the effect on activations.`)
      
      // Reload weights to see updated values
      loadWeights()
    } catch (err: any) {
      console.error('Failed to update weights:', err)
      console.error('Error response:', err.response)
      
      let errorMsg = 'Failed to update weights'
      
      if (err.response?.data) {
        if (typeof err.response.data === 'string') {
          errorMsg = err.response.data
        } else if (err.response.data.detail) {
          errorMsg = err.response.data.detail
        } else {
          errorMsg = `Error: ${JSON.stringify(err.response.data)}`
        }
      } else if (err.message) {
        errorMsg = err.message
      }
      
      alert(errorMsg)
    }
  }

  const WeightCanvas = ({ weights, inChannels }: { weights: number[][][], inChannels: number }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
      if (canvasRef.current && weights && weights.length > 0) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (ctx) {
          // Find dimensions
          let kernelH = 0
          let kernelW = 0
          
          // Get dimensions from first channel
          if (weights[0] && Array.isArray(weights[0])) {
            kernelH = weights[0].length
            if (weights[0][0] && Array.isArray(weights[0][0])) {
              kernelW = weights[0][0].length
            }
          }

          if (kernelH > 0 && kernelW > 0) {
            const scale = 15
            const spacing = 2
            const totalWidth = (kernelW * scale + spacing) * inChannels - spacing
            canvas.width = totalWidth
            canvas.height = kernelH * scale

            ctx.fillStyle = '#000'
            ctx.fillRect(0, 0, canvas.width, canvas.height)

            // Draw each input channel's weights side by side
            for (let c = 0; c < inChannels && c < weights.length; c++) {
              const channelWeights = weights[c]
              if (!channelWeights) continue

              const offsetX = c * (kernelW * scale + spacing)

              for (let y = 0; y < kernelH && y < channelWeights.length; y++) {
                const row = channelWeights[y]
                if (!row) continue

                for (let x = 0; x < kernelW && x < row.length; x++) {
                  const value = row[x]
                  const gray = Math.max(0, Math.min(255, Math.round(value)))
                  ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`
                  ctx.fillRect(
                    offsetX + x * scale,
                    y * scale,
                    scale,
                    scale
                  )
                }
              }
            }
          }
        }
      }
    }, [weights, inChannels])

    return <canvas ref={canvasRef} className="weight-canvas" />
  }

  return (
    <div className="modal-overlay" onClick={onClose} ref={modalRef}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">
            <h2>All Feature Maps</h2>
            <p className="modal-subtitle">
              Layer: {layerName} • {numChannels} channels
            </p>
          </div>
          <div className="modal-actions">
            <button 
              className="modal-info-button"
              onClick={() => setShowInfo(!showInfo)}
              title="Show/Hide Information"
            >
              {showInfo ? '📖 Hide Info' : '📖 Show Info'}
            </button>
            <button 
              className="modal-weights-button"
              onClick={loadWeights}
              disabled={loadingWeights}
              title="View and Edit Weights"
            >
              {loadingWeights ? 'Loading...' : '⚙️ Weights'}
            </button>
            <button className="modal-close" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>
        
        {showInfo && (
          <div className="modal-info-panel">
            <h3>🎓 Understanding Feature Maps</h3>
            <div className="info-content">
              <div className="info-section">
                <h4>What are Feature Maps?</h4>
                <p>
                  Feature maps are the output of convolutional layers. Each channel represents a different 
                  learned feature detector that scans the input image for specific patterns.
                </p>
              </div>
              
              <div className="info-section">
                <h4>Why Grayscale?</h4>
                <p>
                  Each channel outputs a <strong>single activation value</strong> per pixel (not RGB). 
                  The original RGB image (3 channels) gets processed through convolutional layers:
                </p>
                <ul>
                  <li><strong>Input:</strong> RGB image (3 channels: Red, Green, Blue)</li>
                  <li><strong>First Conv Layer:</strong> Takes 3 input channels, produces many output channels (e.g., 64)</li>
                  <li><strong>Each output channel:</strong> Detects different features (edges, textures, patterns)</li>
                  <li><strong>Color information:</strong> Gets "distributed" across multiple feature maps through learned weights</li>
                </ul>
              </div>
              
              <div className="info-section">
                <h4>What Do These Numbers Mean?</h4>
                <ul>
                  <li><strong>Channel Number (#):</strong> The index of this feature detector</li>
                  <li><strong>Mean Activation (μ):</strong> Average activation value across the entire feature map</li>
                  <li><strong>Bright areas:</strong> High activation = strong feature detection</li>
                  <li><strong>Dark areas:</strong> Low activation = weak/no feature detection</li>
                </ul>
                <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(102, 126, 234, 0.1)', borderRadius: '4px' }}>
                  <h5 style={{ color: '#667eea', margin: '0 0 0.5rem 0', fontSize: '0.95rem' }}>Understanding Mean Activation (μ)</h5>
                  <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>
                    The mean activation (μ) is the average of all activation values in the feature map. 
                    For example, <strong>μ: -0.1910</strong> means:
                  </p>
                  <ul style={{ margin: '0.5rem 0 0 0', paddingLeft: '1.5rem', fontSize: '0.9rem' }}>
                    <li><strong>Negative values (e.g., -0.1910):</strong> The feature detector is generally <strong>inhibited</strong> or <strong>not detecting</strong> its target feature in this image. Most pixels have low/negative activations.</li>
                    <li><strong>Positive values (e.g., +0.5234):</strong> The feature detector is <strong>active</strong> and finding its target feature. Higher positive values = stronger detection.</li>
                    <li><strong>Near zero (e.g., 0.0012):</strong> The feature detector is <strong>neutral</strong> - neither strongly detecting nor inhibiting.</li>
                    <li><strong>Magnitude matters:</strong> Absolute value of μ &gt; 0.1 suggests significant activation/inhibition; absolute value of μ &lt; 0.01 suggests minimal response.</li>
                  </ul>
                  <p style={{ margin: '0.75rem 0 0 0', fontSize: '0.85rem', fontStyle: 'italic', color: '#aaa' }}>
                    Note: After ReLU activation, negative values are clamped to 0, so you'll typically see non-negative values in later layers.
                  </p>
                </div>
              </div>
              
              <div className="info-section">
                <h4>How to Interpret Feature Maps</h4>
                <ul>
                  <li><strong>Early layers:</strong> Detect simple features like edges, lines, and basic textures</li>
                  <li><strong>Middle layers:</strong> Detect more complex patterns like shapes and textures</li>
                  <li><strong>Deep layers:</strong> Detect high-level features like object parts or complex patterns</li>
                  <li><strong>Activation intensity:</strong> Shows where the network "sees" the feature it's detecting</li>
                </ul>
              </div>
              
              <div className="info-section">
                <h4>Colormaps</h4>
                <p>
                  Colormaps help visualize activation <strong>intensity</strong> in color - they don't represent 
                  original RGB colors. Different colormaps can highlight different aspects:
                </p>
                <ul>
                  <li><strong>Grayscale:</strong> Classic black-to-white intensity</li>
                  <li><strong>Jet:</strong> Blue (low) → Green → Yellow → Red (high)</li>
                  <li><strong>Viridis:</strong> Purple → Blue → Green → Yellow</li>
                  <li><strong>Hot:</strong> Black → Red → Yellow → White</li>
                </ul>
              </div>
              
              <div className="info-section">
                <h4>Tips for Exploration</h4>
                <ul>
                  <li>Compare channels to see different feature detectors</li>
                  <li>Look for patterns that match parts of the input image</li>
                  <li>Higher mean activation = more active feature detector</li>
                  <li>Try different layers to see how features evolve through the network</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {showWeights && weights && (
          <div className="modal-weights-panel">
            <div className="weights-header">
              <h3>Layer Weights: {weights.layer_name}</h3>
              <button className="close-weights" onClick={() => setShowWeights(false)}>✕</button>
            </div>
            <div className="weights-info">
              <div className="weight-stats">
                <span>Shape: [{weights.shape.join(', ')}]</span>
                <span>Kernel: {weights.kernel_size.join('×')}</span>
                <span>Filters: {weights.out_channels}</span>
                <span>Min: {weights.weight_stats.min.toFixed(4)}</span>
                <span>Max: {weights.weight_stats.max.toFixed(4)}</span>
                <span>Mean: {weights.weight_stats.mean.toFixed(4)}</span>
              </div>
            </div>
            
            {selectedFilter !== null && (
              <div className="weight-editor">
                <h4>Edit Filter #{selectedFilter}</h4>
                <div className="editor-explanation">
                  <p><strong>Scale:</strong> Multiply all weights by this value (e.g., 2.0 doubles all weights, 0.5 halves them)</p>
                  <p><strong>Shift:</strong> Add this value to all weights (e.g., +0.1 increases all weights, -0.1 decreases them)</p>
                  <p><em>Example: Scale=2.0, Shift=0.1 means: new_weight = (old_weight × 2.0) + 0.1</em></p>
                </div>
                <div className="editor-controls">
                  <label>
                    Scale: 
                    <input 
                      type="number" 
                      step="0.1" 
                      value={weightScale} 
                      onChange={(e) => setWeightScale(parseFloat(e.target.value) || 1.0)}
                    />
                  </label>
                  <label>
                    Shift: 
                    <input 
                      type="number" 
                      step="0.01" 
                      value={weightShift} 
                      onChange={(e) => setWeightShift(parseFloat(e.target.value) || 0.0)}
                    />
                  </label>
                  <button onClick={() => updateWeights(selectedFilter)}>
                    Apply Changes
                  </button>
                  <button onClick={() => {
                    setWeightScale(1.0)
                    setWeightShift(0.0)
                    setSelectedFilter(null)
                  }}>
                    Reset
                  </button>
                </div>
              </div>
            )}

            <div className="weights-grid">
              {weights.filters.map((filter) => (
                <div 
                  key={filter.filter_index} 
                  className={`weight-filter-item ${selectedFilter === filter.filter_index ? 'selected' : ''}`}
                  onClick={() => setSelectedFilter(filter.filter_index)}
                >
                  <div className="weight-filter-header">
                    <span>Filter #{filter.filter_index}</span>
                    <div className="weight-stats-small">
                      <span title="Mean (μ): Average of all weight values in this filter">μ: {filter.raw_mean.toFixed(3)}</span>
                      <span title="Standard Deviation (σ): Measure of how spread out the weights are. Higher σ = more variation">σ: {filter.raw_std.toFixed(3)}</span>
                    </div>
                  </div>
                  <WeightCanvas weights={filter.weights} inChannels={weights.in_channels} />
                  <div className="weight-range">
                    Range: [{filter.raw_min.toFixed(3)}, {filter.raw_max.toFixed(3)}]
                  </div>
                  {filter.raw_weights && (
                    <details className="weight-values-details">
                      <summary>View All Weight Values</summary>
                      <div className="weight-values-grid">
                        {filter.raw_weights.map((channel, cIdx) => (
                          <div key={cIdx} className="weight-channel">
                            <div className="weight-channel-label">Channel {cIdx}:</div>
                            <div className="weight-values">
                              {channel.map((row, rIdx) => (
                                <div key={rIdx} className="weight-row">
                                  {row.map((val, vIdx) => (
                                    <span key={vIdx} className="weight-value" title={`Position [${cIdx}, ${rIdx}, ${vIdx}]`}>
                                      {val.toFixed(4)}
                                    </span>
                                  ))}
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div 
          className="modal-channels-grid"
          style={{ 
            gridTemplateColumns: `repeat(${gridColumns}, 1fr)` 
          }}
        >
          {channels.map((channel) => (
            <div key={channel.channel_index} className="modal-channel-item">
              <div className="modal-channel-header">
                <span className="channel-number">#{channel.channel_index}</span>
                <span className="channel-mean">
                  μ: {channel.mean_activation.toFixed(4)}
                </span>
              </div>
              <ChannelCanvas data={channel.data} colormap={colormap} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default AllChannelsModal

