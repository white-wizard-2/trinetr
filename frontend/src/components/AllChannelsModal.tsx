import { useEffect, useRef, useState } from 'react'
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

function AllChannelsModal({ channels, layerName, numChannels, colormap = 'grayscale', onClose }: AllChannelsModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)
  const [gridColumns, setGridColumns] = useState(6)
  const [showInfo, setShowInfo] = useState(false)

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

