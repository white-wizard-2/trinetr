import { useRef, useEffect } from 'react'
import './LinearLayerVisualizer.css'

interface LinearLayerVisualizerProps {
  activations: number[]
  topActivations?: Array<{
    index: number
    value: number
    label: string
  }>
  layerName: string
}

function LinearLayerVisualizer({ activations, topActivations }: LinearLayerVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (canvasRef.current && activations) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (ctx) {
        const width = 1000
        const height = 200
        canvas.width = width
        canvas.height = height

        // Clear canvas
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, width, height)

        // Find min/max for normalization
        const minVal = Math.min(...activations)
        const maxVal = Math.max(...activations)
        const range = maxVal - minVal || 1

        // Draw bar chart
        const barWidth = width / activations.length
        for (let i = 0; i < activations.length; i++) {
          const value = activations[i]
          const normalized = (value - minVal) / range
          const barHeight = normalized * height

          // Color based on value (green for positive, red for negative)
          if (value > 0) {
            ctx.fillStyle = `rgba(67, 233, 123, ${Math.min(normalized + 0.3, 1)})`
          } else {
            ctx.fillStyle = `rgba(255, 107, 107, ${Math.min(Math.abs(normalized) + 0.3, 1)})`
          }

          ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight)
        }

        // Draw zero line
        ctx.strokeStyle = '#888'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, height)
        ctx.lineTo(width, height)
        ctx.stroke()
      }
    }
  }, [activations])

  return (
    <div className="linear-layer-visualizer">
      <h3>Fully Connected Layer Outputs</h3>
      <p className="linear-description">
        This layer outputs 1000 values (logits), one for each ImageNet class. 
        Each value represents how strongly the model thinks the image belongs to that class.
      </p>
      
      <div className="linear-chart">
        <canvas ref={canvasRef} className="linear-canvas" />
        <div className="chart-labels">
          <span>0</span>
          <span>500</span>
          <span>1000 classes</span>
        </div>
      </div>

      {topActivations && topActivations.length > 0 && (
        <div className="top-activations">
          <h4>Top 20 Class Predictions</h4>
          <div className="activations-list">
            {topActivations.map((act, idx) => (
              <div key={act.index} className={`activation-item ${idx === 0 ? 'top-pred' : ''}`}>
                <div className="activation-rank">#{idx + 1}</div>
                <div className="activation-content">
                  <div className="activation-label">{act.label}</div>
                  <div className="activation-value">
                    {act.value.toFixed(4)} (Class {act.index})
                  </div>
                </div>
                <div className="activation-bar-container">
                  <div 
                    className="activation-bar" 
                    style={{ 
                      width: `${Math.abs(act.value / Math.max(...topActivations.map(a => Math.abs(a.value)))) * 100}%`,
                      backgroundColor: act.value > 0 ? '#43e97b' : '#ff6b6b'
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="linear-explanation">
        <h4>🎓 How the 1000 Outputs Work</h4>
        <ul>
          <li><strong>Each output</strong> corresponds to one ImageNet class (e.g., output[0] = "tench", output[1] = "goldfish", etc.)</li>
          <li><strong>Higher values</strong> = stronger belief that the image belongs to that class</li>
          <li><strong>Negative values</strong> = the model thinks it's NOT that class</li>
          <li><strong>Softmax</strong> converts these logits to probabilities (sums to 1.0)</li>
          <li><strong>The highest value</strong> becomes the predicted class</li>
        </ul>
        <div className="mapping-diagram">
          <div className="mapping-step">
            <strong>Input to FC:</strong> Flattened feature vector (e.g., 512 or 2048 values)
          </div>
          <div className="mapping-arrow">↓</div>
          <div className="mapping-step">
            <strong>FC Layer:</strong> Matrix multiplication: [features] × [weight_matrix] = [1000 logits]
          </div>
          <div className="mapping-arrow">↓</div>
          <div className="mapping-step">
            <strong>Output:</strong> 1000 values, one per ImageNet class
          </div>
          <div className="mapping-arrow">↓</div>
          <div className="mapping-step">
            <strong>Softmax:</strong> Converts to probabilities → Highest = Prediction
          </div>
        </div>
      </div>
    </div>
  )
}

export default LinearLayerVisualizer

