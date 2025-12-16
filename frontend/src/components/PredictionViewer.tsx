import { useState, useEffect } from 'react'
import axios from 'axios'
import './PredictionViewer.css'

interface Prediction {
  class_id: number
  label: string
  probability: number
  logit: number
}

interface PredictionViewerProps {
  modelId: string
  imageFile: File | null
}

function PredictionViewer({ modelId, imageFile }: PredictionViewerProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [topPrediction, setTopPrediction] = useState<any>(null)
  const [finalLayerActivations, setFinalLayerActivations] = useState<number[] | null>(null)
  const [finalLayerName, setFinalLayerName] = useState<string | null>(null)

  useEffect(() => {
    const fetchPredictions = async () => {
      if (!modelId || !imageFile) {
        setPredictions([])
        setTopPrediction(null)
        return
      }

      setLoading(true)
      setError(null)

      try {
        const formData = new FormData()
        formData.append('file', imageFile)

        const response = await axios.post(
          `http://localhost:8000/models/${modelId}/predict`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' }
          }
        )

        setPredictions(response.data.predictions)
        setTopPrediction(response.data.top_prediction)
        setFinalLayerActivations(response.data.final_layer_activations)
        setFinalLayerName(response.data.final_layer_name)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load predictions')
      } finally {
        setLoading(false)
      }
    }

    fetchPredictions()
  }, [modelId, imageFile])

  if (!modelId || !imageFile) {
    return (
      <div className="prediction-viewer">
        <div className="info-message">
          Load a model and upload an image to see predictions
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="prediction-viewer">
        <div className="loading">Analyzing image...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="prediction-viewer">
        <div className="error">{error}</div>
      </div>
    )
  }

  return (
    <div className="prediction-viewer">
      <h2>Final Layer Predictions</h2>
      
      {topPrediction && (
        <div className="top-prediction">
          <div className="prediction-badge">
            <div className="prediction-label">Predicted Class</div>
            <div className="prediction-name">{topPrediction.label}</div>
            <div className="prediction-confidence">
              {(topPrediction.probability * 100).toFixed(2)}% confident
            </div>
            <div className="prediction-details">
              <span>Class ID: {topPrediction.class_id}</span>
              <span>Logit: {topPrediction.logit.toFixed(4)}</span>
            </div>
          </div>
        </div>
      )}

      {finalLayerName && (
        <div className="layer-info">
          <h3>Final Layer: {finalLayerName}</h3>
          <p className="layer-description">
            This is the fully connected (Linear) layer that outputs 1000 logits, 
            one for each ImageNet class. The highest logit becomes the predicted class.
          </p>
          {finalLayerActivations && (
            <div className="activations-info">
              <p>Output shape: [{finalLayerActivations.length}] (1000 classes)</p>
              <p>Min logit: {Math.min(...finalLayerActivations).toFixed(4)}</p>
              <p>Max logit: {Math.max(...finalLayerActivations).toFixed(4)}</p>
              <p>Mean logit: {(finalLayerActivations.reduce((a, b) => a + b, 0) / finalLayerActivations.length).toFixed(4)}</p>
            </div>
          )}
        </div>
      )}

      <div className="predictions-list">
        <h3>Top 10 Predictions</h3>
        <div className="predictions-grid">
          {predictions.map((pred, idx) => (
            <div 
              key={pred.class_id} 
              className={`prediction-item ${idx === 0 ? 'top-pred' : ''}`}
            >
              <div className="pred-rank">#{idx + 1}</div>
              <div className="pred-content">
                <div className="pred-label">{pred.label}</div>
                <div className="pred-prob">
                  {(pred.probability * 100).toFixed(2)}%
                </div>
                <div className="pred-meta">
                  <span>ID: {pred.class_id}</span>
                  <span>Logit: {pred.logit.toFixed(2)}</span>
                </div>
              </div>
              <div className="prob-bar-container">
                <div 
                  className="prob-bar" 
                  style={{ width: `${pred.probability * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="learning-section">
        <h3>🎓 How It Works</h3>
        <ul>
          <li><strong>Logits:</strong> Raw scores from the final layer (before softmax)</li>
          <li><strong>Softmax:</strong> Converts logits to probabilities (sums to 1.0)</li>
          <li><strong>Prediction:</strong> The class with the highest probability</li>
          <li><strong>Confidence:</strong> How certain the model is (higher = more confident)</li>
        </ul>
      </div>
    </div>
  )
}

export default PredictionViewer

