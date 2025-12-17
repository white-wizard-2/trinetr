import { useState, useEffect } from 'react'
import axios from 'axios'
import './PreprocessingVisualizer.css'

interface PreprocessingVisualizerProps {
  imageFile: File | null
  onInterpolationChange?: (interpolation: string) => void
}

interface PreprocessingData {
  original_size: number[]
  target_size: number[]
  scale_factor: number[]
  interpolation: string
  resized_image: string
  original_stats: {
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
  }
  resized_stats: {
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
  }
  available_interpolations: string[]
  interpolation_info: { [key: string]: string }
}

function PreprocessingVisualizer({ imageFile, onInterpolationChange }: PreprocessingVisualizerProps) {
  const [data, setData] = useState<PreprocessingData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [interpolation, setInterpolation] = useState('bilinear')
  const [targetSize, setTargetSize] = useState(224)
  const [originalPreview, setOriginalPreview] = useState<string | null>(null)
  const [showInfo, setShowInfo] = useState(false)

  useEffect(() => {
    if (imageFile) {
      setOriginalPreview(URL.createObjectURL(imageFile))
      fetchPreprocessing()
    }
    return () => {
      if (originalPreview) {
        URL.revokeObjectURL(originalPreview)
      }
    }
  }, [imageFile])

  useEffect(() => {
    if (imageFile) {
      fetchPreprocessing()
    }
  }, [interpolation, targetSize])

  const fetchPreprocessing = async () => {
    if (!imageFile) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await axios.post(
        `http://localhost:8000/preprocess/visualize?target_size=${targetSize}&interpolation=${interpolation}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setData(response.data)
      onInterpolationChange?.(interpolation)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch preprocessing data')
    } finally {
      setLoading(false)
    }
  }

  if (!imageFile) {
    return (
      <div className="preprocessing-visualizer">
        <div className="info-box">
          <h3>🖼️ Image Preprocessing</h3>
          <p>Upload an image to see how it's preprocessed for the model</p>
        </div>
      </div>
    )
  }

  const getScaleType = () => {
    if (!data) return ''
    const scale = data.scale_factor[0]
    if (scale < 1) return 'downscale'
    if (scale > 1) return 'upscale'
    return 'no-scale'
  }

  return (
    <div className="preprocessing-visualizer">
      <div className="preprocessing-header">
        <h2>🖼️ Image Preprocessing</h2>
        <button 
          className="info-toggle"
          onClick={() => setShowInfo(!showInfo)}
          title="Learn about interpolation methods"
        >
          {showInfo ? '✕' : 'ℹ️'}
        </button>
      </div>

      {showInfo && data && (
        <div className="interpolation-info">
          <h4>Interpolation Methods</h4>
          <div className="info-grid">
            {Object.entries(data.interpolation_info).map(([method, desc]) => (
              <div key={method} className={`info-item ${method === interpolation ? 'active' : ''}`}>
                <strong>{method}</strong>
                <p>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="preprocessing-controls">
        <div className="control-group">
          <label>Interpolation Kernel:</label>
          <select 
            value={interpolation} 
            onChange={(e) => setInterpolation(e.target.value)}
            disabled={loading}
          >
            {data?.available_interpolations?.map(method => (
              <option key={method} value={method}>{method}</option>
            )) || (
              <>
                <option value="nearest">nearest</option>
                <option value="bilinear">bilinear</option>
                <option value="bicubic">bicubic</option>
                <option value="lanczos">lanczos</option>
                <option value="box">box</option>
                <option value="hamming">hamming</option>
              </>
            )}
          </select>
        </div>
        <div className="control-group">
          <label>Target Size:</label>
          <select 
            value={targetSize} 
            onChange={(e) => setTargetSize(Number(e.target.value))}
            disabled={loading}
          >
            <option value={224}>224×224 (Standard)</option>
            <option value={256}>256×256</option>
            <option value={299}>299×299 (Inception)</option>
            <option value={384}>384×384</option>
            <option value={512}>512×512</option>
          </select>
        </div>
      </div>

      {loading && <div className="loading">Processing...</div>}
      {error && <div className="error">{error}</div>}

      {data && !loading && (
        <>
          <div className="scale-indicator">
            <span className={`scale-badge ${getScaleType()}`}>
              {getScaleType() === 'downscale' ? '⬇️ Downscaling' : 
               getScaleType() === 'upscale' ? '⬆️ Upscaling' : '↔️ No Scaling'}
            </span>
            <span className="scale-factor">
              {data.original_size[0]}×{data.original_size[1]} → {data.target_size[0]}×{data.target_size[1]}
              <span className="scale-percent">
                ({(data.scale_factor[0] * 100).toFixed(1)}%)
              </span>
            </span>
          </div>

          <div className="image-comparison">
            <div className="image-panel original">
              <h4>Original</h4>
              <div className="image-container">
                {originalPreview && <img src={originalPreview} alt="Original" />}
              </div>
              <div className="image-stats">
                <div className="stat-row">
                  <span>Size:</span>
                  <span>{data.original_size[0]}×{data.original_size[1]}</span>
                </div>
                <div className="stat-row">
                  <span>Mean (RGB):</span>
                  <span className="rgb-values">
                    <span className="r">{data.original_stats.mean[0].toFixed(1)}</span>
                    <span className="g">{data.original_stats.mean[1].toFixed(1)}</span>
                    <span className="b">{data.original_stats.mean[2].toFixed(1)}</span>
                  </span>
                </div>
              </div>
            </div>

            <div className="transform-arrow">
              <div className="arrow-line"></div>
              <div className="kernel-label">{interpolation}</div>
              <div className="arrow-head">→</div>
            </div>

            <div className="image-panel processed">
              <h4>Processed ({interpolation})</h4>
              <div className="image-container">
                <img src={data.resized_image} alt="Processed" />
              </div>
              <div className="image-stats">
                <div className="stat-row">
                  <span>Size:</span>
                  <span>{data.target_size[0]}×{data.target_size[1]}</span>
                </div>
                <div className="stat-row">
                  <span>Mean (RGB):</span>
                  <span className="rgb-values">
                    <span className="r">{data.resized_stats.mean[0].toFixed(1)}</span>
                    <span className="g">{data.resized_stats.mean[1].toFixed(1)}</span>
                    <span className="b">{data.resized_stats.mean[2].toFixed(1)}</span>
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="stats-comparison">
            <h4>Pixel Statistics Change</h4>
            <div className="stats-table">
              <div className="stats-header">
                <span>Channel</span>
                <span>Original Mean</span>
                <span>Processed Mean</span>
                <span>Δ Mean</span>
              </div>
              {['Red', 'Green', 'Blue'].map((channel, i) => {
                const origMean = data.original_stats.mean[i]
                const procMean = data.resized_stats.mean[i]
                const delta = procMean - origMean
                return (
                  <div key={channel} className={`stats-row ${channel.toLowerCase()}`}>
                    <span>{channel}</span>
                    <span>{origMean.toFixed(2)}</span>
                    <span>{procMean.toFixed(2)}</span>
                    <span className={delta > 0 ? 'positive' : delta < 0 ? 'negative' : ''}>
                      {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default PreprocessingVisualizer

