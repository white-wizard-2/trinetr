import { useState, useEffect } from 'react'
import axios from 'axios'
import './ImageLab.css'

interface ImageLabProps {
  imageFile: File | null
  modelId: string | null
}

interface Prediction {
  label: string
  probability: number
  class_id: number
}

interface Adjustments {
  brightness: number
  contrast: number
  saturation: number
  redShift: number
  greenShift: number
  blueShift: number
  blur: number
  noise: number
  rotation: number
  flipH: boolean
  flipV: boolean
  occlusionX: number
  occlusionY: number
  occlusionSize: number
  occlusionEnabled: boolean
}

interface ProcessedData {
  original_size: number[]
  target_size: number[]
  scale_factor: number[]
  interpolation: string
  original_image: string
  processed_image: string
  original_stats: { mean: number[], std: number[] }
  processed_stats: { mean: number[], std: number[] }
}

const defaultAdjustments: Adjustments = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  redShift: 0,
  greenShift: 0,
  blueShift: 0,
  blur: 0,
  noise: 0,
  rotation: 0,
  flipH: false,
  flipV: false,
  occlusionX: 50,
  occlusionY: 50,
  occlusionSize: 20,
  occlusionEnabled: false,
}

function ImageLab({ imageFile, modelId }: ImageLabProps) {
  const [adjustments, setAdjustments] = useState<Adjustments>(defaultAdjustments)
  const [interpolation, setInterpolation] = useState('bilinear')
  const [targetSize, setTargetSize] = useState(224)
  const [processedData, setProcessedData] = useState<ProcessedData | null>(null)
  const [originalPredictions, setOriginalPredictions] = useState<Prediction[]>([])
  const [modifiedPredictions, setModifiedPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [activeTab, setActiveTab] = useState<'preprocess' | 'adjust' | 'color' | 'transform' | 'occlusion'>('preprocess')
  const [showInfo, setShowInfo] = useState(false)
  const [showKernel, setShowKernel] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState('bilinear')
  const [customKernel, setCustomKernel] = useState<number[][]>([
    [0.25, 0.25],
    [0.25, 0.25]
  ])
  const [useCustomKernel, setUseCustomKernel] = useState(false)

  // Predefined kernels for visualization
  const kernelPresets: { [key: string]: { size: number, weights: number[][], description: string } } = {
    nearest: {
      size: 1,
      weights: [[1]],
      description: 'Takes the value of the nearest pixel. Fast but pixelated.'
    },
    bilinear: {
      size: 2,
      weights: [
        [0.25, 0.25],
        [0.25, 0.25]
      ],
      description: 'Linear interpolation using 4 neighboring pixels. Weights based on distance.'
    },
    bicubic: {
      size: 4,
      weights: [
        [-0.01, -0.03, -0.03, -0.01],
        [-0.03,  0.36,  0.36, -0.03],
        [-0.03,  0.36,  0.36, -0.03],
        [-0.01, -0.03, -0.03, -0.01]
      ],
      description: 'Cubic interpolation using 16 pixels. Smoother results with slight sharpening.'
    },
    sharpen: {
      size: 3,
      weights: [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
      ],
      description: 'Enhances edges by emphasizing center and subtracting neighbors.'
    },
    blur: {
      size: 3,
      weights: [
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111]
      ],
      description: 'Box blur - averages all 9 pixels equally.'
    },
    gaussian: {
      size: 3,
      weights: [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625]
      ],
      description: 'Gaussian blur - weighted average with center emphasis.'
    },
    edge_detect: {
      size: 3,
      weights: [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
      ],
      description: 'Laplacian edge detection - highlights edges in all directions.'
    },
    emboss: {
      size: 3,
      weights: [
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
      ],
      description: 'Creates embossed/3D effect by highlighting directional changes.'
    }
  }

  const updateKernelCell = (row: number, col: number, value: string) => {
    const newKernel = customKernel.map((r, ri) => 
      r.map((c, ci) => (ri === row && ci === col) ? parseFloat(value) || 0 : c)
    )
    setCustomKernel(newKernel)
    setSelectedPreset('custom')
    setUseCustomKernel(true)
  }

  const resizeKernel = (newSize: number) => {
    const newKernel: number[][] = []
    for (let i = 0; i < newSize; i++) {
      newKernel.push(new Array(newSize).fill(0))
    }
    // Copy existing values where possible
    for (let i = 0; i < Math.min(newSize, customKernel.length); i++) {
      for (let j = 0; j < Math.min(newSize, customKernel[0].length); j++) {
        newKernel[i][j] = customKernel[i][j]
      }
    }
    // Set center to 1 if new
    if (newSize > customKernel.length) {
      const center = Math.floor(newSize / 2)
      newKernel[center][center] = 1
    }
    setCustomKernel(newKernel)
  }

  const loadPreset = (presetName: string) => {
    const preset = kernelPresets[presetName]
    if (preset) {
      setSelectedPreset(presetName)
      setCustomKernel(preset.weights.map(row => [...row]))
      setUseCustomKernel(true)
    }
  }

  const getKernelSum = () => {
    return customKernel.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0)
  }

  const normalizeKernel = () => {
    const sum = getKernelSum()
    if (sum !== 0) {
      setCustomKernel(customKernel.map(row => row.map(v => v / sum)))
    }
  }

  // Process image when file or settings change
  useEffect(() => {
    if (imageFile) {
      processImage()
    }
  }, [imageFile, interpolation, targetSize, adjustments, customKernel, useCustomKernel])

  // Fetch original predictions when model/file changes
  useEffect(() => {
    if (imageFile && modelId) {
      fetchOriginalPredictions()
    }
  }, [imageFile, modelId])

  const processImage = async () => {
    if (!imageFile) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const params = new URLSearchParams({
        target_size: targetSize.toString(),
        interpolation: interpolation,
        brightness: adjustments.brightness.toString(),
        contrast: adjustments.contrast.toString(),
        saturation: adjustments.saturation.toString(),
        red_shift: adjustments.redShift.toString(),
        green_shift: adjustments.greenShift.toString(),
        blue_shift: adjustments.blueShift.toString(),
        blur: adjustments.blur.toString(),
        use_custom_kernel: useCustomKernel.toString(),
        custom_kernel: JSON.stringify(customKernel),
        noise: adjustments.noise.toString(),
        rotation: adjustments.rotation.toString(),
        flip_h: adjustments.flipH.toString(),
        flip_v: adjustments.flipV.toString(),
        occlusion_enabled: adjustments.occlusionEnabled.toString(),
        occlusion_x: adjustments.occlusionX.toString(),
        occlusion_y: adjustments.occlusionY.toString(),
        occlusion_size: adjustments.occlusionSize.toString(),
      })

      const response = await axios.post(
        `http://localhost:8000/preprocess/transform?${params}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setProcessedData(response.data)
    } catch (err: any) {
      console.error('Failed to process image:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchOriginalPredictions = async () => {
    if (!imageFile || !modelId) return

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await axios.post(
        `http://localhost:8000/models/${modelId}/predict`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setOriginalPredictions(response.data.predictions.slice(0, 5))
    } catch (err) {
      console.error('Failed to fetch predictions:', err)
    }
  }

  const runModifiedPrediction = async () => {
    if (!processedData?.processed_image || !modelId) return

    setPredicting(true)
    try {
      // Convert base64 to blob
      const base64Data = processedData.processed_image.split(',')[1]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/png' })

      const formData = new FormData()
      formData.append('file', blob, 'processed.png')

      const response = await axios.post(
        `http://localhost:8000/models/${modelId}/predict`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setModifiedPredictions(response.data.predictions.slice(0, 5))
    } catch (err) {
      console.error('Failed to fetch modified predictions:', err)
    } finally {
      setPredicting(false)
    }
  }

  const resetAdjustments = () => {
    setAdjustments(defaultAdjustments)
    setModifiedPredictions([])
  }

  const updateAdjustment = (key: keyof Adjustments, value: number | boolean) => {
    setAdjustments((prev) => ({ ...prev, [key]: value }))
  }

  const hasAdjustments = () => {
    if (useCustomKernel && selectedPreset !== 'bilinear') return true
    return Object.entries(adjustments).some(([key, value]) => {
      if (key === 'occlusionX' || key === 'occlusionY' || key === 'occlusionSize') return false
      if (typeof value === 'boolean') return value
      return value !== 0
    })
  }

  if (!imageFile) {
    return (
      <div className="image-lab">
        <div className="info-box">
          <h3>🔬 Image Lab</h3>
          <p>Upload an image to experiment with preprocessing and transformations</p>
        </div>
      </div>
    )
  }

  const getPredictionDelta = (label: string) => {
    const orig = originalPredictions.find((p) => p.label === label)
    const mod = modifiedPredictions.find((p) => p.label === label)
    if (!orig || !mod) return null
    return mod.probability - orig.probability
  }

  const getScaleType = () => {
    if (!processedData) return ''
    const scale = processedData.scale_factor[0]
    if (scale < 1) return 'downscale'
    if (scale > 1) return 'upscale'
    return 'no-scale'
  }

  return (
    <div className="image-lab">
      <div className="lab-header">
        <h2>🔬 Image Lab</h2>
        <div className="lab-actions">
          {modelId && (
            <button onClick={runModifiedPrediction} disabled={predicting || loading} className="run-btn">
              {predicting ? '...' : '▶ Predict'}
            </button>
          )}
          {hasAdjustments() && (
            <button onClick={resetAdjustments} className="reset-btn">↺ Reset</button>
          )}
          <button className="info-btn" onClick={() => setShowInfo(!showInfo)}>
            {showInfo ? '✕' : 'ℹ️'}
          </button>
        </div>
      </div>

      {showInfo && (
        <div className="info-panel">
          <p><strong>Interpolation kernels</strong> determine how pixels are sampled when resizing:</p>
          <ul>
            <li><strong>nearest</strong> - Fastest, pixelated. Takes nearest pixel.</li>
            <li><strong>bilinear</strong> - Smooth, uses 4 neighbors (default)</li>
            <li><strong>bicubic</strong> - Smoother, uses 16 neighbors</li>
            <li><strong>lanczos</strong> - Highest quality for downscaling</li>
          </ul>
        </div>
      )}

      <div className="lab-content">
        <div className="images-section">
          {processedData && (
            <>
              <div className="image-box">
                <h4>Original ({processedData.original_size[0]}×{processedData.original_size[1]})</h4>
                <img src={processedData.original_image} alt="Original" />
              </div>
              <div className="arrow">→</div>
              <div className="image-box processed">
                <h4>Processed ({processedData.target_size[0]}×{processedData.target_size[1]})</h4>
                <img src={processedData.processed_image} alt="Processed" />
                {hasAdjustments() && <span className="modified-badge">Modified</span>}
              </div>
            </>
          )}
          {loading && <div className="loading-overlay">Processing...</div>}
        </div>

        <div className="scale-info">
          <span className={`scale-badge ${getScaleType()}`}>
            {getScaleType() === 'downscale' ? '⬇️' : getScaleType() === 'upscale' ? '⬆️' : '↔️'}
            {processedData && ` ${(processedData.scale_factor[0] * 100).toFixed(0)}%`}
          </span>
          <span className="interpolation-badge">{interpolation}</span>
          <button className="kernel-btn" onClick={() => setShowKernel(!showKernel)}>
            🔢 {showKernel ? 'Hide' : 'Show'} Kernel
          </button>
        </div>

        {showKernel && (
          <div className="kernel-panel">
            <div className="kernel-header">
              <h4>Convolution Kernel: {selectedPreset}</h4>
              <p className="kernel-desc">
                {kernelPresets[selectedPreset]?.description || 'Custom kernel - edit values below'}
              </p>
              {useCustomKernel && (
                <span className="kernel-active-badge">✓ Applied to image</span>
              )}
            </div>
            
            <div className="kernel-content">
              <div className="kernel-presets">
                <label>Load Preset:</label>
                <div className="preset-buttons">
                  {Object.keys(kernelPresets).map(name => (
                    <button 
                      key={name} 
                      onClick={() => loadPreset(name)}
                      className={selectedPreset === name ? 'active' : ''}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="kernel-editor">
                <div className="kernel-size-control">
                  <label>Kernel Size: {customKernel.length}×{customKernel.length}</label>
                  <input 
                    type="range" 
                    min="1" 
                    max="7" 
                    step="2"
                    value={customKernel.length}
                    onChange={(e) => resizeKernel(Number(e.target.value))}
                  />
                </div>
                
                <div className="kernel-grid" style={{ 
                  gridTemplateColumns: `repeat(${customKernel.length}, 1fr)` 
                }}>
                  {customKernel.map((row, ri) => 
                    row.map((val, ci) => (
                      <input
                        key={`${ri}-${ci}`}
                        type="number"
                        step="0.01"
                        value={val.toFixed(2)}
                        onChange={(e) => updateKernelCell(ri, ci, e.target.value)}
                        className={`kernel-cell ${val > 0 ? 'positive' : val < 0 ? 'negative' : 'zero'}`}
                        style={{
                          backgroundColor: val > 0 
                            ? `rgba(67, 233, 123, ${Math.min(Math.abs(val), 1) * 0.5})` 
                            : val < 0 
                              ? `rgba(255, 107, 107, ${Math.min(Math.abs(val), 1) * 0.5})`
                              : 'transparent'
                        }}
                      />
                    ))
                  )}
                </div>

                <div className="kernel-stats">
                  <span>Sum: {getKernelSum().toFixed(3)}</span>
                  <button onClick={normalizeKernel} className="normalize-btn">
                    Normalize to 1.0
                  </button>
                </div>
              </div>

              <div className="kernel-explanation">
                <h5>How Kernels Work:</h5>
                <p>Each cell represents a weight multiplied with the corresponding pixel. 
                The weighted sum produces the output pixel value.</p>
                <ul>
                  <li><span className="pos-indicator">Green</span> = positive (brightens)</li>
                  <li><span className="neg-indicator">Red</span> = negative (darkens/edges)</li>
                  <li>Sum ≈ 1.0 preserves brightness</li>
                  <li>Sum = 0 detects edges</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        <div className="controls-section">
          <div className="tabs">
            {['preprocess', 'adjust', 'color', 'transform', 'occlusion'].map((tab) => (
              <button
                key={tab}
                className={activeTab === tab ? 'active' : ''}
                onClick={() => setActiveTab(tab as typeof activeTab)}
              >
                {tab === 'preprocess' ? '📐' : tab === 'adjust' ? '🎚️' : tab === 'color' ? '🎨' : tab === 'transform' ? '🔄' : '⬛'}
              </button>
            ))}
          </div>

          <div className="tab-panel">
            {activeTab === 'preprocess' && (
              <div className="controls-grid">
                <div className="control">
                  <label>Interpolation:</label>
                  <select value={interpolation} onChange={(e) => setInterpolation(e.target.value)}>
                    <option value="nearest">nearest</option>
                    <option value="bilinear">bilinear</option>
                    <option value="bicubic">bicubic</option>
                    <option value="lanczos">lanczos</option>
                    <option value="box">box</option>
                    <option value="hamming">hamming</option>
                  </select>
                </div>
                <div className="control">
                  <label>Target Size:</label>
                  <select value={targetSize} onChange={(e) => setTargetSize(Number(e.target.value))}>
                    <option value={224}>224×224</option>
                    <option value={256}>256×256</option>
                    <option value={299}>299×299</option>
                    <option value={384}>384×384</option>
                  </select>
                </div>
              </div>
            )}

            {activeTab === 'adjust' && (
              <div className="controls-grid">
                <div className="control">
                  <label>Brightness: {adjustments.brightness}</label>
                  <input type="range" min="-100" max="100" value={adjustments.brightness}
                    onChange={(e) => updateAdjustment('brightness', Number(e.target.value))} />
                </div>
                <div className="control">
                  <label>Contrast: {adjustments.contrast}</label>
                  <input type="range" min="-100" max="100" value={adjustments.contrast}
                    onChange={(e) => updateAdjustment('contrast', Number(e.target.value))} />
                </div>
                <div className="control">
                  <label>Saturation: {adjustments.saturation}</label>
                  <input type="range" min="-100" max="100" value={adjustments.saturation}
                    onChange={(e) => updateAdjustment('saturation', Number(e.target.value))} />
                </div>
                <div className="control">
                  <label>Blur: {adjustments.blur}px</label>
                  <input type="range" min="0" max="10" value={adjustments.blur}
                    onChange={(e) => updateAdjustment('blur', Number(e.target.value))} />
                </div>
                <div className="control">
                  <label>Noise: {adjustments.noise}</label>
                  <input type="range" min="0" max="50" value={adjustments.noise}
                    onChange={(e) => updateAdjustment('noise', Number(e.target.value))} />
                </div>
              </div>
            )}

            {activeTab === 'color' && (
              <div className="controls-grid">
                <div className="control red">
                  <label>Red: {adjustments.redShift > 0 ? '+' : ''}{adjustments.redShift}</label>
                  <input type="range" min="-100" max="100" value={adjustments.redShift}
                    onChange={(e) => updateAdjustment('redShift', Number(e.target.value))} />
                </div>
                <div className="control green">
                  <label>Green: {adjustments.greenShift > 0 ? '+' : ''}{adjustments.greenShift}</label>
                  <input type="range" min="-100" max="100" value={adjustments.greenShift}
                    onChange={(e) => updateAdjustment('greenShift', Number(e.target.value))} />
                </div>
                <div className="control blue">
                  <label>Blue: {adjustments.blueShift > 0 ? '+' : ''}{adjustments.blueShift}</label>
                  <input type="range" min="-100" max="100" value={adjustments.blueShift}
                    onChange={(e) => updateAdjustment('blueShift', Number(e.target.value))} />
                </div>
              </div>
            )}

            {activeTab === 'transform' && (
              <div className="controls-grid">
                <div className="control">
                  <label>Rotation: {adjustments.rotation}°</label>
                  <input type="range" min="-180" max="180" value={adjustments.rotation}
                    onChange={(e) => updateAdjustment('rotation', Number(e.target.value))} />
                </div>
                <div className="checkbox-row">
                  <label>
                    <input type="checkbox" checked={adjustments.flipH}
                      onChange={(e) => updateAdjustment('flipH', e.target.checked)} />
                    Flip H
                  </label>
                  <label>
                    <input type="checkbox" checked={adjustments.flipV}
                      onChange={(e) => updateAdjustment('flipV', e.target.checked)} />
                    Flip V
                  </label>
                </div>
              </div>
            )}

            {activeTab === 'occlusion' && (
              <div className="controls-grid">
                <div className="checkbox-row">
                  <label>
                    <input type="checkbox" checked={adjustments.occlusionEnabled}
                      onChange={(e) => updateAdjustment('occlusionEnabled', e.target.checked)} />
                    Enable gray patch
                  </label>
                </div>
                <div className="control">
                  <label>X: {adjustments.occlusionX}%</label>
                  <input type="range" min="0" max="100" value={adjustments.occlusionX}
                    onChange={(e) => updateAdjustment('occlusionX', Number(e.target.value))}
                    disabled={!adjustments.occlusionEnabled} />
                </div>
                <div className="control">
                  <label>Y: {adjustments.occlusionY}%</label>
                  <input type="range" min="0" max="100" value={adjustments.occlusionY}
                    onChange={(e) => updateAdjustment('occlusionY', Number(e.target.value))}
                    disabled={!adjustments.occlusionEnabled} />
                </div>
                <div className="control">
                  <label>Size: {adjustments.occlusionSize}%</label>
                  <input type="range" min="5" max="50" value={adjustments.occlusionSize}
                    onChange={(e) => updateAdjustment('occlusionSize', Number(e.target.value))}
                    disabled={!adjustments.occlusionEnabled} />
                </div>
              </div>
            )}
          </div>
        </div>

        {modelId && (
          <div className="predictions-section">
            <div className="pred-column">
              <h4>Original</h4>
              {originalPredictions.length > 0 ? (
                originalPredictions.map((p, i) => (
                  <div key={i} className="pred-item">
                    <span className="label">{p.label}</span>
                    <span className="prob">{(p.probability * 100).toFixed(1)}%</span>
                  </div>
                ))
              ) : (
                <span className="loading-text">Loading...</span>
              )}
            </div>
            <div className="pred-column modified">
              <h4>Modified</h4>
              {modifiedPredictions.length > 0 ? (
                modifiedPredictions.map((p, i) => {
                  const delta = getPredictionDelta(p.label)
                  return (
                    <div key={i} className="pred-item">
                      <span className="label">{p.label}</span>
                      <span className="prob">{(p.probability * 100).toFixed(1)}%</span>
                      {delta !== null && (
                        <span className={`delta ${delta > 0 ? 'pos' : delta < 0 ? 'neg' : ''}`}>
                          {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  )
                })
              ) : (
                <span className="hint">Click Predict</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageLab

