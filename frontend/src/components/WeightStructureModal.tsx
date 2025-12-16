import { useState, useEffect } from 'react'
import axios from 'axios'
import './WeightStructureModal.css'

interface ParameterInfo {
  shape: number[]
  numel: number
  requires_grad: boolean
  dtype: string
  min: number
  max: number
  mean: number
  std: number
}

interface BufferInfo {
  shape: number[]
  numel: number
  dtype: string
}

interface LayerWeightInfo {
  layer_name: string
  layer_type: string
  parameters: { [key: string]: ParameterInfo }
  buffers?: { [key: string]: BufferInfo }
  details?: any
}

interface WeightStructureData {
  model_id: string
  total_parameters: number
  trainable_parameters: number
  non_trainable_parameters: number
  layers: LayerWeightInfo[]
}

interface WeightStructureModalProps {
  modelId: string | null
  modelName: string
  onClose: () => void
}

function WeightStructureModal({ modelId, modelName, onClose }: WeightStructureModalProps) {
  const [data, setData] = useState<WeightStructureData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedLayers, setExpandedLayers] = useState<Set<string>>(new Set())
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)

  useEffect(() => {
    if (modelId) {
      loadWeightStructure()
    }
  }, [modelId])

  const loadWeightStructure = async () => {
    if (!modelId) return
    
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get(`http://localhost:8000/models/${modelId}/weight-structure`)
      setData(response.data)
      // Auto-expand first few layers
      const firstFew = new Set(response.data.layers.slice(0, 3).map(l => l.layer_name))
      setExpandedLayers(firstFew)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load weight structure')
    } finally {
      setLoading(false)
    }
  }

  const toggleLayer = (layerName: string) => {
    const newExpanded = new Set(expandedLayers)
    if (newExpanded.has(layerName)) {
      newExpanded.delete(layerName)
    } else {
      newExpanded.add(layerName)
    }
    setExpandedLayers(newExpanded)
  }

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`
    return num.toString()
  }

  const formatShape = (shape: number[]): string => {
    return `[${shape.join(', ')}]`
  }

  if (!modelId) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content weight-structure-modal" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2>Weight Structure</h2>
            <button className="modal-close" onClick={onClose}>✕</button>
          </div>
          <div className="modal-body">
            <p>Please load a model first to view its weight structure.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content weight-structure-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Weight Structure: {modelName}</h2>
            {data && (
              <div className="weight-summary">
                <span>Total: {formatNumber(data.total_parameters)} params</span>
                <span>Trainable: {formatNumber(data.trainable_parameters)}</span>
                <span>Frozen: {formatNumber(data.non_trainable_parameters)}</span>
              </div>
            )}
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        
        <div className="modal-body">
          {loading && <div className="loading">Loading weight structure...</div>}
          {error && <div className="error">{error}</div>}
          
          {data && (
            <>
              <div className="weight-explanation">
                <h3>🎓 Understanding Weight Storage</h3>
                <p>
                  Neural networks store their learned knowledge in <strong>parameters</strong> (weights and biases) 
                  and <strong>buffers</strong> (non-learnable statistics). Each layer has its own set of parameters 
                  that define how it transforms data.
                </p>
                <ul>
                  <li><strong>Weight:</strong> The learnable parameters that transform input data</li>
                  <li><strong>Bias:</strong> An additive parameter (optional, used in Conv2d and Linear layers)</li>
                  <li><strong>Buffers:</strong> Non-learnable parameters (e.g., running_mean, running_var in BatchNorm)</li>
                  <li><strong>Shape:</strong> Dimensions of the parameter tensor</li>
                  <li><strong>Numel:</strong> Total number of elements (product of shape dimensions)</li>
                </ul>
              </div>

              <div className="layers-list">
                {data.layers.map((layer) => {
                  const isExpanded = expandedLayers.has(layer.layer_name)
                  const paramCount = Object.values(layer.parameters).reduce((sum, p) => sum + p.numel, 0)
                  
                  return (
                    <div key={layer.layer_name} className="layer-weight-item">
                      <div 
                        className="layer-weight-header"
                        onClick={() => toggleLayer(layer.layer_name)}
                      >
                        <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
                        <span className="layer-name">{layer.layer_name}</span>
                        <span className="layer-type">{layer.layer_type}</span>
                        <span className="param-count">{formatNumber(paramCount)} params</span>
                      </div>
                      
                      {isExpanded && (
                        <div className="layer-weight-details">
                          {layer.details && (
                            <div className="layer-details">
                              <h4>Layer Configuration:</h4>
                              <div className="details-explanation">
                                <p>
                                  These are the <strong>hyperparameters</strong> that define how this layer operates. 
                                  They are set during model design and don't change during training.
                                </p>
                              </div>
                              <pre>{JSON.stringify(layer.details, null, 2)}</pre>
                              {layer.layer_type === 'Conv2d' && (
                                <div className="type-explanation">
                                  <strong>Conv2d Parameters:</strong>
                                  <ul>
                                    <li><strong>in_channels:</strong> Number of input feature maps</li>
                                    <li><strong>out_channels:</strong> Number of output feature maps (filters)</li>
                                    <li><strong>kernel_size:</strong> Size of the convolution filter (e.g., 3×3)</li>
                                    <li><strong>stride:</strong> How much the filter moves each step</li>
                                    <li><strong>padding:</strong> Pixels added around the input</li>
                                  </ul>
                                </div>
                              )}
                              {layer.layer_type === 'Linear' && (
                                <div className="type-explanation">
                                  <strong>Linear (FC) Parameters:</strong>
                                  <ul>
                                    <li><strong>in_features:</strong> Number of input neurons</li>
                                    <li><strong>out_features:</strong> Number of output neurons</li>
                                    <li><strong>has_bias:</strong> Whether this layer uses a bias term</li>
                                  </ul>
                                </div>
                              )}
                              {layer.layer_type === 'BatchNorm2d' && (
                                <div className="type-explanation">
                                  <strong>BatchNorm Parameters:</strong>
                                  <ul>
                                    <li><strong>num_features:</strong> Number of channels being normalized</li>
                                    <li><strong>eps:</strong> Small value to prevent division by zero</li>
                                    <li><strong>momentum:</strong> How quickly running statistics update</li>
                                    <li><strong>affine:</strong> Whether to use learnable scale/shift parameters</li>
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}
                          
                          <div className="parameters-section">
                            <div className="section-header-with-explanation">
                              <h4>Parameters (Weights & Biases):</h4>
                              <div className="section-explanation">
                                <p>
                                  <strong>Parameters</strong> are the learnable values that the model adjusts during training. 
                                  Each parameter has a specific role:
                                </p>
                                <ul>
                                  <li><strong>weight:</strong> The main transformation matrix/kernel that processes input data</li>
                                  <li><strong>bias:</strong> An additive constant that shifts the output (optional in some layers)</li>
                                </ul>
                                <p>
                                  The <strong>shape</strong> determines how data flows: for Conv2d, weight shape is 
                                  [out_channels, in_channels, kernel_h, kernel_w]. For Linear, it's [out_features, in_features].
                                </p>
                              </div>
                            </div>
                            {Object.entries(layer.parameters).map(([paramName, param]) => (
                              <div key={paramName} className="parameter-item">
                                <div className="parameter-header">
                                  <span className="param-name">{paramName}</span>
                                  <span className={`param-badge ${param.requires_grad ? 'trainable' : 'frozen'}`}>
                                    {param.requires_grad ? 'Trainable' : 'Frozen'}
                                  </span>
                                </div>
                                <div className="parameter-info">
                                  <div className="info-row">
                                    <span className="info-label">Shape:</span>
                                    <span className="info-value">{formatShape(param.shape)}</span>
                                  </div>
                                  <div className="info-row">
                                    <span className="info-label">Elements:</span>
                                    <span className="info-value">{formatNumber(param.numel)}</span>
                                  </div>
                                  <div className="info-row">
                                    <span className="info-label">Dtype:</span>
                                    <span className="info-value">{param.dtype}</span>
                                  </div>
                                  <div className="info-row">
                                    <span className="info-label">Range:</span>
                                    <span className="info-value">[{param.min.toFixed(4)}, {param.max.toFixed(4)}]</span>
                                    <span className="info-hint">Min and max values in this parameter tensor</span>
                                  </div>
                                  <div className="info-row">
                                    <span className="info-label">Mean ± Std:</span>
                                    <span className="info-value">{param.mean.toFixed(4)} ± {param.std.toFixed(4)}</span>
                                    <span className="info-hint">Average value and spread of weights</span>
                                  </div>
                                  <div className="parameter-explanation">
                                    <details>
                                      <summary>What do these statistics mean?</summary>
                                      <ul>
                                        <li><strong>Shape:</strong> The dimensions of the weight tensor. For Conv2d: [filters, channels, height, width]. For Linear: [outputs, inputs].</li>
                                        <li><strong>Elements:</strong> Total number of values stored (product of all shape dimensions). More elements = more memory needed.</li>
                                        <li><strong>Dtype:</strong> Data type (usually float32). Determines precision and memory usage.</li>
                                        <li><strong>Range:</strong> The smallest and largest weight values. Large ranges might indicate unstable training.</li>
                                        <li><strong>Mean ± Std:</strong> Average weight value and standard deviation. Near-zero mean with small std is often desirable.</li>
                                        <li><strong>Trainable vs Frozen:</strong> Trainable parameters are updated during training. Frozen parameters are fixed (used in transfer learning).</li>
                                      </ul>
                                    </details>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          {layer.buffers && Object.keys(layer.buffers).length > 0 && (
                            <div className="buffers-section">
                              <div className="section-header-with-explanation">
                                <h4>Buffers (Non-learnable):</h4>
                                <div className="section-explanation">
                                  <p>
                                    <strong>Buffers</strong> are non-learnable parameters that store statistics or state information. 
                                    They are updated during forward passes but not by gradient descent.
                                  </p>
                                  <ul>
                                    <li><strong>running_mean / running_var:</strong> Used in BatchNorm to track moving averages of mean and variance across batches</li>
                                    <li><strong>num_batches_tracked:</strong> Counter for how many batches have been processed (BatchNorm)</li>
                                  </ul>
                                  <p>
                                    Unlike weights, buffers don't require gradients and are typically updated using exponential moving averages.
                                  </p>
                                </div>
                              </div>
                              {Object.entries(layer.buffers).map(([bufferName, buffer]) => (
                                <div key={bufferName} className="buffer-item">
                                  <div className="parameter-header">
                                    <span className="param-name">{bufferName}</span>
                                    <span className="param-badge buffer">Buffer</span>
                                  </div>
                                  <div className="parameter-info">
                                    <div className="info-row">
                                      <span className="info-label">Shape:</span>
                                      <span className="info-value">{formatShape(buffer.shape)}</span>
                                    </div>
                                    <div className="info-row">
                                      <span className="info-label">Elements:</span>
                                      <span className="info-value">{formatNumber(buffer.numel)}</span>
                                    </div>
                                    <div className="info-row">
                                      <span className="info-label">Dtype:</span>
                                      <span className="info-value">{buffer.dtype}</span>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default WeightStructureModal

