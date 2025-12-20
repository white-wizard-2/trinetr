import { useState, useEffect } from 'react'
import axios from 'axios'
import ImageUploader from './ImageUploader'
import './WorldModelWorkspace.css'

interface WorldModelWorkspaceProps {
  modelId: string | null
  modelName: string
}

interface EncodeResult {
  latent: number[]
  mu: number[]
  logvar: number[]
  reconstruction: string | null
  latent_stats: {
    mean: number
    std: number
    min: number
    max: number
  }
}

interface PredictResult {
  predictions: Array<{
    step: number
    predicted_latent: number[]
    stats: {
      mean: number
      std: number
      min: number
      max: number
    }
  }>
  initial_latent: number[] | null
  action_used: number[]
}

interface ControlResult {
  action: number[]
  action_probabilities: number[]
  action_stats: {
    mean: number
    std: number
    min: number
    max: number
  }
}

interface SimulateStep {
  step: number
  observation: string
  latent: number[]
  action: number[]
  predicted_latent: number[]
  predicted_observation: string
}

interface SimulateResult {
  steps: SimulateStep[]
  total_steps: number
}

function WorldModelWorkspace({ modelId }: WorldModelWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<'explanation' | 'vision' | 'memory' | 'controller' | 'simulation'>('explanation')
  const [encodeResult, setEncodeResult] = useState<EncodeResult | null>(null)
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null)
  const [controlResult, setControlResult] = useState<ControlResult | null>(null)
  const [simulateResult, setSimulateResult] = useState<SimulateResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [selectedStep, setSelectedStep] = useState(0)
  const [predictSteps, setPredictSteps] = useState(5)
  const [simulateSteps, setSimulateSteps] = useState(10)
  const [architectureInfo, setArchitectureInfo] = useState<any>(null)
  const [loadingArchitecture, setLoadingArchitecture] = useState(false)

  // Load default image on mount
  useEffect(() => {
    const loadDefaultImage = async () => {
      try {
        const response = await fetch('/trinetr.png')
        const blob = await response.blob()
        const file = new File([blob], 'trinetr.png', { type: 'image/png' })
        setImageFile(file)
      } catch (error) {
        console.error('Failed to load default image:', error)
      }
    }
    loadDefaultImage()
  }, [])

  const encodeImage = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      let imageBase64 = null
      if (imageFile) {
        const reader = new FileReader()
        imageBase64 = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string)
          reader.onerror = reject
          reader.readAsDataURL(imageFile)
        })
      }

      const response = await axios.post(`http://localhost:8000/world-models/${modelId}/encode`, {
        image: imageBase64,
        return_reconstruction: true
      })
      setEncodeResult(response.data)
      setActiveTab('vision')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to encode image')
    } finally {
      setLoading(false)
    }
  }

  const predictState = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const latent = encodeResult?.latent || null
      const response = await axios.post(`http://localhost:8000/world-models/${modelId}/predict`, {
        latent,
        steps: predictSteps
      })
      setPredictResult(response.data)
      setActiveTab('memory')
      setSelectedStep(0)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to predict state')
    } finally {
      setLoading(false)
    }
  }

  const generateAction = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const latent = encodeResult?.latent || predictResult?.predictions[0]?.predicted_latent || null
      if (!latent) {
        setError('Please encode an image or predict a state first')
        return
      }

      const response = await axios.post(`http://localhost:8000/world-models/${modelId}/control`, {
        latent
      })
      setControlResult(response.data)
      setActiveTab('controller')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate action')
    } finally {
      setLoading(false)
    }
  }

  const runSimulation = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      let imageBase64 = null
      if (imageFile) {
        const reader = new FileReader()
        imageBase64 = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string)
          reader.onerror = reject
          reader.readAsDataURL(imageFile)
        })
      }

      const response = await axios.post(`http://localhost:8000/world-models/${modelId}/simulate`, {
        initial_image: imageBase64,
        steps: simulateSteps
      })
      setSimulateResult(response.data)
      setActiveTab('simulation')
      setSelectedStep(0)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run simulation')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="world-model-workspace">
      <div className="world-model-left-panel">
        <div className="data-flow-section">
          <h3>🌍 World Model Data Flow</h3>
          <div className="flow-steps">
            <div className="flow-step">
              <div className="flow-icon">👁️</div>
              <div className="flow-content">
                <strong>Observation</strong>
                <p>Raw image from environment</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🔍</div>
              <div className="flow-content">
                <strong>V: Vision Encoder</strong>
                <p>Compress to latent space</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🧠</div>
              <div className="flow-content">
                <strong>M: Memory (RNN)</strong>
                <p>Predict future latent state</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🎮</div>
              <div className="flow-content">
                <strong>C: Controller</strong>
                <p>Generate action from latent</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">⚡</div>
              <div className="flow-content">
                <strong>Action</strong>
                <p>Execute in environment</p>
              </div>
            </div>
          </div>
        </div>

        <div className="explanation-section">
          <h3>🧠 How World Models Work</h3>
          <div className="explanation-content">
            <div className="explanation-block">
              <h4>1. What are World Models?</h4>
              <p>World Models learn compressed representations of environments. Instead of learning directly from raw observations, agents learn to "dream" by imagining future states in a compact latent space.</p>
            </div>

            <div className="explanation-block">
              <h4>2. The Three Components</h4>
              <p><strong>V (Vision):</strong> Encodes observations (images) into a low-dimensional latent space. This compression makes it easier to learn and predict.</p>
              <p><strong>M (Memory):</strong> An RNN that learns temporal dynamics. Given current latent state and action, it predicts the next latent state.</p>
              <p><strong>C (Controller):</strong> A policy network that generates actions from latent states. It learns to act in the compressed space.</p>
            </div>

            <div className="explanation-block">
              <h4>3. Why World Models?</h4>
              <ul>
                <li><strong>Efficiency:</strong> Learning in latent space is faster than raw observations</li>
                <li><strong>Planning:</strong> Agents can "dream" multiple steps ahead without interacting with the environment</li>
                <li><strong>Generalization:</strong> Compressed representations capture essential features</li>
                <li><strong>Sample Efficiency:</strong> Model-based RL requires fewer environment interactions</li>
              </ul>
            </div>

            <div className="explanation-block">
              <h4>4. The "Dreaming" Process</h4>
              <p>Once trained, the agent can imagine future states by:</p>
              <ol>
                <li>Encoding current observation → latent z</li>
                <li>Predicting next latent z' using M</li>
                <li>Generating action using C</li>
                <li>Repeating to "dream" multiple steps ahead</li>
              </ol>
              <p>This allows the agent to plan and explore without costly environment interactions.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="world-model-right-panel">
        <div className="world-model-tabs">
          <button 
            className={`tab-btn ${activeTab === 'explanation' ? 'active' : ''}`}
            onClick={() => setActiveTab('explanation')}
          >
            📚 How It Works
          </button>
          <button 
            className={`tab-btn ${activeTab === 'vision' ? 'active' : ''}`}
            onClick={() => setActiveTab('vision')}
            disabled={!modelId}
          >
            👁️ Vision (V)
          </button>
          <button 
            className={`tab-btn ${activeTab === 'memory' ? 'active' : ''}`}
            onClick={() => setActiveTab('memory')}
            disabled={!modelId}
          >
            🧠 Memory (M)
          </button>
          <button 
            className={`tab-btn ${activeTab === 'controller' ? 'active' : ''}`}
            onClick={() => setActiveTab('controller')}
            disabled={!modelId}
          >
            🎮 Controller (C)
          </button>
          <button 
            className={`tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulation')}
            disabled={!modelId}
          >
            ⚡ Simulation
          </button>
        </div>

        <div className="tab-content-area">
          {activeTab === 'explanation' && (
            <div className="explanation-tab">
              <h2>🎓 Understanding World Models</h2>
              
              <div className="concept-section">
                <h3>The Core Idea</h3>
                <p>World Models enable agents to learn efficient representations of environments by separating perception (V), prediction (M), and control (C). This allows agents to "dream" and plan in a compressed latent space, making learning more sample-efficient.</p>
              </div>

              <div className="concept-section">
                <h3>Component Details</h3>
                <div className="component-list">
                  <div className="component-item">
                    <strong>👁️ V: Vision Encoder</strong>
                    <p>A Variational Autoencoder (VAE) that compresses high-dimensional observations (images) into low-dimensional latent vectors. It learns to capture essential features while discarding irrelevant details.</p>
                    <p><strong>Key Features:</strong> Encoder (observation → latent), Decoder (latent → reconstruction), learns compressed representation</p>
                  </div>
                  <div className="component-item">
                    <strong>🧠 M: Memory (RNN)</strong>
                    <p>A Recurrent Neural Network (LSTM) that learns temporal dynamics. Given current latent state and action, it predicts the next latent state. This enables the agent to imagine future states.</p>
                    <p><strong>Key Features:</strong> Learns state transitions, predicts future latents, enables multi-step "dreaming"</p>
                  </div>
                  <div className="component-item">
                    <strong>🎮 C: Controller</strong>
                    <p>A policy network (MLP) that generates actions from latent states. It learns to act optimally in the compressed space, making decisions based on the learned representation.</p>
                    <p><strong>Key Features:</strong> Maps latent → action, learns optimal policy, operates in compressed space</p>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>The Training Process</h3>
                <div className="step-list">
                  <div className="step-item">
                    <div className="step-number">1</div>
                    <div className="step-content">
                      <strong>Collect Data</strong>
                      <p>Agent interacts with environment, collecting observations and actions</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>Train V (Vision)</strong>
                      <p>VAE learns to encode/decode observations, creating compressed representations</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>Train M (Memory)</strong>
                      <p>RNN learns to predict next latent state given current state and action</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">4</div>
                    <div className="step-content">
                      <strong>Train C (Controller)</strong>
                      <p>Policy network learns to generate actions that maximize reward in latent space</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>Advantages of World Models</h3>
                <ul>
                  <li><strong>Sample Efficiency:</strong> Model-based RL requires fewer environment interactions</li>
                  <li><strong>Planning:</strong> Agents can imagine future states without acting</li>
                  <li><strong>Generalization:</strong> Compressed representations capture essential features</li>
                  <li><strong>Interpretability:</strong> Latent space can be visualized and understood</li>
                  <li><strong>Transfer Learning:</strong> Learned representations can transfer to new tasks</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'vision' && (
            <div className="vision-tab">
              <h2>👁️ Vision Component (V)</h2>
              
              <div className="input-section">
                <h3>Upload Observation</h3>
                <ImageUploader onImageUpload={setImageFile} imageFile={imageFile} />
                <button 
                  onClick={encodeImage}
                  disabled={loading || !modelId}
                  className="action-btn"
                >
                  {loading ? '🔄 Encoding...' : '🔍 Encode Image'}
                </button>
              </div>

              {encodeResult && (
                <div className="encode-results">
                  <div className="image-comparison">
                    <div className="image-box">
                      <h4>Original Observation</h4>
                      {imageFile && (
                        <img 
                          src={URL.createObjectURL(imageFile)} 
                          alt="Original" 
                          className="comparison-image"
                        />
                      )}
                    </div>
                    {encodeResult.reconstruction && (
                      <>
                        <div className="arrow">→</div>
                        <div className="image-box">
                          <h4>Reconstruction</h4>
                          <img 
                            src={`data:image/png;base64,${encodeResult.reconstruction}`}
                            alt="Reconstruction"
                            className="comparison-image"
                          />
                        </div>
                      </>
                    )}
                  </div>

                  <div className="latent-info">
                    <h3>Latent Representation</h3>
                    <div className="stats-grid">
                      <div className="stat-card">
                        <div className="stat-label">Latent Dimension</div>
                        <div className="stat-value">{encodeResult.latent.length}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Mean</div>
                        <div className="stat-value">{encodeResult.latent_stats.mean.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Std Dev</div>
                        <div className="stat-value">{encodeResult.latent_stats.std.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Min</div>
                        <div className="stat-value">{encodeResult.latent_stats.min.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Max</div>
                        <div className="stat-value">{encodeResult.latent_stats.max.toFixed(4)}</div>
                      </div>
                    </div>

                    <div className="latent-values">
                      <h4>Latent Vector Values (First 20 dimensions)</h4>
                      <div className="value-list">
                        {encodeResult.latent.slice(0, 20).map((val, idx) => (
                          <div 
                            key={idx} 
                            className={`value-item ${val >= 0 ? 'positive' : 'negative'}`}
                          >
                            <span className="value-dim">z{idx}</span>
                            <span className="value-num">{val.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="explanation-box">
                    <h3>What's Happening?</h3>
                    <p>The Vision encoder (V) compresses the high-dimensional observation (64×64×3 = 12,288 pixels) into a low-dimensional latent vector ({encodeResult.latent.length} dimensions). This compression:</p>
                    <ul>
                      <li>Captures essential features of the observation</li>
                      <li>Discards irrelevant details</li>
                      <li>Makes it easier to learn and predict</li>
                      <li>Enables efficient planning in latent space</li>
                    </ul>
                    <p>The decoder can reconstruct the observation from the latent vector, showing that essential information is preserved.</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'memory' && (
            <div className="memory-tab">
              <h2>🧠 Memory Component (M)</h2>
              
              <div className="input-section">
                <h3>Predict Future States</h3>
                <div className="param-group">
                  <label>
                    <strong>Steps to Predict:</strong>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={predictSteps}
                      onChange={(e) => setPredictSteps(Number(e.target.value))}
                      className="param-input"
                    />
                  </label>
                </div>
                <button 
                  onClick={predictState}
                  disabled={loading || !modelId}
                  className="action-btn"
                >
                  {loading ? '🔄 Predicting...' : '🔮 Predict Future States'}
                </button>
                <p className="help-text">Uses the encoded latent from Vision (V) or generates a random latent if none available.</p>
              </div>

              {predictResult && (
                <div className="predict-results">
                  <div className="step-selector">
                    <label>
                      <strong>View Step:</strong>
                      <select
                        value={selectedStep}
                        onChange={(e) => setSelectedStep(Number(e.target.value))}
                        className="step-select"
                      >
                        {predictResult.predictions.map((pred, idx) => (
                          <option key={idx} value={idx}>
                            Step {pred.step}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  {predictResult.predictions[selectedStep] && (
                    <div className="prediction-details">
                      <h3>Step {predictResult.predictions[selectedStep].step} Prediction</h3>
                      <div className="stats-grid">
                        <div className="stat-card">
                          <div className="stat-label">Mean</div>
                          <div className="stat-value">
                            {predictResult.predictions[selectedStep].stats.mean.toFixed(4)}
                          </div>
                        </div>
                        <div className="stat-card">
                          <div className="stat-label">Std Dev</div>
                          <div className="stat-value">
                            {predictResult.predictions[selectedStep].stats.std.toFixed(4)}
                          </div>
                        </div>
                        <div className="stat-card">
                          <div className="stat-label">Min</div>
                          <div className="stat-value">
                            {predictResult.predictions[selectedStep].stats.min.toFixed(4)}
                          </div>
                        </div>
                        <div className="stat-card">
                          <div className="stat-label">Max</div>
                          <div className="stat-value">
                            {predictResult.predictions[selectedStep].stats.max.toFixed(4)}
                          </div>
                        </div>
                      </div>

                      <div className="latent-values">
                        <h4>Predicted Latent Vector (First 20 dimensions)</h4>
                        <div className="value-list">
                          {predictResult.predictions[selectedStep].predicted_latent.slice(0, 20).map((val, idx) => (
                            <div 
                              key={idx} 
                              className={`value-item ${val >= 0 ? 'positive' : 'negative'}`}
                            >
                              <span className="value-dim">z{idx}</span>
                              <span className="value-num">{val.toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="explanation-box">
                    <h3>What's Happening?</h3>
                    <p>The Memory component (M) is an RNN that learns temporal dynamics. Given the current latent state and an action, it predicts the next latent state.</p>
                    <p>This enables the agent to "dream" - imagine future states without interacting with the environment. The RNN maintains hidden state across steps, allowing it to learn complex temporal patterns.</p>
                    <p><strong>Key Insight:</strong> By predicting in latent space (not raw observations), the model can efficiently learn and plan.</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'controller' && (
            <div className="controller-tab">
              <h2>🎮 Controller Component (C)</h2>
              
              <div className="input-section">
                <button 
                  onClick={generateAction}
                  disabled={loading || !modelId}
                  className="action-btn"
                >
                  {loading ? '🔄 Generating...' : '🎯 Generate Action'}
                </button>
                <p className="help-text">Uses the current latent state (from Vision or Memory prediction) to generate an action.</p>
              </div>

              {controlResult && (
                <div className="control-results">
                  <div className="action-display">
                    <h3>Generated Action</h3>
                    <div className="action-values">
                      {controlResult.action.map((val, idx) => (
                        <div key={idx} className="action-item">
                          <div className="action-label">Action {idx + 1}</div>
                          <div className="action-value">{val.toFixed(4)}</div>
                          <div className="action-prob">
                            {(controlResult.action_probabilities[idx] * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="stats-section">
                    <h3>Action Statistics</h3>
                    <div className="stats-grid">
                      <div className="stat-card">
                        <div className="stat-label">Mean</div>
                        <div className="stat-value">{controlResult.action_stats.mean.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Std Dev</div>
                        <div className="stat-value">{controlResult.action_stats.std.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Min</div>
                        <div className="stat-value">{controlResult.action_stats.min.toFixed(4)}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Max</div>
                        <div className="stat-value">{controlResult.action_stats.max.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>

                  <div className="explanation-box">
                    <h3>What's Happening?</h3>
                    <p>The Controller (C) is a policy network that generates actions from latent states. It learns to map compressed representations to optimal actions.</p>
                    <p>By operating in latent space (not raw observations), the controller can:</p>
                    <ul>
                      <li>Make decisions based on essential features</li>
                      <li>Learn more efficiently</li>
                      <li>Generalize better to new situations</li>
                      <li>Plan ahead using predicted latent states</li>
                    </ul>
                    <p><strong>Key Insight:</strong> The controller learns to act optimally in the compressed space, making it more sample-efficient than learning directly from raw observations.</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'simulation' && (
            <div className="simulation-tab">
              <h2>⚡ Full Simulation</h2>
              
              <div className="input-section">
                <h3>Run World Model Simulation</h3>
                <div className="param-group">
                  <label>
                    <strong>Simulation Steps:</strong>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={simulateSteps}
                      onChange={(e) => setSimulateSteps(Number(e.target.value))}
                      className="param-input"
                    />
                  </label>
                </div>
                <ImageUploader onImageUpload={setImageFile} imageFile={imageFile} />
                <button 
                  onClick={runSimulation}
                  disabled={loading || !modelId}
                  className="action-btn"
                >
                  {loading ? '🔄 Simulating...' : '🚀 Run Simulation'}
                </button>
                <p className="help-text">Runs the full pipeline: Observation → V → Latent → M → Predicted Latent → C → Action (repeated)</p>
              </div>

              {simulateResult && (
                <div className="simulation-results">
                  <div className="step-selector">
                    <label>
                      <strong>View Step:</strong>
                      <select
                        value={selectedStep}
                        onChange={(e) => setSelectedStep(Number(e.target.value))}
                        className="step-select"
                      >
                        {simulateResult.steps.map((step, idx) => (
                          <option key={idx} value={idx}>
                            Step {step.step}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  {simulateResult.steps[selectedStep] && (
                    <div className="simulation-step">
                      <h3>Step {simulateResult.steps[selectedStep].step}</h3>
                      
                      <div className="step-flow-diagram">
                        <div className="flow-step-box">
                          <div className="flow-label">👁️ Observation Input</div>
                          <div className="step-box">
                            <h4>
                              {selectedStep === 0 ? 'Initial Observation' : `Observation (from Step ${selectedStep}'s prediction)`}
                            </h4>
                            <img 
                              src={`data:image/png;base64,${simulateResult.steps[selectedStep].observation}`}
                              alt="Observation"
                              className="step-image"
                            />
                            {selectedStep === 0 ? (
                              <div className="change-indicator">
                                <small>📸 Real observation (from image or random)</small>
                              </div>
                            ) : (
                              <div className="change-indicator">
                                <small>🔮 This is the predicted observation from Step {selectedStep}</small>
                                <small style={{display: 'block', marginTop: '0.25rem'}}>
                                  (The model is "dreaming" - using its own predictions as input)
                                </small>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        <div className="flow-arrow-large">↓</div>
                        <div className="flow-step-box">
                          <div className="flow-label">🔍 V: Vision Encoder</div>
                          <div className="latent-preview">
                            <div className="latent-stats-mini">
                              <div className="stat-mini">
                                <span className="stat-label-mini">Latent Mean:</span>
                                <span className="stat-value-mini">
                                  {simulateResult.steps[selectedStep].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].latent.length > 0 ? '🟢' : '🔴'}
                                  {(simulateResult.steps[selectedStep].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].latent.length).toFixed(3)}
                                </span>
                              </div>
                              <div className="stat-mini">
                                <span className="stat-label-mini">Latent Std:</span>
                                <span className="stat-value-mini">
                                  {(() => {
                                    const mean = simulateResult.steps[selectedStep].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].latent.length;
                                    const variance = simulateResult.steps[selectedStep].latent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / simulateResult.steps[selectedStep].latent.length;
                                    return Math.sqrt(variance).toFixed(3);
                                  })()}
                                </span>
                              </div>
                            </div>
                            <div className="latent-sample-values">
                              {simulateResult.steps[selectedStep].latent.slice(0, 8).map((val, idx) => (
                                <span key={idx} className={`latent-value-mini ${val >= 0 ? 'positive' : 'negative'}`}>
                                  {val.toFixed(2)}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flow-arrow-large">↓</div>
                        <div className="flow-step-box">
                          <div className="flow-label">🎮 C: Controller</div>
                          <div className="action-display-mini">
                            {simulateResult.steps[selectedStep].action.map((val, idx) => (
                              <div key={idx} className="action-bar">
                                <div className="action-label-mini">A{idx + 1}</div>
                                <div className="action-bar-container">
                                  <div 
                                    className={`action-bar-fill ${val >= 0 ? 'positive' : 'negative'}`}
                                    style={{ width: `${Math.abs(val) * 100}%` }}
                                  />
                                </div>
                                <div className="action-value-mini">{val.toFixed(3)}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div className="flow-arrow-large">↓</div>
                        <div className="flow-step-box">
                          <div className="flow-label">🧠 M: Memory (RNN)</div>
                          <div className="latent-preview">
                            <div className="latent-stats-mini">
                              <div className="stat-mini">
                                <span className="stat-label-mini">Predicted Mean:</span>
                                <span className="stat-value-mini">
                                  {simulateResult.steps[selectedStep].predicted_latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].predicted_latent.length > 0 ? '🟢' : '🔴'}
                                  {(simulateResult.steps[selectedStep].predicted_latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].predicted_latent.length).toFixed(3)}
                                </span>
                              </div>
                              <div className="stat-mini">
                                <span className="stat-label-mini">Change:</span>
                                <span className="stat-value-mini">
                                  {(() => {
                                    const currentMean = simulateResult.steps[selectedStep].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].latent.length;
                                    const predictedMean = simulateResult.steps[selectedStep].predicted_latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].predicted_latent.length;
                                    const change = predictedMean - currentMean;
                                    return change >= 0 ? `+${change.toFixed(3)}` : change.toFixed(3);
                                  })()}
                                </span>
                              </div>
                            </div>
                            <div className="latent-sample-values">
                              {simulateResult.steps[selectedStep].predicted_latent.slice(0, 8).map((val, idx) => (
                                <span key={idx} className={`latent-value-mini ${val >= 0 ? 'positive' : 'negative'}`}>
                                  {val.toFixed(2)}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flow-arrow-large">↓</div>
                        <div className="flow-step-box">
                          <div className="flow-label">🖼️ Decoded Prediction</div>
                          <div className="step-box">
                            <h4>Predicted Observation (Step {simulateResult.steps[selectedStep].step + 1} Input)</h4>
                            <img 
                              src={`data:image/png;base64,${simulateResult.steps[selectedStep].predicted_observation}`}
                              alt="Predicted"
                              className="step-image"
                            />
                            {selectedStep < simulateResult.steps.length - 1 ? (
                              <div className="change-indicator">
                                <small>➡️ This becomes Step {simulateResult.steps[selectedStep].step + 1}'s input</small>
                                <small style={{display: 'block', marginTop: '0.25rem'}}>
                                  The model uses its own prediction as the next observation
                                </small>
                              </div>
                            ) : (
                              <div className="change-indicator">
                                <small>🏁 Final prediction (no next step)</small>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {selectedStep > 0 && (
                        <div className="step-comparison">
                          <h4>📊 Changes from Previous Step</h4>
                          <div className="comparison-stats">
                            <div className="comparison-item">
                              <strong>Action Change:</strong>
                              <span>
                                {(() => {
                                  const prev = simulateResult.steps[selectedStep - 1].action;
                                  const curr = simulateResult.steps[selectedStep].action;
                                  const changes = curr.map((val, idx) => (val - prev[idx]).toFixed(3));
                                  return `[${changes.join(', ')}]`;
                                })()}
                              </span>
                            </div>
                            <div className="comparison-item">
                              <strong>Latent Change:</strong>
                              <span>
                                {(() => {
                                  const prevMean = simulateResult.steps[selectedStep - 1].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep - 1].latent.length;
                                  const currMean = simulateResult.steps[selectedStep].latent.reduce((a, b) => a + b, 0) / simulateResult.steps[selectedStep].latent.length;
                                  const change = currMean - prevMean;
                                  return change >= 0 ? `+${change.toFixed(4)}` : change.toFixed(4);
                                })()}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="step-info-detailed">
                        <h4>📋 Step Details</h4>
                        <div className="info-grid-detailed">
                          <div className="info-item-detailed">
                            <strong>Action Vector:</strong>
                            <div className="action-vector">
                              {simulateResult.steps[selectedStep].action.map((val, idx) => (
                                <span key={idx} className={`action-tag ${val >= 0 ? 'positive' : 'negative'}`}>
                                  A{idx + 1}: {val.toFixed(4)}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="info-item-detailed">
                            <strong>Latent Dimensions:</strong> {simulateResult.steps[selectedStep].latent.length}
                          </div>
                          <div className="info-item-detailed">
                            <strong>Latent Range:</strong>
                            [{Math.min(...simulateResult.steps[selectedStep].latent).toFixed(3)}, {Math.max(...simulateResult.steps[selectedStep].latent).toFixed(3)}]
                          </div>
                          <div className="info-item-detailed">
                            <strong>Predicted Latent Range:</strong>
                            [{Math.min(...simulateResult.steps[selectedStep].predicted_latent).toFixed(3)}, {Math.max(...simulateResult.steps[selectedStep].predicted_latent).toFixed(3)}]
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="explanation-box">
                    <h3>The "Dreaming" Process</h3>
                    <p>This simulation demonstrates how the agent can "dream" - imagine future states without interacting with the environment:</p>
                    <div className="process-flow-explanation">
                      <div className="process-step-explained">
                        <strong>Step 1:</strong>
                        <ul>
                          <li>Starts with <strong>initial observation</strong> (your uploaded image or random)</li>
                          <li>V encodes it → latent z₁</li>
                          <li>C generates action a₁</li>
                          <li>M predicts next latent z₂</li>
                          <li>Decoder produces <strong>predicted observation</strong> (this is the "dream")</li>
                        </ul>
                      </div>
                      <div className="process-step-explained">
                        <strong>Step 2:</strong>
                        <ul>
                          <li>Uses <strong>predicted observation from Step 1</strong> as input (not the original!)</li>
                          <li>V encodes the predicted observation → latent z₂</li>
                          <li>C generates action a₂</li>
                          <li>M predicts next latent z₃</li>
                          <li>Decoder produces another predicted observation</li>
                        </ul>
                      </div>
                      <div className="process-step-explained">
                        <strong>Step 3+:</strong>
                        <ul>
                          <li>Each step uses the <strong>previous step's prediction</strong> as input</li>
                          <li>This is the "dreaming" - the agent imagines multiple steps into the future</li>
                          <li>All without interacting with the real environment!</li>
                        </ul>
                      </div>
                    </div>
                    <p><strong>Key Insight:</strong> The model uses its own predictions as inputs for subsequent steps. This allows it to "dream" multiple steps ahead, exploring different futures in its imagination.</p>
                    <div className="info-note">
                      <strong>Note:</strong> This is a demonstration model with random initialization (not trained). 
                      The images show how the model processes data, but won't produce meaningful reconstructions until trained on real environment data.
                      In a trained model, the V component would learn to compress observations while preserving essential features, making the "dreams" more realistic.
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default WorldModelWorkspace

