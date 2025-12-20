import { useState } from 'react'
import axios from 'axios'
import './StateSpaceWorkspace.css'

interface StateSpaceWorkspaceProps {
  modelId: string | null
  modelName: string
}

interface HiddenState {
  layer: number
  shape: number[]
  mean: number
  std: number
  min: number
  max: number
  sample: number[]
}

interface StateSpaceOutput {
  input_tokens: string[]
  hidden_states: HiddenState[]
  top_predictions: { token: string; probability: number; logit: number }[]
  architecture_info: {
    d_model: number
    n_layers: number
    vocab_size: number
    model_type: string
  }
  state_space_info: {
    state_dim: number
    sequence_length: number
    efficient_attention: boolean
    linear_complexity: boolean
  }
  logits_stats: {
    min: number
    max: number
    mean: number
    std: number
  }
}

function StateSpaceWorkspace({ modelId }: StateSpaceWorkspaceProps) {
  const [inputText, setInputText] = useState('State space models are a new architecture')
  const [output, setOutput] = useState<StateSpaceOutput | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'explanation' | 'inference' | 'states' | 'architecture'>('explanation')
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [architectureInfo, setArchitectureInfo] = useState<any>(null)
  const [loadingArchitecture, setLoadingArchitecture] = useState(false)

  const runInference = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`http://localhost:8000/state-space/${modelId}/infer`, {
        text: inputText,
        max_length: 100,
        return_states: true
      })
      setOutput(response.data)
      setActiveTab('inference')
      setSelectedLayer(0)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run inference')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="state-space-workspace">
      <div className="state-space-left-panel">
        <div className="data-flow-section">
          <h3>🔄 State Space Model Data Flow</h3>
          <div className="flow-steps">
            <div className="flow-step">
              <div className="flow-icon">📝</div>
              <div className="flow-content">
                <strong>Text Input</strong>
                <p>Your input text sequence</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🔤</div>
              <div className="flow-content">
                <strong>Tokenization</strong>
                <p>Convert text to token IDs</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🎯</div>
              <div className="flow-content">
                <strong>Embedding</strong>
                <p>Convert tokens to dense vectors</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">⚡</div>
              <div className="flow-content">
                <strong>State Space Blocks</strong>
                <p>Process with linear complexity</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">📊</div>
              <div className="flow-content">
                <strong>Hidden States</strong>
                <p>Contextualized representations</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🎲</div>
              <div className="flow-content">
                <strong>Output Logits</strong>
                <p>Predictions for next token</p>
              </div>
            </div>
          </div>
        </div>

        <div className="explanation-section">
          <h3>🧠 How State Space Models Work</h3>
          <div className="explanation-content">
            <div className="explanation-block">
              <h4>1. What is a State Space Model?</h4>
              <p>State space models use continuous-time differential equations to process sequences. They maintain an internal "state" that evolves over time, allowing efficient processing of long sequences.</p>
              <div className="formula-box">
                <code>dx/dt = Ax + Bu</code>
                <code>y = Cx + Du</code>
                <p className="formula-explanation">State space equations: x is the state, u is input, y is output. A, B, C, D are learned matrices.</p>
              </div>
            </div>

            <div className="explanation-block">
              <h4>2. Linear vs Quadratic Complexity</h4>
              <p>Transformers use attention which is O(n²) - every token attends to every other token. State space models are O(n) - they process tokens sequentially with constant memory per token.</p>
              <div className="comparison-box">
                <div className="comparison-item">
                  <strong>Transformers:</strong> O(n²) - 1000 tokens = 1M operations
                </div>
                <div className="comparison-item">
                  <strong>State Space:</strong> O(n) - 1000 tokens = 1000 operations
                </div>
              </div>
            </div>

            <div className="explanation-block">
              <h4>3. Selective State Spaces (Mamba)</h4>
              <p>Mamba's key innovation: the model learns which information to remember or forget based on the input. This "selective" mechanism allows it to handle long-range dependencies efficiently.</p>
              <div className="formula-box">
                <code>s&#123;t&#125; = A · s&#123;t-1&#125; + B · x&#123;t&#125;</code>
                <code>y&#123;t&#125; = C · s&#123;t&#125;</code>
                <p className="formula-explanation">At each step, the state s is updated based on previous state and current input. The output y is computed from the state.</p>
              </div>
            </div>

            <div className="explanation-block">
              <h4>4. Why State Space Models?</h4>
              <ul>
                <li><strong>Efficiency:</strong> Linear complexity means they can handle much longer sequences</li>
                <li><strong>Memory:</strong> Constant memory per token, not quadratic</li>
                <li><strong>Speed:</strong> Faster inference for long sequences</li>
                <li><strong>Quality:</strong> Competitive with transformers on many tasks</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="state-space-right-panel">
        <div className="state-space-tabs">
          <button 
            className={`tab-btn ${activeTab === 'explanation' ? 'active' : ''}`}
            onClick={() => setActiveTab('explanation')}
          >
            📚 How It Works
          </button>
          <button 
            className={`tab-btn ${activeTab === 'inference' ? 'active' : ''}`}
            onClick={() => setActiveTab('inference')}
            disabled={!output}
          >
            🔍 Inference
          </button>
          <button 
            className={`tab-btn ${activeTab === 'states' ? 'active' : ''}`}
            onClick={() => setActiveTab('states')}
            disabled={!output}
          >
            📊 Hidden States
          </button>
          <button 
            className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
            disabled={!modelId}
          >
            🏗️ Architecture
          </button>
        </div>

        <div className="tab-content-area">
          {activeTab === 'explanation' && (
            <div className="explanation-tab">
              <h2>🎓 Understanding State Space Models</h2>
              
              <div className="concept-section">
                <h3>The Core Idea</h3>
                <p>State space models process sequences by maintaining an evolving internal state. Unlike transformers that compute all pairwise relationships, state space models process tokens sequentially, updating their state at each step.</p>
              </div>

              <div className="concept-section">
                <h3>Key Components</h3>
                <div className="component-list">
                  <div className="component-item">
                    <strong>🔄 State Space Block</strong>
                    <p>The core building block. Contains the state space equations (A, B, C, D matrices) that process sequences.</p>
                  </div>
                  <div className="component-item">
                    <strong>⚡ Selective Mechanism</strong>
                    <p>Mamba's innovation: learns which information to keep or discard based on input. This makes the model context-aware.</p>
                  </div>
                  <div className="component-item">
                    <strong>📐 State Dimension</strong>
                    <p>The size of the internal state vector (e.g., 768, 1024). Larger states can remember more information.</p>
                  </div>
                  <div className="component-item">
                    <strong>🔗 Residual Connections</strong>
                    <p>Like transformers, state space blocks use residual connections to help with training and gradient flow.</p>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>The Processing Flow (Step-by-Step)</h3>
                <div className="step-list">
                  <div className="step-item">
                    <div className="step-number">1</div>
                    <div className="step-content">
                      <strong>Input Token</strong>
                      <p>Each token in the sequence is embedded into a dense vector representation.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>State Update</strong>
                      <p>The state space block updates its internal state: s&#123;t&#125; = A·s&#123;t-1&#125; + B·x&#123;t&#125;</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>Output Generation</strong>
                      <p>Output is computed from the state: y&#123;t&#125; = C·s&#123;t&#125;</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">4</div>
                    <div className="step-content">
                      <strong>Selective Filtering</strong>
                      <p>Mamba selectively decides what to remember based on the input, making it context-aware.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">5</div>
                    <div className="step-content">
                      <strong>Layer Stacking</strong>
                      <p>Multiple state space blocks are stacked to build deep representations.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">6</div>
                    <div className="step-content">
                      <strong>Final Prediction</strong>
                      <p>The final hidden state is used to predict the next token or classification.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>State Space vs Transformers</h3>
                <div className="comparison-table">
                  <div className="comparison-row">
                    <div className="comparison-cell"><strong>Feature</strong></div>
                    <div className="comparison-cell"><strong>Transformers</strong></div>
                    <div className="comparison-cell"><strong>State Space</strong></div>
                  </div>
                  <div className="comparison-row">
                    <div className="comparison-cell">Complexity</div>
                    <div className="comparison-cell">O(n²)</div>
                    <div className="comparison-cell">O(n)</div>
                  </div>
                  <div className="comparison-row">
                    <div className="comparison-cell">Memory</div>
                    <div className="comparison-cell">O(n²)</div>
                    <div className="comparison-cell">O(n)</div>
                  </div>
                  <div className="comparison-row">
                    <div className="comparison-cell">Long Sequences</div>
                    <div className="comparison-cell">Expensive</div>
                    <div className="comparison-cell">Efficient</div>
                  </div>
                  <div className="comparison-row">
                    <div className="comparison-cell">Attention</div>
                    <div className="comparison-cell">Full attention</div>
                    <div className="comparison-cell">Selective state</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'inference' && output && (
            <div className="inference-tab">
              <h2>🔍 Inference Results</h2>
              
              <div className="input-section">
                <h3>Input Text</h3>
                <div className="token-list">
                  {output.input_tokens.map((token, idx) => (
                    <span key={idx} className="token-badge">{token}</span>
                  ))}
                </div>
              </div>

              <div className="predictions-section">
                <h3>🎯 Top Predictions (Next Token)</h3>
                <div className="predictions-list">
                  {output.top_predictions.map((pred, idx) => (
                    <div key={idx} className="prediction-item">
                      <div className="prediction-rank">{idx + 1}</div>
                      <div className="prediction-content">
                        <div className="prediction-token">{pred.token}</div>
                        <div className="prediction-prob">{(pred.probability * 100).toFixed(2)}%</div>
                      </div>
                      <div className="prediction-bar">
                        <div 
                          className="prediction-bar-fill"
                          style={{ width: `${pred.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="stats-section">
                <h3>📊 Logits Statistics</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Min</div>
                    <div className="stat-value">{output.logits_stats.min.toFixed(2)}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Max</div>
                    <div className="stat-value">{output.logits_stats.max.toFixed(2)}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Mean</div>
                    <div className="stat-value">{output.logits_stats.mean.toFixed(2)}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Std Dev</div>
                    <div className="stat-value">{output.logits_stats.std.toFixed(2)}</div>
                  </div>
                </div>
              </div>

              <div className="info-section">
                <h3>ℹ️ Model Information</h3>
                <div className="info-grid">
                  <div className="info-item">
                    <strong>State Dimension:</strong> {output.state_space_info.state_dim}
                  </div>
                  <div className="info-item">
                    <strong>Sequence Length:</strong> {output.state_space_info.sequence_length}
                  </div>
                  <div className="info-item">
                    <strong>Linear Complexity:</strong> {output.state_space_info.linear_complexity ? 'Yes' : 'No'}
                  </div>
                  <div className="info-item">
                    <strong>Vocabulary Size:</strong> {output.architecture_info.vocab_size.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'states' && output && (
            <div className="states-tab">
              <h2>📊 Hidden States</h2>
              
              <div className="layer-selector">
                <label>
                  <strong>Select Layer:</strong>
                  <select
                    value={selectedLayer}
                    onChange={(e) => setSelectedLayer(Number(e.target.value))}
                    className="layer-select"
                  >
                    {output.hidden_states.map((state, idx) => (
                      <option key={idx} value={idx}>
                        Layer {state.layer}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {output.hidden_states[selectedLayer] && (
                <div className="state-details">
                  <div className="state-stats">
                    <h3>Layer {output.hidden_states[selectedLayer].layer} Statistics</h3>
                    <div className="stats-grid">
                      <div className="stat-card">
                        <div className="stat-label">Shape</div>
                        <div className="stat-value">
                          {output.hidden_states[selectedLayer].shape.join(' × ')}
                        </div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Mean</div>
                        <div className="stat-value">
                          {output.hidden_states[selectedLayer].mean.toFixed(4)}
                        </div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Std Dev</div>
                        <div className="stat-value">
                          {output.hidden_states[selectedLayer].std.toFixed(4)}
                        </div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Min</div>
                        <div className="stat-value">
                          {output.hidden_states[selectedLayer].min.toFixed(4)}
                        </div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Max</div>
                        <div className="stat-value">
                          {output.hidden_states[selectedLayer].max.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="state-sample">
                    <h3>Sample Values (First 10 Dimensions)</h3>
                    <div className="value-list">
                      {output.hidden_states[selectedLayer].sample.map((val, idx) => (
                        <div 
                          key={idx} 
                          className={`value-item ${val >= 0 ? 'positive' : 'negative'}`}
                        >
                          <span className="value-dim">d{idx}</span>
                          <span className="value-num">{val.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="state-explanation">
                    <h3>What This Layer Represents</h3>
                    <p>
                      This hidden state represents the model's internal representation after processing the input through {output.hidden_states[selectedLayer].layer} state space blocks.
                      Each dimension captures different aspects of the input sequence.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'architecture' && modelId && (
            <div className="architecture-tab">
              <h2>🏗️ Model Architecture</h2>
              {!architectureInfo && !loadingArchitecture && (
                <button 
                  onClick={async () => {
                    setLoadingArchitecture(true)
                    try {
                      const response = await axios.get(`http://localhost:8000/state-space/${modelId}/architecture`)
                      setArchitectureInfo(response.data)
                    } catch (err: any) {
                      setError(err.response?.data?.detail || 'Failed to load architecture')
                    } finally {
                      setLoadingArchitecture(false)
                    }
                  }}
                  className="load-architecture-btn"
                >
                  Load Architecture Details
                </button>
              )}
              {loadingArchitecture && <p>Loading architecture details...</p>}
              {architectureInfo && (
                <div className="architecture-details">
                  <div className="component-section">
                    <h3>🔄 State Space Block</h3>
                    <p className="component-description">
                      The core component that processes sequences using state space equations.
                      Each block maintains an internal state that evolves as it processes tokens.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>State Dimension:</strong> {architectureInfo.components.state_space_block.d_model}
                      </div>
                      <div className="spec-item">
                        <strong>Number of Layers:</strong> {architectureInfo.components.state_space_block.n_layers}
                      </div>
                    </div>
                  </div>

                  <div className="component-section">
                    <h3>⚡ State Space Model (SSM)</h3>
                    <p className="component-description">
                      The mathematical framework that enables linear-time processing.
                      Uses continuous-time differential equations discretized for sequence processing.
                    </p>
                  </div>

                  <div className="component-section">
                    <h3>🔤 Tokenizer</h3>
                    <p className="component-description">
                      Converts text into token IDs that the model can process.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>Vocabulary Size:</strong> {architectureInfo.components.tokenizer.vocab_size.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div className="total-params">
                    <strong>Total Parameters:</strong> {(architectureInfo.total_params / 1e6).toFixed(1)}M
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab !== 'explanation' && !output && (
            <div className="empty-state">
              <h3>🔍 Run Inference First</h3>
              <p>Load a model and run inference to see results here.</p>
            </div>
          )}
        </div>

        {activeTab !== 'explanation' && (
          <div className="inference-controls">
            <div className="input-group">
              <label>
                <strong>Input Text:</strong>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter text to process..."
                  rows={3}
                  className="text-input"
                />
              </label>
            </div>
            <button 
              onClick={runInference}
              disabled={loading || !modelId}
              className="inference-btn"
            >
              {loading ? '🔄 Processing...' : '🚀 Run Inference'}
            </button>
            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default StateSpaceWorkspace

