import { useState } from 'react'
import axios from 'axios'
import ImageUploader from './ImageUploader'
import TransformerArchitecture from './TransformerArchitecture'
import './TransformerWorkspace.css'

interface TransformerWorkspaceProps {
  modelId: string | null
  modelName: string
  transformerType: 'text' | 'image'
  imageFile: File | null
  onImageUpload: (file: File) => void
}

interface AttentionData {
  layer: number
  head: number
  attention_weights: number[][]
  tokens: string[]
}

interface GenerationStep {
  step: number
  input_so_far: string
  top_predictions: { token: string; probability: number }[]
  selected_token: string
  selected_token_id: number
  logits_stats: { min: number; max: number; mean: number }
}

interface TransformerOutput {
  input_tokens?: string[]
  output_tokens?: string[]
  attention_layers?: AttentionData[]
  predictions?: { label: string; probability: number }[]
  embeddings?: number[][]
  hidden_states?: number[][][]
  is_decoder?: boolean
  generation_steps?: GenerationStep[]
  generated_text?: string
  full_text?: string
}

function TransformerWorkspace({ modelId, modelName, transformerType, imageFile, onImageUpload }: TransformerWorkspaceProps) {
  const [inputText, setInputText] = useState('The transformer architecture revolutionized')
  const [output, setOutput] = useState<TransformerOutput | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [activeTab, setActiveTab] = useState<'input' | 'attention' | 'embeddings' | 'generation' | 'output'>('input')
  const [showExplanation, setShowExplanation] = useState(true)
  const [generateTokens, setGenerateTokens] = useState(5)
  const [selectedStep, setSelectedStep] = useState(0)
  const [selectedCell, setSelectedCell] = useState<{row: number, col: number, from: string, to: string, weight: number} | null>(null)

  const isDecoderModel = modelName.includes('gpt')

  const runInference = async (shouldGenerate: boolean = false) => {
    if (!modelId) return
    
    setLoading(true)
    setError(null)
    
    try {
      if (transformerType === 'text') {
        const response = await axios.post(`http://localhost:8000/transformers/${modelId}/infer`, {
          text: inputText,
          generate_tokens: shouldGenerate && isDecoderModel ? generateTokens : 0
        })
        setOutput(response.data)
        setSelectedStep(0)
        if (shouldGenerate && response.data.generation_steps?.length > 0) {
          setActiveTab('generation')
        } else {
          setActiveTab('attention')
        }
      } else {
        if (!imageFile) {
          setError('Please upload an image')
          return
        }
        const formData = new FormData()
        formData.append('file', imageFile)
        
        const response = await axios.post(
          `http://localhost:8000/transformers/${modelId}/infer-image`,
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        setOutput(response.data)
        setActiveTab('output')
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  const renderAttentionMatrix = () => {
    if (!output?.attention_layers?.length) return null
    
    const layerData = output.attention_layers.find(
      l => l.layer === selectedLayer && l.head === selectedHead
    )
    if (!layerData) return null

    const tokens = layerData.tokens.slice(0, 15) // Limit for display
    const weights = layerData.attention_weights.slice(0, 15).map(row => row.slice(0, 15))
    const maxWeight = Math.max(...weights.flat())

    const handleCellClick = (i: number, j: number) => {
      setSelectedCell({
        row: i,
        col: j,
        from: tokens[i],
        to: tokens[j],
        weight: weights[i][j]
      })
    }

    return (
      <div className="attention-matrix-container">
        {selectedCell && (
          <div className="selected-cell-panel">
            <button className="close-panel" onClick={() => setSelectedCell(null)}>✕</button>
            <h5>🔍 Attention Detail</h5>
            <div className="cell-detail">
              <div className="cell-relationship">
                <span className="from-token">"{selectedCell.from}"</span>
                <span className="arrow">→ attends to →</span>
                <span className="to-token">"{selectedCell.to}"</span>
              </div>
              <div className="cell-weight">
                <div className="weight-bar-bg">
                  <div className="weight-bar-fill" style={{ width: `${selectedCell.weight * 100}%` }} />
                </div>
                <span className="weight-value">{(selectedCell.weight * 100).toFixed(2)}%</span>
              </div>
              <div className="cell-interpretation">
                <strong>What this means:</strong>
                <p>
                  {selectedCell.weight > 0.3 
                    ? `"${selectedCell.from}" is strongly influenced by "${selectedCell.to}" when computing its output representation.`
                    : selectedCell.weight > 0.1
                    ? `"${selectedCell.from}" pays moderate attention to "${selectedCell.to}".`
                    : `"${selectedCell.from}" pays minimal attention to "${selectedCell.to}" - mostly ignoring it.`
                  }
                </p>
              </div>
            </div>
          </div>
        )}
        <div className="attention-matrix">
          <div className="matrix-header">
            <span className="corner-cell">From↓ To→</span>
            {tokens.map((t, i) => (
              <span key={i} className={`token-label ${selectedCell?.col === i ? 'highlighted' : ''}`} title={t}>{t}</span>
            ))}
          </div>
          {weights.map((row, i) => (
            <div key={i} className="matrix-row">
              <span className={`token-label ${selectedCell?.row === i ? 'highlighted' : ''}`} title={tokens[i]}>{tokens[i]}</span>
              {row.map((weight, j) => (
                <div
                  key={j}
                  className={`attention-cell ${selectedCell?.row === i && selectedCell?.col === j ? 'selected' : ''} ${weight > 0.2 ? 'high-weight' : ''}`}
                  style={{
                    backgroundColor: `rgba(67, 233, 123, ${weight / maxWeight})`,
                  }}
                  title={`Click for details: ${tokens[i]} → ${tokens[j]}: ${(weight * 100).toFixed(1)}%`}
                  onClick={() => handleCellClick(i, j)}
                >
                  <span className="cell-value">{(weight * 100).toFixed(0)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderTokenEmbeddings = () => {
    if (!output?.embeddings) return null

    return (
      <div className="embeddings-container">
        <div className="embedding-rows">
          {output.input_tokens?.slice(0, 20).map((token, i) => (
            <div key={i} className="embedding-row">
              <span className="token-name" title={token}>{token}</span>
              <div className="embedding-values">
                {output.embeddings![i]?.slice(0, 64).map((val, j) => (
                  <div
                    key={j}
                    className="embedding-cell"
                    style={{
                      backgroundColor: val > 0 
                        ? `rgba(67, 233, 123, ${Math.min(Math.abs(val) * 2, 1)})` 
                        : `rgba(255, 107, 107, ${Math.min(Math.abs(val) * 2, 1)})`
                    }}
                    title={`dim ${j}: ${val.toFixed(4)}`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!modelId) {
    return (
      <div className="transformer-workspace">
        <div className="workspace-placeholder">
          <div className="placeholder-icon">🔀</div>
          <h2>Transformer Workspace</h2>
          <p>Load a transformer model to begin exploring attention mechanisms and embeddings!</p>
          <div className="feature-grid">
            <div className="feature-card">
              <span className="feature-icon">📝</span>
              <h4>Text Models</h4>
              <p>BERT, GPT-2, DistilBERT, RoBERTa - Analyze how text is tokenized, encoded, and understood</p>
            </div>
            <div className="feature-card">
              <span className="feature-icon">🖼️</span>
              <h4>Vision Transformers</h4>
              <p>ViT, DeiT, Swin, CLIP - See how images are split into patches and processed</p>
            </div>
            <div className="feature-card">
              <span className="feature-icon">🎯</span>
              <h4>Attention Visualization</h4>
              <p>Explore how different tokens/patches attend to each other across layers and heads</p>
            </div>
            <div className="feature-card">
              <span className="feature-icon">📊</span>
              <h4>Embeddings</h4>
              <p>Visualize high-dimensional token representations in an interpretable way</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="transformer-workspace">
      <div className="workspace-layout">
        <div className="architecture-panel">
          <TransformerArchitecture 
            modelId={modelId}
            modelName={modelName}
            transformerType={transformerType}
            selectedLayer={selectedLayer}
            onLayerSelect={setSelectedLayer}
          />
        </div>

        <div className="visualization-panel">
          <div className="panel-header">
            <div className="tab-buttons">
              <button 
                className={activeTab === 'input' ? 'active' : ''}
                onClick={() => setActiveTab('input')}
              >
                {transformerType === 'text' ? '📝 Input' : '🖼️ Input'}
              </button>
              <button 
                className={activeTab === 'attention' ? 'active' : ''}
                onClick={() => setActiveTab('attention')}
                disabled={!output}
              >
                🎯 Attention
              </button>
              <button 
                className={activeTab === 'embeddings' ? 'active' : ''}
                onClick={() => setActiveTab('embeddings')}
                disabled={!output}
              >
                📊 Embeddings
              </button>
              {isDecoderModel && (
                <button 
                  className={activeTab === 'generation' ? 'active' : ''}
                  onClick={() => setActiveTab('generation')}
                  disabled={!output?.generation_steps?.length}
                >
                  ✨ Generation
                </button>
              )}
              <button 
                className={activeTab === 'output' ? 'active' : ''}
                onClick={() => setActiveTab('output')}
                disabled={!output}
              >
                📤 Output
              </button>
            </div>
            <button 
              className="explain-toggle"
              onClick={() => setShowExplanation(!showExplanation)}
            >
              {showExplanation ? '🎓 Hide Help' : '🎓 Show Help'}
            </button>
          </div>

          <div className="panel-content">
            {activeTab === 'input' && (
              <div className="input-tab">
                {showExplanation && (
                  <div className="explanation-box">
                    <h4>📖 What happens here?</h4>
                    {transformerType === 'text' ? (
                      <p>Your text is first <strong>tokenized</strong> - split into subword units. For example, "playing" might become ["play", "##ing"]. Each token gets converted to an ID, then to a dense vector (embedding). This is the starting point for the transformer.</p>
                    ) : (
                      <p>Your image is resized to 224×224 and split into <strong>16×16 patches</strong> (196 total). Each patch is flattened and linearly projected into an embedding. A special [CLS] token is prepended for classification. Position embeddings are added so the model knows spatial relationships.</p>
                    )}
                  </div>
                )}

                <div className="input-area">
                  <h3>{transformerType === 'text' ? '📝 Enter Text' : '🖼️ Upload Image'}</h3>
                  
                  {transformerType === 'text' ? (
                    <div className="text-input">
                      <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Enter text to analyze..."
                        rows={5}
                      />
                      <div className="char-count">{inputText.length} characters</div>
                    </div>
                  ) : (
                    <div className="image-input">
                      <ImageUploader onImageUpload={onImageUpload} imageFile={imageFile} />
                    </div>
                  )}

                  <div className="action-buttons">
                    <button 
                      onClick={() => runInference(false)} 
                      disabled={loading || (transformerType === 'text' ? !inputText.trim() : !imageFile)}
                      className="run-btn"
                    >
                      {loading ? '⏳ Processing...' : '▶ Analyze'}
                    </button>
                    
                    {isDecoderModel && transformerType === 'text' && (
                      <div className="generate-controls">
                        <button 
                          onClick={() => runInference(true)} 
                          disabled={loading || !inputText.trim()}
                          className="generate-btn"
                        >
                          {loading ? '⏳...' : '✨ Generate'}
                        </button>
                        <div className="token-slider">
                          <label>Tokens: {generateTokens}</label>
                          <input 
                            type="range" 
                            min="1" 
                            max="20" 
                            value={generateTokens}
                            onChange={(e) => setGenerateTokens(Number(e.target.value))}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {error && <div className="error-message">⚠️ {error}</div>}
                </div>

                {/* Vision Transformer Processing Pipeline */}
                {transformerType === 'image' && (
                  <div className="vit-pipeline-section">
                    <h3>🔄 Vision Transformer Processing Pipeline</h3>
                    <div className="pipeline-steps">
                      <div className="pipeline-step">
                        <div className="step-icon">🖼️</div>
                        <div className="step-content">
                          <h4>Raw Image Input</h4>
                          <p>Your input image (resized to 224×224)</p>
                          {imageFile && (
                            <div className="step-visual">
                              <img src={URL.createObjectURL(imageFile)} alt="Input" className="input-preview" />
                              <span className="size-badge">224 × 224 px</span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step">
                        <div className="step-icon">🧩</div>
                        <div className="step-content">
                          <h4>Patch Extraction</h4>
                          <p>Split image into 16×16 patches → <strong>196 patches</strong></p>
                          <div className="step-visual">
                            <div className="patch-grid">
                              {Array.from({ length: 196 }, (_, i) => (
                                <div key={i} className="mini-patch" title={`Patch ${i}`} />
                              ))}
                            </div>
                            <div className="patch-info">
                              <span>14 × 14 = 196 patches</span>
                              <span>Each patch: 16×16 pixels = 768 values (RGB flattened)</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step">
                        <div className="step-icon">🔢</div>
                        <div className="step-content">
                          <h4>Patch Embeddings</h4>
                          <p>Each patch is flattened and linearly projected into embedding space</p>
                          <div className="step-visual embedding-visual">
                            <div className="patch-to-embedding">
                              {/* Single patch visualization */}
                              <div className="patch-flatten-demo">
                                <div className="patch-3d">
                                  <div className="patch-label">16×16×3</div>
                                  <div className="patch-cube">
                                    <div className="cube-face front"></div>
                                    <div className="cube-face back"></div>
                                    <div className="cube-face right"></div>
                                  </div>
                                </div>
                                <div className="flatten-arrow">
                                  <span>flatten</span>
                                  <div className="arrow-line">→</div>
                                </div>
                                <div className="flat-vector">
                                  <div className="vector-bar">
                                    {Array.from({ length: 24 }).map((_, i) => (
                                      <div key={i} className="vector-cell" style={{ 
                                        background: `hsl(${i * 15}, 70%, ${50 + Math.sin(i * 0.5) * 20}%)`
                                      }} />
                                    ))}
                                  </div>
                                  <div className="vector-label">768 values</div>
                                </div>
                                <div className="flatten-arrow">
                                  <span>Linear</span>
                                  <div className="arrow-line">→</div>
                                </div>
                                <div className="embedding-vector">
                                  <div className="vector-bar embedding">
                                    {Array.from({ length: 24 }).map((_, i) => (
                                      <div key={i} className="vector-cell" style={{ 
                                        background: `hsl(${140 + i * 5}, 70%, ${40 + Math.cos(i * 0.3) * 25}%)`
                                      }} />
                                    ))}
                                  </div>
                                  <div className="vector-label">768-dim embedding</div>
                                </div>
                              </div>
                              
                              {/* All patches matrix */}
                              <div className="all-patches-matrix">
                                <div className="matrix-visual">
                                  <div className="matrix-bracket left">[</div>
                                  <div className="matrix-content">
                                    <div className="matrix-row">
                                      {Array.from({ length: 12 }).map((_, i) => (
                                        <div key={i} className="matrix-cell" style={{
                                          background: `hsl(${140 + i * 10}, 60%, 50%)`
                                        }} />
                                      ))}
                                      <span className="ellipsis">···</span>
                                    </div>
                                    <div className="matrix-row">
                                      {Array.from({ length: 12 }).map((_, i) => (
                                        <div key={i} className="matrix-cell" style={{
                                          background: `hsl(${160 + i * 10}, 60%, 45%)`
                                        }} />
                                      ))}
                                      <span className="ellipsis">···</span>
                                    </div>
                                    <div className="matrix-ellipsis">⋮</div>
                                    <div className="matrix-row">
                                      {Array.from({ length: 12 }).map((_, i) => (
                                        <div key={i} className="matrix-cell" style={{
                                          background: `hsl(${200 + i * 10}, 60%, 50%)`
                                        }} />
                                      ))}
                                      <span className="ellipsis">···</span>
                                    </div>
                                  </div>
                                  <div className="matrix-bracket right">]</div>
                                </div>
                                <div className="matrix-dims">
                                  <span className="dim-label rows">196 patches</span>
                                  <span className="dim-label cols">768 dimensions</span>
                                </div>
                              </div>
                            </div>
                            <p className="dim-explain">
                              <strong>Output:</strong> A matrix of shape [196 × 768] where each row is a learned representation of one image patch
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step">
                        <div className="step-icon">🎯</div>
                        <div className="step-content">
                          <h4>[CLS] Token</h4>
                          <p>Prepend learnable classification token</p>
                          <div className="step-visual cls-visual">
                            <div className="token-sequence">
                              <span className="cls-token">[CLS]</span>
                              <span className="patch-tokens">P₀, P₁, P₂, ... P₁₉₅</span>
                            </div>
                            <p className="dim-explain">Sequence length: 1 + 196 = <strong>197 tokens</strong></p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step">
                        <div className="step-icon">📍</div>
                        <div className="step-content">
                          <h4>Position Embeddings</h4>
                          <p>Add learnable position embeddings (0 to 196)</p>
                          <div className="step-visual pos-visual">
                            <div className="pos-equation">
                              <span>Input = Patch Embedding + Position Embedding</span>
                            </div>
                            <div className="pos-grid">
                              {Array.from({ length: 14 }, (_, i) => (
                                <div key={i} className="pos-cell">{i}</div>
                              ))}
                            </div>
                            <p className="dim-explain">Position tells model spatial relationships between patches</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step">
                        <div className="step-icon">🔄</div>
                        <div className="step-content">
                          <h4>Transformer Layers (×12)</h4>
                          <p>Stack of self-attention + MLP layers</p>
                          <div className="step-visual transformer-visual">
                            <div className="layer-block">
                              <div className="layer-component">Multi-Head Self-Attention</div>
                              <div className="layer-arrow">↓</div>
                              <div className="layer-component">Layer Norm + Residual</div>
                              <div className="layer-arrow">↓</div>
                              <div className="layer-component">MLP (Feed-Forward)</div>
                              <div className="layer-arrow">↓</div>
                              <div className="layer-component">Layer Norm + Residual</div>
                            </div>
                            <p className="dim-explain">Repeat 12 times → Each patch attends to all others</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="pipeline-arrow">↓</div>
                      
                      <div className="pipeline-step final-step">
                        <div className="step-icon">🏷️</div>
                        <div className="step-content">
                          <h4>Classification Head</h4>
                          <p>MLP on [CLS] token → 1000 ImageNet classes</p>
                          <div className="step-visual cls-head-visual">
                            <div className="cls-flow">
                              <span className="cls-output">[CLS] output</span>
                              <span className="cls-arrow">→ Linear →</span>
                              <span className="cls-logits">1000 logits</span>
                              <span className="cls-arrow">→ Softmax →</span>
                              <span className="cls-probs">Probabilities</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Text tokenization result */}
                {transformerType === 'text' && output?.input_tokens && (
                  <div className="tokens-section">
                    <h3>✂️ Tokenized Result</h3>
                    {showExplanation && (
                      <div className="mini-explanation">
                        Each colored box is a token. Special tokens like [CLS] (classification) and [SEP] (separator) are added by the model. Subwords may have ## prefix.
                      </div>
                    )}
                    <div className="token-list">
                      {output.input_tokens.map((token, i) => (
                        <span 
                          key={i} 
                          className={`token ${token.startsWith('[') ? 'special' : ''}`}
                          title={`Token ${i}: ${token}`}
                        >
                          {token}
                        </span>
                      ))}
                    </div>
                    <div className="token-count">{output.input_tokens.length} tokens</div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'attention' && (
              <div className="attention-tab">
                {showExplanation && (
                  <div className="explanation-box attention-deep-dive">
                    <h4>📖 What is Self-Attention? A Complete Breakdown</h4>
                    
                    <div className="attention-concept">
                      <h5>🎯 The Core Idea</h5>
                      <p>
                        Self-attention allows each token to "look at" every other token in the sequence and decide 
                        <strong> how relevant each one is</strong> for understanding the current token. 
                        This is what makes transformers so powerful - they can capture long-range dependencies.
                      </p>
                    </div>

                    <div className="attention-concept">
                      <h5>🔢 The Computation (Step by Step)</h5>
                      <div className="attention-steps">
                        <div className="att-step">
                          <span className="step-num">1</span>
                          <div>
                            <strong>Create Q, K, V vectors</strong>
                            <p>Each token's embedding (768 dims) is transformed into three vectors:</p>
                            <ul>
                              <li><strong>Query (Q):</strong> "What am I looking for?"</li>
                              <li><strong>Key (K):</strong> "What information do I contain?"</li>
                              <li><strong>Value (V):</strong> "What do I actually pass along?"</li>
                            </ul>
                          </div>
                        </div>
                        <div className="att-step">
                          <span className="step-num">2</span>
                          <div>
                            <strong>Compute Attention Scores</strong>
                            <p>For each token pair: <code>score = Q · K^T / √64</code></p>
                            <p>This dot product measures "how much does token A want to attend to token B?"</p>
                          </div>
                        </div>
                        <div className="att-step">
                          <span className="step-num">3</span>
                          <div>
                            <strong>Apply Softmax</strong>
                            <p>Convert scores to probabilities (each row sums to 1)</p>
                            <p>This is what you see in the matrix below!</p>
                          </div>
                        </div>
                        <div className="att-step">
                          <span className="step-num">4</span>
                          <div>
                            <strong>Weighted Sum of Values</strong>
                            <p><code>output = Σ (attention_weight × V)</code></p>
                            <p>Each token gets a new representation based on what it attended to</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="attention-concept">
                      <h5>📊 How to Read the Matrix Below</h5>
                      <div className="matrix-reading-guide">
                        <div className="guide-item">
                          <span className="guide-icon">→</span>
                          <div>
                            <strong>Rows = "From" token (the one asking)</strong>
                            <p>Read left to right: "Where is this token looking?"</p>
                          </div>
                        </div>
                        <div className="guide-item">
                          <span className="guide-icon">↓</span>
                          <div>
                            <strong>Columns = "To" token (the one being attended to)</strong>
                            <p>Read top to bottom: "Who is paying attention to this token?"</p>
                          </div>
                        </div>
                        <div className="guide-item">
                          <span className="guide-icon">🟢</span>
                          <div>
                            <strong>Cell brightness = Attention weight (0 to 1)</strong>
                            <p>Bright green = strong attention, Dark = weak attention</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="attention-concept">
                      <h5>🧠 What Different Heads Learn</h5>
                      <p>Each of the 12 heads learns different patterns:</p>
                      <ul className="head-patterns">
                        <li><strong>Positional heads:</strong> Attend to previous/next token (diagonal patterns)</li>
                        <li><strong>Syntactic heads:</strong> Connect subjects to verbs, adjectives to nouns</li>
                        <li><strong>Semantic heads:</strong> Link related concepts across the sentence</li>
                        <li><strong>Delimiter heads:</strong> Attend to [CLS], [SEP], periods</li>
                      </ul>
                    </div>
                  </div>
                )}

                <div className="attention-controls">
                  <div className="control-group">
                    <label>Layer:</label>
                    <select 
                      value={selectedLayer} 
                      onChange={(e) => setSelectedLayer(Number(e.target.value))}
                    >
                      {Array.from({ length: 12 }, (_, i) => (
                        <option key={i} value={i}>Layer {i + 1}</option>
                      ))}
                    </select>
                    <span className="control-hint">
                      {selectedLayer < 4 ? '🔤 Surface patterns (syntax, position)' : 
                       selectedLayer < 8 ? '🔗 Intermediate features' : 
                       '💡 Abstract concepts (meaning, relations)'}
                    </span>
                  </div>
                  <div className="control-group">
                    <label>Head:</label>
                    <select 
                      value={selectedHead} 
                      onChange={(e) => setSelectedHead(Number(e.target.value))}
                    >
                      {Array.from({ length: 12 }, (_, i) => (
                        <option key={i} value={i}>Head {i + 1}</option>
                      ))}
                    </select>
                    <span className="control-hint">Each head captures different relationship patterns</span>
                  </div>
                </div>

                {output?.attention_layers ? (
                  <div className="attention-viz">
                    <div className="matrix-header-info">
                      <h4>🎯 Attention Matrix (Layer {selectedLayer + 1}, Head {selectedHead + 1})</h4>
                      <div className="matrix-instructions">
                        <span>📖 Row = "From" token asking</span>
                        <span>📖 Column = "To" token being attended</span>
                        <span>💡 Hover over cells for exact values</span>
                      </div>
                    </div>
                    
                    <div className="matrix-with-annotations">
                      <div className="row-label-annotation">
                        <span>← Token asking "what should I attend to?"</span>
                      </div>
                      {renderAttentionMatrix()}
                      <div className="col-label-annotation">
                        <span>↑ Tokens being attended to</span>
                      </div>
                    </div>

                    <div className="attention-insights">
                      <h5>🔍 Pattern Analysis</h5>
                      <div className="insight-cards">
                        <div className="insight-card">
                          <span className="insight-icon">⬛</span>
                          <div>
                            <strong>Diagonal Pattern?</strong>
                            <p>Tokens attending to themselves or neighbors (positional attention)</p>
                          </div>
                        </div>
                        <div className="insight-card">
                          <span className="insight-icon">📍</span>
                          <div>
                            <strong>First Column Bright?</strong>
                            <p>[CLS] token is often attended to - it aggregates sentence info</p>
                          </div>
                        </div>
                        <div className="insight-card">
                          <span className="insight-icon">📏</span>
                          <div>
                            <strong>Vertical Lines?</strong>
                            <p>Certain key tokens are attended to by many others</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="matrix-legend">
                      <span className="legend-label">0% attention</span>
                      <div className="legend-gradient" />
                      <span className="legend-label">100% attention</span>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder-box">
                    <p>Run inference to see attention patterns</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'embeddings' && (
              <div className="embeddings-tab">
                {showExplanation && (
                  <div className="explanation-box">
                    <h4>📖 Understanding Embeddings</h4>
                    <p>
                      Each token is represented as a high-dimensional vector (typically 768 dimensions). 
                      Here we visualize the first 64 dimensions. <strong>Green = positive values</strong>, 
                      <strong> Red = negative values</strong>. Similar words have similar patterns.
                      These embeddings capture semantic meaning - the model's "understanding" of each token.
                    </p>
                  </div>
                )}

                {output?.embeddings ? (
                  <div className="embeddings-viz">
                    <h4>📊 Token Embeddings (first 64 dimensions)</h4>
                    {renderTokenEmbeddings()}
                    <div className="embedding-legend">
                      <div className="legend-item">
                        <span className="legend-color negative" />
                        <span>Negative</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color zero" />
                        <span>Zero</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color positive" />
                        <span>Positive</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder-box">
                    <p>Run inference to see token embeddings</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'generation' && output?.generation_steps && (
              <div className="generation-tab">
                {showExplanation && (
                  <div className="explanation-box">
                    <h4>📖 Step-by-Step Token Generation</h4>
                    <p>
                      GPT-2 generates text <strong>one token at a time</strong>. At each step:
                    </p>
                    <ol className="generation-steps-explain">
                      <li><strong>Input Processing:</strong> The current text is tokenized and passed through all transformer layers</li>
                      <li><strong>Hidden State:</strong> The final layer produces a hidden state vector (768 dims) for the last position</li>
                      <li><strong>Logits:</strong> Hidden state is multiplied by the embedding matrix to get scores for all ~50k tokens</li>
                      <li><strong>Softmax:</strong> Scores are converted to probabilities (sum to 1)</li>
                      <li><strong>Selection:</strong> The highest probability token is selected (greedy decoding)</li>
                      <li><strong>Append & Repeat:</strong> Selected token is added to input, process repeats</li>
                    </ol>
                  </div>
                )}

                <div className="generation-result">
                  <h4>✨ Generated Text</h4>
                  <div className="full-text-display">
                    <span className="original-text">{inputText}</span>
                    <span className="generated-text-inline">{output.generated_text}</span>
                  </div>
                </div>

                <div className="step-navigator">
                  <h4>🔍 Step-by-Step Breakdown</h4>
                  <div className="step-buttons">
                    {output.generation_steps.map((step, i) => (
                      <button
                        key={i}
                        className={`step-btn ${selectedStep === i ? 'active' : ''}`}
                        onClick={() => setSelectedStep(i)}
                      >
                        Step {step.step}: "{step.selected_token}"
                      </button>
                    ))}
                  </div>
                </div>

                {output.generation_steps[selectedStep] && (
                  <div className="step-detail">
                    <div className="step-info-card">
                      <h5>Step {output.generation_steps[selectedStep].step}</h5>
                      <div className="step-context">
                        <label>Input so far:</label>
                        <div className="context-text">{output.generation_steps[selectedStep].input_so_far}</div>
                      </div>
                    </div>

                    <div className="step-process">
                      <div className="process-step">
                        <span className="process-num">1</span>
                        <div className="process-content">
                          <strong>Tokenize & Encode</strong>
                          <p>Input text → Token IDs → Embeddings → Through 12 transformer layers</p>
                        </div>
                      </div>
                      <div className="process-arrow">↓</div>
                      
                      <div className="process-step">
                        <span className="process-num">2</span>
                        <div className="process-content">
                          <strong>Compute Logits</strong>
                          <p>Hidden state × Embedding matrix = {(50257).toLocaleString()} raw scores</p>
                          <div className="logits-stats">
                            min: {output.generation_steps[selectedStep].logits_stats.min.toFixed(2)} | 
                            max: {output.generation_steps[selectedStep].logits_stats.max.toFixed(2)} | 
                            mean: {output.generation_steps[selectedStep].logits_stats.mean.toFixed(2)}
                          </div>
                        </div>
                      </div>
                      <div className="process-arrow">↓</div>
                      
                      <div className="process-step">
                        <span className="process-num">3</span>
                        <div className="process-content">
                          <strong>Softmax → Probabilities</strong>
                          <p>Convert scores to probability distribution (sums to 1)</p>
                        </div>
                      </div>
                      <div className="process-arrow">↓</div>
                      
                      <div className="process-step">
                        <span className="process-num">4</span>
                        <div className="process-content">
                          <strong>Top Token Predictions</strong>
                          <div className="top-predictions">
                            {output.generation_steps[selectedStep].top_predictions.map((pred, i) => (
                              <div key={i} className={`prediction-item ${i === 0 ? 'selected' : ''}`}>
                                <span className="pred-rank">#{i + 1}</span>
                                <span className="pred-token">"{pred.token}"</span>
                                <div className="pred-bar-container">
                                  <div 
                                    className="pred-bar" 
                                    style={{ width: `${pred.probability * 100}%` }}
                                  />
                                </div>
                                <span className="pred-prob">{(pred.probability * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="process-arrow">↓</div>
                      
                      <div className="process-step selected-step">
                        <span className="process-num">5</span>
                        <div className="process-content">
                          <strong>Selected (Greedy):</strong>
                          <span className="selected-token">"{output.generation_steps[selectedStep].selected_token}"</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'output' && (
              <div className="output-tab">
                {showExplanation && (
                  <div className="explanation-box">
                    <h4>📖 Understanding the Output</h4>
                    {transformerType === 'text' ? (
                      <p>
                        For encoder models (BERT), the output is contextualized embeddings for each token.
                        The [CLS] token embedding is often used for classification tasks.
                        For decoder models (GPT-2), the output is the next predicted token.
                      </p>
                    ) : (
                      <p>
                        The [CLS] token's final embedding goes through a classification head (MLP) to produce
                        logits for each class. Softmax converts these to probabilities. The highest probability
                        class is the model's prediction.
                      </p>
                    )}
                  </div>
                )}

                {/* Classification Head Visualization */}
                {transformerType === 'image' && (
                  <div className="classification-head-viz">
                    <h4>🧠 Classification Head (Linear Layer)</h4>
                    <div className="head-pipeline">
                      {/* [CLS] Token Embedding */}
                      <div className="pipeline-block cls-embedding">
                        <div className="block-label">[CLS] Embedding</div>
                        <div className="vector-visual">
                          <div className="vector-container">
                            {Array.from({ length: 32 }).map((_, i) => (
                              <div 
                                key={i} 
                                className="vec-cell"
                                style={{ 
                                  background: `hsl(${200 + Math.sin(i * 0.5) * 40}, 70%, ${45 + Math.cos(i * 0.3) * 20}%)`
                                }}
                              />
                            ))}
                            <span className="ellipsis">···</span>
                          </div>
                          <div className="dim-tag">768 dimensions</div>
                        </div>
                      </div>

                      <div className="pipeline-connector">
                        <div className="connector-arrow">→</div>
                      </div>

                      {/* Weight Matrix */}
                      <div className="pipeline-block weight-matrix">
                        <div className="block-label">Linear Layer Weights</div>
                        <div className="matrix-visual-compact">
                          <div className="matrix-bracket-small">[</div>
                          <div className="matrix-grid">
                            {Array.from({ length: 6 }).map((_, row) => (
                              <div key={row} className="matrix-row-compact">
                                {Array.from({ length: 8 }).map((_, col) => (
                                  <div 
                                    key={col} 
                                    className="matrix-cell-small"
                                    style={{
                                      background: `hsl(${(row * col * 20) % 360}, 60%, ${40 + (row + col) * 3}%)`
                                    }}
                                  />
                                ))}
                                <span className="row-ellipsis">···</span>
                              </div>
                            ))}
                            <div className="matrix-vellipsis">⋮</div>
                          </div>
                          <div className="matrix-bracket-small">]</div>
                        </div>
                        <div className="dim-tag">768 × 1000</div>
                        <div className="param-count">768,000 weights</div>
                      </div>

                      <div className="pipeline-connector">
                        <div className="connector-arrow">→</div>
                        <div className="connector-label">matmul + bias</div>
                      </div>

                      {/* Logits Output */}
                      <div className="pipeline-block logits-output">
                        <div className="block-label">Logits</div>
                        <div className="logits-visual">
                          <div className="logits-container">
                            {Array.from({ length: 20 }).map((_, i) => (
                              <div 
                                key={i} 
                                className="logit-bar"
                                style={{ 
                                  height: `${15 + Math.random() * 25}px`,
                                  background: i < 3 ? '#43e97b' : '#444'
                                }}
                              />
                            ))}
                            <span className="ellipsis">···</span>
                          </div>
                          <div className="dim-tag">1000 classes</div>
                        </div>
                      </div>

                      <div className="pipeline-connector">
                        <div className="connector-arrow">→</div>
                        <div className="connector-label">softmax</div>
                      </div>

                      {/* Probabilities */}
                      <div className="pipeline-block probabilities">
                        <div className="block-label">Probabilities</div>
                        <div className="prob-visual">
                          <div className="prob-distribution">
                            {output?.predictions?.slice(0, 5).map((pred, i) => (
                              <div 
                                key={i} 
                                className="prob-spike"
                                style={{ 
                                  height: `${pred.probability * 100}px`,
                                  background: i === 0 ? '#43e97b' : `rgba(67, 233, 123, ${0.5 - i * 0.1})`
                                }}
                                title={`${pred.label}: ${(pred.probability * 100).toFixed(1)}%`}
                              />
                            ))}
                            <div className="rest-probs" title="Other 995 classes">
                              {Array.from({ length: 10 }).map((_, i) => (
                                <div key={i} className="tiny-bar" />
                              ))}
                            </div>
                          </div>
                          <div className="dim-tag">sum = 1.0</div>
                        </div>
                      </div>
                    </div>

                    <div className="head-formula">
                      <code>P(class) = softmax(W × [CLS] + b)</code>
                    </div>
                  </div>
                )}

                {output?.predictions && output.predictions.length > 0 ? (
                  <div className="predictions-section">
                    <h4>🎯 Top Predictions</h4>
                    <div className="predictions-list">
                      {output.predictions.map((pred, i) => (
                        <div key={i} className={`prediction-row ${i === 0 ? 'top' : ''}`}>
                          <span className="rank">#{i + 1}</span>
                          <span className="label">{pred.label}</span>
                          <div className="prob-bar-container">
                            <div 
                              className="prob-bar"
                              style={{ width: `${pred.probability * 100}%` }}
                            />
                          </div>
                          <span className="prob-value">{(pred.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : output?.output_tokens ? (
                  <div className="generated-section">
                    <h4>📝 Model Output</h4>
                    <div className="generated-text">
                      {output.output_tokens.join('')}
                    </div>
                  </div>
                ) : (
                  <div className="placeholder-box">
                    <p>Run inference to see model output</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default TransformerWorkspace
