import { useState } from 'react'
import axios from 'axios'
import './DiffusionWorkspace.css'

interface DiffusionWorkspaceProps {
  modelId: string | null
  modelName: string
}

interface IntermediateStep {
  step: number
  timestep: number
  latent_shape: number[]
  latent_stats: {
    mean: number
    std: number
    min: number
    max: number
  }
}

interface IntermediateImage {
  step: number
  timestep: number
  image: string
}

interface ArchitectureInfo {
  unet: {
    input_channels: number
    output_channels: number
    block_out_channels: number[]
    attention_head_dim: number
    num_attention_heads: number
    cross_attention_dim: number
  }
  vae: {
    latent_channels: number
    sample_size: number
  }
  text_encoder: {
    max_position_embeddings: number
    hidden_size: number
  }
  scheduler: {
    type: string
    num_train_timesteps: number
  }
}

interface GenerationResult {
  image: string
  prompt: string
  intermediate_steps: IntermediateStep[]
  intermediate_images: IntermediateImage[]
  architecture_info: ArchitectureInfo
  generation_params: {
    num_inference_steps: number
    guidance_scale: number
    seed: number | null
  }
}

function DiffusionWorkspace({ modelId }: DiffusionWorkspaceProps) {
  const [prompt, setPrompt] = useState('a beautiful sunset over mountains')
  const [numSteps, setNumSteps] = useState(50)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [seed, setSeed] = useState<number | null>(null)
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'generate' | 'process' | 'architecture' | 'explanation'>('explanation')
  const [selectedStep, setSelectedStep] = useState(0)
  const [architectureInfo, setArchitectureInfo] = useState<any>(null)
  const [loadingArchitecture, setLoadingArchitecture] = useState(false)

  const generateImage = async () => {
    if (!modelId) {
      setError('Please load a model first')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`http://localhost:8000/diffusion/${modelId}/generate`, {
        prompt,
        num_inference_steps: numSteps,
        guidance_scale: guidanceScale,
        seed: seed || undefined,
        return_intermediates: true
      })
      setResult(response.data)
      setActiveTab('process')
      setSelectedStep(0)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate image')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="diffusion-workspace">
      <div className="diffusion-left-panel">
        <div className="data-flow-section">
          <h3>🎨 Diffusion Model Data Flow</h3>
          <div className="flow-steps">
            <div className="flow-step">
              <div className="flow-icon">📝</div>
              <div className="flow-content">
                <strong>Text Prompt</strong>
                <p>Your input text describing what to generate</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🔤</div>
              <div className="flow-content">
                <strong>Text Encoder (CLIP)</strong>
                <p>Converts text into embeddings (768-dim vectors)</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🎲</div>
              <div className="flow-content">
                <strong>Random Noise</strong>
                <p>Start with pure random noise (latent space)</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🔄</div>
              <div className="flow-content">
                <strong>Denoising Loop</strong>
                <p>U-Net predicts and removes noise step by step</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">🖼️</div>
              <div className="flow-content">
                <strong>VAE Decoder</strong>
                <p>Converts latent space back to image pixels</p>
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">
              <div className="flow-icon">✨</div>
              <div className="flow-content">
                <strong>Final Image</strong>
                <p>Generated image matching your prompt</p>
              </div>
            </div>
          </div>
        </div>

        <div className="explanation-section">
          <h3>🧠 How Diffusion Models Work</h3>
          <div className="explanation-content">
            <div className="explanation-block">
              <h4>1. Forward Diffusion (Training)</h4>
              <p>During training, we gradually add noise to real images over many steps (typically 1000). This teaches the model what noise looks like at different stages.</p>
              <div className="formula-box">
                <code>x&#123;t&#125; = √(α&#123;t&#125;) · x&#123;0&#125; + √(1 - α&#123;t&#125;) · ε</code>
                <p className="formula-explanation">At step t, we mix the original image x_0 with noise ε, where α_t controls how much noise is added.</p>
              </div>
            </div>

            <div className="explanation-block">
              <h4>2. Reverse Diffusion (Generation)</h4>
              <p>To generate, we start with pure noise and use a U-Net to predict what noise to remove at each step, gradually revealing the image.</p>
              <div className="formula-box">
                <code>x&#123;t-1&#125; = x&#123;t&#125; - predicted_noise · step_size</code>
                <p className="formula-explanation">At each step, we subtract the predicted noise to get a slightly cleaner image.</p>
              </div>
            </div>

            <div className="explanation-block">
              <h4>3. Guidance (Classifier-Free)</h4>
              <p>We train the model with and without text prompts. During generation, we use both predictions and amplify the difference to follow the prompt better.</p>
              <div className="formula-box">
                <code>prediction = uncond_pred + guidance_scale · (cond_pred - uncond_pred)</code>
                <p className="formula-explanation">Higher guidance_scale makes the model follow the prompt more strictly (but can reduce diversity).</p>
              </div>
            </div>

            <div className="explanation-block">
              <h4>4. Latent Space</h4>
              <p>Instead of working with full-resolution images (512×512 = 262K pixels), we work in a compressed latent space (64×64 = 4K values). This is 64× faster!</p>
              <div className="formula-box">
                <code>Image (512×512×3) → VAE Encoder → Latent (64×64×4) → Denoise → VAE Decoder → Image</code>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="diffusion-right-panel">
        <div className="diffusion-tabs">
          <button 
            className={`tab-btn ${activeTab === 'explanation' ? 'active' : ''}`}
            onClick={() => setActiveTab('explanation')}
          >
            📚 How It Works
          </button>
          <button 
            className={`tab-btn ${activeTab === 'generate' ? 'active' : ''}`}
            onClick={() => setActiveTab('generate')}
          >
            🎨 Generate
          </button>
          <button 
            className={`tab-btn ${activeTab === 'process' ? 'active' : ''}`}
            onClick={() => setActiveTab('process')}
            disabled={!result}
          >
            🔄 Process
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
              <h2>🎓 Understanding Diffusion Models</h2>
              
              <div className="concept-section">
                <h3>What is Diffusion?</h3>
                <p>Diffusion models learn to generate images by reversing a noise-adding process. Think of it like watching a video in reverse: you start with noise and gradually reveal the image.</p>
              </div>

              <div className="concept-section">
                <h3>Key Components</h3>
                <div className="component-list">
                  <div className="component-item">
                    <strong>🔤 Text Encoder (CLIP)</strong>
                    <p>Converts your text prompt into numerical embeddings that the model can understand. It's trained on millions of image-text pairs to understand relationships.</p>
                  </div>
                  <div className="component-item">
                    <strong>🎯 U-Net (Noise Predictor)</strong>
                    <p>The core of the model. Takes noisy latents and text embeddings, predicts what noise to remove. Uses attention mechanisms to understand spatial relationships.</p>
                  </div>
                  <div className="component-item">
                    <strong>🔄 VAE (Variational Autoencoder)</strong>
                    <p>Encodes images to a compressed latent space (64×64×4) and decodes back. This makes generation 64× faster than working with full-resolution images.</p>
                  </div>
                  <div className="component-item">
                    <strong>⏱️ Scheduler</strong>
                    <p>Controls the denoising schedule - how much noise to remove at each step. Different schedulers (DDPM, DPM-Solver, etc.) have different strategies.</p>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>The Denoising Process (Step-by-Step)</h3>
                <div className="step-list">
                  <div className="step-item">
                    <div className="step-number">1</div>
                    <div className="step-content">
                      <strong>Start with Noise</strong>
                      <p>Initialize with random noise in latent space (64×64×4 tensor). This is our "blank canvas".</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>Encode Text Prompt</strong>
                      <p>Convert your text prompt into embeddings using CLIP text encoder. These embeddings guide the generation.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>Predict Noise</strong>
                      <p>U-Net looks at the noisy latents and text embeddings, predicts what noise should be removed at this step.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">4</div>
                    <div className="step-content">
                      <strong>Remove Noise</strong>
                      <p>Subtract the predicted noise from the latents, making them slightly cleaner. This is one denoising step.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">5</div>
                    <div className="step-content">
                      <strong>Repeat</strong>
                      <p>Repeat steps 3-4 for the specified number of steps (typically 20-50). Each step makes the image clearer.</p>
                    </div>
                  </div>
                  <div className="step-item">
                    <div className="step-number">6</div>
                    <div className="step-content">
                      <strong>Decode to Image</strong>
                      <p>Once denoising is complete, use VAE decoder to convert clean latents back to a full-resolution image.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="concept-section">
                <h3>Why This Works</h3>
                <p>During training, the model sees millions of examples of "noisy image → clean image" transitions. It learns patterns like:</p>
                <ul>
                  <li>How to recognize structure in noise</li>
                  <li>How text descriptions relate to visual features</li>
                  <li>How to gradually refine details</li>
                </ul>
                <p>At generation time, it applies this learned knowledge in reverse, starting from pure noise and gradually revealing a coherent image.</p>
              </div>
            </div>
          )}

          {activeTab === 'generate' && (
            <div className="generate-tab">
              <h2>🎨 Generate Image</h2>
              
              <div className="input-section">
                <label>
                  <strong>Text Prompt</strong>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the image you want to generate..."
                    rows={4}
                    className="prompt-input"
                  />
                </label>
              </div>

              <div className="params-section">
                <div className="param-group">
                  <label>
                    <strong>Inference Steps</strong>
                    <span className="param-value">{numSteps}</span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                    className="param-slider"
                  />
                  <p className="param-help">More steps = better quality but slower. 20-50 is usually good.</p>
                </div>

                <div className="param-group">
                  <label>
                    <strong>Guidance Scale</strong>
                    <span className="param-value">{guidanceScale.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    className="param-slider"
                  />
                  <p className="param-help">How closely to follow the prompt. Higher = more faithful but less creative. 7-9 is typical.</p>
                </div>

                <div className="param-group">
                  <label>
                    <strong>Seed (Optional)</strong>
                    <input
                      type="number"
                      value={seed || ''}
                      onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : null)}
                      placeholder="Random"
                      className="seed-input"
                    />
                  </label>
                  <p className="param-help">Same seed + same prompt = same image. Leave empty for random.</p>
                </div>
              </div>

              <button 
                onClick={generateImage}
                disabled={loading || !modelId}
                className="generate-btn"
              >
                {loading ? '🔄 Generating...' : '✨ Generate Image'}
              </button>

              {error && (
                <div className="error-message">
                  ⚠️ {error}
                </div>
              )}

              {result && (
                <div className="result-preview">
                  <h3>✨ Generated Image</h3>
                  <img src={`data:image/png;base64,${result.image}`} alt="Generated" className="generated-image" />
                  <div className="result-info">
                    <p><strong>Prompt:</strong> {result.prompt}</p>
                    <p><strong>Steps:</strong> {result.generation_params.num_inference_steps}</p>
                    <p><strong>Guidance:</strong> {result.generation_params.guidance_scale}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'process' && result && (
            <div className="process-tab">
              <h2>🔄 Denoising Process</h2>
              
              <div className="step-selector">
                <label>
                  <strong>View Step:</strong>
                  <select
                    value={selectedStep}
                    onChange={(e) => setSelectedStep(Number(e.target.value))}
                    className="step-select"
                  >
                    {result.intermediate_steps.map((step, idx) => (
                      <option key={idx} value={idx}>
                        Step {step.step} (t={step.timestep.toFixed(0)})
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="step-visualization">
                {result.intermediate_images[selectedStep] && (
                  <div className="intermediate-image">
                    <h3>Step {result.intermediate_steps[selectedStep].step}</h3>
                    <img 
                      src={`data:image/png;base64,${result.intermediate_images[selectedStep].image}`} 
                      alt={`Step ${selectedStep}`}
                      className="step-image"
                    />
                  </div>
                )}

                <div className="latent-stats">
                  <h3>Latent Space Statistics</h3>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-label">Shape</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].latent_shape.join(' × ')}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Mean</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].latent_stats.mean.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Std Dev</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].latent_stats.std.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Min</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].latent_stats.min.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Max</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].latent_stats.max.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Timestep</div>
                      <div className="stat-value">
                        {result.intermediate_steps[selectedStep].timestep.toFixed(0)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="process-explanation">
                  <h3>What's Happening at This Step</h3>
                  <p>
                    At step {result.intermediate_steps[selectedStep].step}, the model has removed some noise from the initial random noise.
                    The latent values are becoming more structured, and patterns are starting to emerge.
                    As we progress through steps, the noise decreases and the image becomes clearer.
                  </p>
                  <p>
                    <strong>Timestep {result.intermediate_steps[selectedStep].timestep.toFixed(0)}</strong> represents how much noise remains.
                    Higher timesteps = more noise. We start at a high timestep (e.g., 1000) and work down to 0.
                  </p>
                </div>
              </div>
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
                      const response = await axios.get(`http://localhost:8000/diffusion/${modelId}/architecture`)
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
                    <h3>🎯 U-Net (Noise Predictor)</h3>
                    <p className="component-description">
                      The core component that predicts noise to remove at each denoising step.
                      Uses attention mechanisms to understand spatial relationships and text guidance.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>Input Channels:</strong> {architectureInfo.components.unet.input_channels}
                      </div>
                      <div className="spec-item">
                        <strong>Output Channels:</strong> {architectureInfo.components.unet.output_channels}
                      </div>
                      <div className="spec-item">
                        <strong>Attention Heads:</strong> {architectureInfo.components.unet.num_attention_heads}
                      </div>
                      <div className="spec-item">
                        <strong>Head Dimension:</strong> {architectureInfo.components.unet.attention_head_dim}
                      </div>
                      <div className="spec-item">
                        <strong>Cross-Attention Dim:</strong> {architectureInfo.components.unet.cross_attention_dim}
                      </div>
                    </div>
                  </div>

                  <div className="component-section">
                    <h3>🔄 VAE (Variational Autoencoder)</h3>
                    <p className="component-description">
                      Encodes images to compressed latent space (64×64×4) and decodes back to full resolution.
                      This compression makes generation 64× faster than working with full-resolution images.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>Latent Channels:</strong> {architectureInfo.components.vae.latent_channels}
                      </div>
                      <div className="spec-item">
                        <strong>Sample Size:</strong> {architectureInfo.components.vae.sample_size}
                      </div>
                    </div>
                  </div>

                  <div className="component-section">
                    <h3>🔤 Text Encoder (CLIP)</h3>
                    <p className="component-description">
                      Converts text prompts into numerical embeddings that guide the generation process.
                      Trained on millions of image-text pairs to understand semantic relationships.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>Hidden Size:</strong> {architectureInfo.components.text_encoder.hidden_size}
                      </div>
                      <div className="spec-item">
                        <strong>Max Position Embeddings:</strong> {architectureInfo.components.text_encoder.max_position_embeddings}
                      </div>
                    </div>
                  </div>

                  <div className="component-section">
                    <h3>⏱️ Scheduler</h3>
                    <p className="component-description">
                      Controls the denoising schedule - determines how much noise to remove at each step.
                      Different schedulers use different strategies for optimal quality vs speed trade-offs.
                    </p>
                    <div className="component-specs">
                      <div className="spec-item">
                        <strong>Type:</strong> {architectureInfo.components.scheduler.type}
                      </div>
                      <div className="spec-item">
                        <strong>Training Timesteps:</strong> {architectureInfo.components.scheduler.num_train_timesteps}
                      </div>
                    </div>
                  </div>

                  <div className="total-params">
                    <strong>Total Parameters:</strong> {(architectureInfo.total_params / 1e9).toFixed(2)}B
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DiffusionWorkspace

