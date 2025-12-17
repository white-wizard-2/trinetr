import { useState, useEffect } from 'react'
import axios from 'axios'
import './TransformerArchitecture.css'

interface TransformerArchitectureProps {
  modelId: string
  modelName: string
  transformerType: 'text' | 'image'
  onLayerSelect?: (layer: number) => void
  selectedLayer?: number
}

interface LayerInfo {
  name: string
  type: string
  params: number
  in_features?: number
  out_features?: number
  num_heads?: number
  hidden_size?: number
}

const MODEL_EXPLANATIONS: { [key: string]: { description: string; architecture: string; useCase: string } } = {
  'bert-base': {
    description: 'BERT (Bidirectional Encoder Representations from Transformers) reads text in both directions simultaneously, understanding context from left and right.',
    architecture: '12 transformer encoder layers, each with 12 attention heads. Hidden size: 768. Total: ~110M parameters.',
    useCase: 'Text classification, named entity recognition, question answering, sentiment analysis.'
  },
  'distilbert': {
    description: 'DistilBERT is a smaller, faster version of BERT that retains 97% of its language understanding while being 60% faster.',
    architecture: '6 transformer encoder layers (half of BERT), 12 attention heads. Hidden size: 768. Total: ~66M parameters.',
    useCase: 'Same as BERT but for resource-constrained environments or when speed is critical.'
  },
  'gpt2': {
    description: 'GPT-2 is an autoregressive model that generates text by predicting the next token based on all previous tokens.',
    architecture: '12 transformer decoder layers with masked self-attention. Hidden size: 768. Total: ~117M parameters.',
    useCase: 'Text generation, story writing, code completion, creative writing.'
  },
  'roberta': {
    description: 'RoBERTa is an optimized BERT with improved training methodology - more data, larger batches, and dynamic masking.',
    architecture: '12 transformer encoder layers, 12 attention heads. Hidden size: 768. Total: ~125M parameters.',
    useCase: 'All BERT use cases with generally better performance on benchmarks.'
  },
  'vit-base': {
    description: 'Vision Transformer (ViT) treats images as sequences of patches, applying transformer architecture to computer vision.',
    architecture: '12 transformer encoder layers. Patch size: 16x16. Input: 224x224 → 196 patches + 1 [CLS] token.',
    useCase: 'Image classification, feature extraction for downstream tasks.'
  },
  'deit-small': {
    description: 'DeiT (Data-efficient Image Transformer) is trained with knowledge distillation, requiring less data than ViT.',
    architecture: '12 transformer encoder layers with distillation token. Smaller hidden dimension than ViT-base.',
    useCase: 'Image classification when training data is limited.'
  },
  'swin-tiny': {
    description: 'Swin Transformer uses shifted windows for efficient self-attention, enabling hierarchical feature extraction.',
    architecture: '4 stages with increasing channels (96→192→384→768). Window size: 7x7 with shifts.',
    useCase: 'Image classification, object detection, semantic segmentation.'
  },
  'clip-vit': {
    description: 'CLIP connects images and text in a shared embedding space, trained on 400M image-text pairs.',
    architecture: 'ViT-B/16 vision encoder paired with transformer text encoder. Contrastive learning objective.',
    useCase: 'Zero-shot classification, image-text matching, image search.'
  }
}

const DATA_FLOW_STEPS = {
  text: [
    { id: 'input', name: 'Raw Text Input', icon: '📝', description: 'Your input text string' },
    { id: 'tokenize', name: 'Tokenization', icon: '✂️', description: 'Split text into subword tokens using BPE/WordPiece' },
    { id: 'embed', name: 'Token Embeddings', icon: '🔢', description: 'Convert token IDs to dense vectors (768-dim)' },
    { id: 'pos', name: 'Positional Encoding', icon: '📍', description: 'Add position information to embeddings' },
    { id: 'layers', name: 'Transformer Layers', icon: '🔄', description: 'Stack of self-attention + feed-forward layers' },
    { id: 'output', name: 'Output Embeddings', icon: '📤', description: 'Contextualized representations for each token' },
  ],
  image: [
    { id: 'input', name: 'Raw Image Input', icon: '🖼️', description: 'Your input image (resized to 224x224)' },
    { id: 'patch', name: 'Patch Extraction', icon: '🧩', description: 'Split image into 16x16 patches → 196 patches' },
    { id: 'embed', name: 'Patch Embeddings', icon: '🔢', description: 'Linear projection of flattened patches' },
    { id: 'cls', name: '[CLS] Token', icon: '🎯', description: 'Prepend learnable classification token' },
    { id: 'pos', name: 'Position Embeddings', icon: '📍', description: 'Add learnable position embeddings' },
    { id: 'layers', name: 'Transformer Layers', icon: '🔄', description: 'Stack of self-attention + MLP layers' },
    { id: 'head', name: 'Classification Head', icon: '🏷️', description: 'MLP on [CLS] token for final prediction' },
  ]
}

function TransformerArchitecture({ modelId, modelName, transformerType, onLayerSelect, selectedLayer }: TransformerArchitectureProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [architecture, setArchitecture] = useState<{ layers: LayerInfo[], total_params: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

  useEffect(() => {
    if (modelId) {
      fetchArchitecture()
    }
  }, [modelId])

  const fetchArchitecture = async () => {
    setLoading(true)
    try {
      const response = await axios.get(`http://localhost:8000/transformers/${modelId}/architecture`)
      setArchitecture(response.data)
    } catch (err) {
      console.error('Failed to fetch architecture:', err)
    } finally {
      setLoading(false)
    }
  }

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(section)) {
      newExpanded.delete(section)
    } else {
      newExpanded.add(section)
    }
    setExpandedSections(newExpanded)
  }

  const modelInfo = MODEL_EXPLANATIONS[modelName] || {
    description: 'A transformer-based neural network model.',
    architecture: 'Standard transformer architecture with self-attention layers.',
    useCase: 'General purpose NLP/vision tasks.'
  }

  const dataFlow = DATA_FLOW_STEPS[transformerType]

  const formatParams = (params: number) => {
    if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`
    if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`
    if (params >= 1e3) return `${(params / 1e3).toFixed(2)}K`
    return params.toString()
  }

  return (
    <div className="transformer-architecture">
      <div className="arch-header">
        <h3>🔀 {modelName.toUpperCase()}</h3>
        <button className="info-toggle" onClick={() => setShowInfo(!showInfo)}>
          {showInfo ? '✕' : 'ℹ️'}
        </button>
      </div>

      {showInfo && (
        <div className="model-info-panel">
          <div className="info-section">
            <h4>📖 What is it?</h4>
            <p>{modelInfo.description}</p>
          </div>
          <div className="info-section">
            <h4>🏗️ Architecture</h4>
            <p>{modelInfo.architecture}</p>
          </div>
          <div className="info-section">
            <h4>🎯 Use Cases</h4>
            <p>{modelInfo.useCase}</p>
          </div>
        </div>
      )}

      <div className="data-flow-section">
        <div 
          className={`section-header ${expandedSections.has('flow') ? 'expanded' : ''}`}
          onClick={() => toggleSection('flow')}
        >
          <span className="expand-icon">{expandedSections.has('flow') ? '▼' : '▶'}</span>
          <span>📊 Data Flow</span>
          <span className="step-count">{dataFlow.length} steps</span>
        </div>
        
        {expandedSections.has('flow') && (
          <div className="flow-steps">
            {dataFlow.map((step, idx) => (
              <div key={step.id} className="flow-step">
                <div className="step-connector">
                  <div className="step-dot" />
                  {idx < dataFlow.length - 1 && <div className="step-line" />}
                </div>
                <div className="step-content">
                  <div className="step-icon">{step.icon}</div>
                  <div className="step-info">
                    <span className="step-name">{step.name}</span>
                    <span className="step-desc">{step.description}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="attention-section">
        <div 
          className={`section-header ${expandedSections.has('attention') ? 'expanded' : ''}`}
          onClick={() => toggleSection('attention')}
        >
          <span className="expand-icon">{expandedSections.has('attention') ? '▼' : '▶'}</span>
          <span>🎯 Self-Attention Explained</span>
        </div>
        
        {expandedSections.has('attention') && (
          <div className="attention-explainer">
            <div className="concept">
              <strong>Query, Key, Value (Q, K, V)</strong>
              <p>Each token creates three vectors. Query asks "what am I looking for?", Key says "what do I contain?", Value holds "what information do I provide?"</p>
            </div>
            <div className="concept">
              <strong>Attention Scores</strong>
              <p>Computed as softmax(QK^T / √d). High scores mean strong relationships between tokens.</p>
            </div>
            <div className="concept">
              <strong>Multi-Head Attention</strong>
              <p>Multiple attention "heads" look at different aspects (syntax, semantics, etc.) simultaneously.</p>
            </div>
            <div className="formula">
              <code>Attention(Q,K,V) = softmax(QK^T/√d_k)V</code>
            </div>
          </div>
        )}
      </div>

      <div className="layers-section">
        <div 
          className={`section-header ${expandedSections.has('layers') ? 'expanded' : ''}`}
          onClick={() => toggleSection('layers')}
        >
          <span className="expand-icon">{expandedSections.has('layers') ? '▼' : '▶'}</span>
          <span>🔄 Transformer Layers</span>
          <span className="step-count">12 layers</span>
        </div>
        
        {expandedSections.has('layers') && (
          <div className="layer-list">
            {Array.from({ length: 12 }, (_, i) => (
              <div 
                key={i}
                className={`layer-item ${selectedLayer === i ? 'selected' : ''}`}
                onClick={() => onLayerSelect?.(i)}
              >
                <div className="layer-icon">L{i + 1}</div>
                <div className="layer-info">
                  <span className="layer-name">Transformer Block {i + 1}</span>
                  <span className="layer-components">
                    Self-Attention → LayerNorm → FFN → LayerNorm
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {architecture && (
        <div className="params-summary">
          <div className="param-item">
            <span className="param-label">Total Parameters</span>
            <span className="param-value">{formatParams(architecture.total_params)}</span>
          </div>
        </div>
      )}

      <div className="arch-legend">
        <h4>🎨 Color Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#43e97b' }} />
            <span>High attention</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#1a3a1a' }} />
            <span>Low attention</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#ff6b6b' }} />
            <span>Negative embedding</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TransformerArchitecture

