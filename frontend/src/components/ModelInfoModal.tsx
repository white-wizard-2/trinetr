import { useEffect, useRef } from 'react'
import './ModelInfoModal.css'

interface ModelInfoModalProps {
  modelName: string
  onClose: () => void
}

function ModelInfoModal({ modelName, onClose }: ModelInfoModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', handleEscape)
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [onClose])

  const getModelInfo = () => {
    switch (modelName) {
      case 'resnet18':
        return {
          name: 'ResNet-18',
          fullName: 'Residual Network (18 layers)',
          year: 2015,
          authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
          description: 'ResNet-18 is a deep convolutional neural network that introduced the revolutionary "skip connections" or "residual connections" concept, solving the vanishing gradient problem in very deep networks.',
          architecture: {
            input: '224×224×3 RGB image',
            layers: [
              { name: 'Conv1', type: 'Convolution', details: '7×7 conv, 64 filters, stride 2, padding 3 → 112×112×64' },
              { name: 'MaxPool', type: 'Pooling', details: '3×3 max pool, stride 2 → 56×56×64' },
              { name: 'Layer1', type: 'Residual Block', details: '2× BasicBlock (2× 3×3 conv) → 56×56×64' },
              { name: 'Layer2', type: 'Residual Block', details: '2× BasicBlock, stride 2 → 28×28×128' },
              { name: 'Layer3', type: 'Residual Block', details: '2× BasicBlock, stride 2 → 14×14×256' },
              { name: 'Layer4', type: 'Residual Block', details: '2× BasicBlock, stride 2 → 7×7×512' },
              { name: 'AvgPool', type: 'Pooling', details: 'Global average pool → 1×1×512' },
              { name: 'FC', type: 'Fully Connected', details: '512 → 1000 classes' }
            ],
            totalParams: '~11.7 million',
            keyInnovation: 'Residual Connections (Skip Connections)'
          },
          dataFlow: [
            'Input Image (224×224×3)',
            '↓ Conv1: Extract low-level features (edges, textures)',
            '↓ MaxPool: Reduce spatial dimensions',
            '↓ Layer1: Basic features, maintain resolution',
            '↓ Layer2: More complex patterns, 2× downsampling',
            '↓ Layer3: High-level features, 2× downsampling',
            '↓ Layer4: Very high-level features, 2× downsampling',
            '↓ Global AvgPool: Aggregate spatial information',
            '↓ Fully Connected: Classify into 1000 categories',
            'Output: Class probabilities'
          ],
          operations: {
            convolution: {
              title: 'Convolution Operation',
              description: 'A convolution applies a filter (kernel) to the input image, sliding it across to detect features.',
              steps: [
                '1. Filter (kernel) slides over input feature map',
                '2. Element-wise multiplication at each position',
                '3. Sum all products to get single output value',
                '4. Repeat for all positions → output feature map',
                '5. Multiple filters → multiple feature maps (channels)'
              ],
              example: '3×3 conv with stride 1: Input 224×224 → Output 224×224 (with padding)',
              purpose: 'Detects local patterns: edges, textures, shapes'
            },
            residual: {
              title: 'Residual Connection (Skip Connection)',
              description: 'The key innovation of ResNet - allows gradients to flow directly through "shortcut" connections.',
              structure: 'Output = F(x) + x (where F(x) is the learned transformation, x is the input)',
              benefits: [
                'Solves vanishing gradient problem',
                'Enables training of very deep networks (100+ layers)',
                'Allows identity mapping when optimal',
                'Easier optimization'
              ],
              diagram: 'Input → [Conv → BN → ReLU → Conv → BN] → Add(Input) → ReLU → Output'
            },
            pooling: {
              title: 'Pooling Operation',
              description: 'Reduces spatial dimensions while preserving important features.',
              types: [
                'Max Pooling: Takes maximum value in each window (most common)',
                'Average Pooling: Takes average value in each window',
                'Global Average Pooling: Averages entire feature map → single value per channel'
              ],
              purpose: 'Reduces computation, increases receptive field, provides translation invariance'
            }
          },
          learningPoints: [
            'Residual connections allow information to flow directly, making deep networks trainable',
            'Each residual block learns a "residual" (difference) rather than the full transformation',
            'The network can learn identity mappings when needed (F(x) = 0, so output = x)',
            '18 layers refers to learnable layers (conv + FC), not total layers including pooling/activation'
          ]
        }
      case 'resnet50':
        return {
          name: 'ResNet-50',
          fullName: 'Residual Network (50 layers)',
          year: 2015,
          authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
          description: 'ResNet-50 is a deeper variant of ResNet with bottleneck blocks, using 1×1 convolutions to reduce and expand dimensions efficiently.',
          architecture: {
            input: '224×224×3 RGB image',
            layers: [
              { name: 'Conv1', type: 'Convolution', details: '7×7 conv, 64 filters, stride 2 → 112×112×64' },
              { name: 'MaxPool', type: 'Pooling', details: '3×3 max pool, stride 2 → 56×56×64' },
              { name: 'Layer1', type: 'Bottleneck Block', details: '3× BottleneckBlock → 56×56×256' },
              { name: 'Layer2', type: 'Bottleneck Block', details: '4× BottleneckBlock, stride 2 → 28×28×512' },
              { name: 'Layer3', type: 'Bottleneck Block', details: '6× BottleneckBlock, stride 2 → 14×14×1024' },
              { name: 'Layer4', type: 'Bottleneck Block', details: '3× BottleneckBlock, stride 2 → 7×7×2048' },
              { name: 'AvgPool', type: 'Pooling', details: 'Global average pool → 1×1×2048' },
              { name: 'FC', type: 'Fully Connected', details: '2048 → 1000 classes' }
            ],
            totalParams: '~25.6 million',
            keyInnovation: 'Bottleneck Architecture + Residual Connections'
          },
          dataFlow: [
            'Input Image (224×224×3)',
            '↓ Conv1: Initial feature extraction',
            '↓ MaxPool: First downsampling',
            '↓ Layer1: Bottleneck blocks, 256 channels',
            '↓ Layer2: Deeper features, 512 channels, 2× downsampling',
            '↓ Layer3: High-level features, 1024 channels, 2× downsampling',
            '↓ Layer4: Very high-level features, 2048 channels, 2× downsampling',
            '↓ Global AvgPool: Spatial aggregation',
            '↓ Fully Connected: Classification',
            'Output: 1000 class probabilities'
          ],
          operations: {
            bottleneck: {
              title: 'Bottleneck Block',
              description: 'Efficient block design using 1×1 convolutions to reduce and expand dimensions.',
              structure: '1×1 conv (reduce) → 3×3 conv (feature extraction) → 1×1 conv (expand)',
              example: 'Input 256 channels → 64 channels → 64 channels → 256 channels',
              benefits: [
                'Reduces computation by 4× compared to standard 3×3 conv blocks',
                'More efficient use of parameters',
                'Enables deeper networks with same computational budget'
              ],
              diagram: '256ch → [1×1:64ch] → [3×3:64ch] → [1×1:256ch] → Add(256ch) → Output'
            },
            convolution: {
              title: 'Convolution Operation',
              description: 'Same as ResNet-18, but used within bottleneck blocks for efficiency.',
              steps: [
                '1×1 conv: Reduces channels (dimensionality reduction)',
                '3×3 conv: Extracts spatial features (main operation)',
                '1×1 conv: Expands channels back (dimensionality expansion)'
              ],
              purpose: 'Efficiently extracts features while reducing computational cost'
            },
            residual: {
              title: 'Residual Connection',
              description: 'Same skip connection mechanism as ResNet-18, but applied to bottleneck blocks.',
              structure: 'Output = BottleneckBlock(x) + x',
              benefits: [
                'Maintains gradient flow in deeper network',
                'Allows identity learning',
                'Enables 50+ layer training'
              ]
            }
          },
          learningPoints: [
            'Bottleneck design makes ResNet-50 more efficient than ResNet-18 despite being deeper',
            '1×1 convolutions act as "dimensionality reduction" - reducing channels before expensive 3×3 conv',
            'The network progressively increases channels (64 → 256 → 512 → 1024 → 2048)',
            'More parameters (~25M) than ResNet-18 but more accurate on complex tasks'
          ]
        }
      case 'vgg16':
        return {
          name: 'VGG-16',
          fullName: 'Visual Geometry Group (16 layers)',
          year: 2014,
          authors: 'Karen Simonyan, Andrew Zisserman',
          description: 'VGG-16 is a classic CNN architecture known for its simplicity - using only 3×3 convolutions and 2×2 max pooling throughout. It demonstrated that depth is crucial for performance.',
          architecture: {
            input: '224×224×3 RGB image',
            layers: [
              { name: 'Conv Block 1', type: 'Convolution', details: '2× (3×3 conv, 64 filters) → 224×224×64' },
              { name: 'MaxPool', type: 'Pooling', details: '2×2 max pool, stride 2 → 112×112×64' },
              { name: 'Conv Block 2', type: 'Convolution', details: '2× (3×3 conv, 128 filters) → 112×112×128' },
              { name: 'MaxPool', type: 'Pooling', details: '2×2 max pool, stride 2 → 56×56×128' },
              { name: 'Conv Block 3', type: 'Convolution', details: '3× (3×3 conv, 256 filters) → 56×56×256' },
              { name: 'MaxPool', type: 'Pooling', details: '2×2 max pool, stride 2 → 28×28×256' },
              { name: 'Conv Block 4', type: 'Convolution', details: '3× (3×3 conv, 512 filters) → 28×28×512' },
              { name: 'MaxPool', type: 'Pooling', details: '2×2 max pool, stride 2 → 14×14×512' },
              { name: 'Conv Block 5', type: 'Convolution', details: '3× (3×3 conv, 512 filters) → 14×14×512' },
              { name: 'MaxPool', type: 'Pooling', details: '2×2 max pool, stride 2 → 7×7×512' },
              { name: 'FC1', type: 'Fully Connected', details: '25088 → 4096' },
              { name: 'FC2', type: 'Fully Connected', details: '4096 → 4096' },
              { name: 'FC3', type: 'Fully Connected', details: '4096 → 1000 classes' }
            ],
            totalParams: '~138 million',
            keyInnovation: 'Uniform Architecture with Small Filters'
          },
          dataFlow: [
            'Input Image (224×224×3)',
            '↓ Conv Block 1: 64 channels, detect basic features',
            '↓ MaxPool: 112×112×64',
            '↓ Conv Block 2: 128 channels, more complex patterns',
            '↓ MaxPool: 56×56×128',
            '↓ Conv Block 3: 256 channels, high-level features',
            '↓ MaxPool: 28×28×256',
            '↓ Conv Block 4: 512 channels, very high-level features',
            '↓ MaxPool: 14×14×512',
            '↓ Conv Block 5: 512 channels, final feature extraction',
            '↓ MaxPool: 7×7×512',
            '↓ Flatten: 25088 features',
            '↓ FC1: 4096 neurons',
            '↓ FC2: 4096 neurons',
            '↓ FC3: 1000 classes',
            'Output: Class probabilities'
          ],
          operations: {
            convolution: {
              title: '3×3 Convolution Strategy',
              description: 'VGG uses only small 3×3 filters instead of larger filters (like 5×5 or 7×7).',
              insight: 'Two 3×3 convs have same receptive field as one 5×5 conv, but with fewer parameters!',
              benefits: [
                'More non-linearities (ReLU after each conv)',
                'Fewer parameters than larger filters',
                'Easier to train',
                'More expressive power'
              ],
              comparison: '5×5 conv ≈ 25 params vs 2× (3×3 conv) ≈ 18 params for same receptive field',
              purpose: 'Detects features at multiple scales through depth'
            },
            pooling: {
              title: 'Max Pooling',
              description: 'VGG uses 2×2 max pooling with stride 2 after each conv block.',
              effect: 'Reduces spatial dimensions by 2×, doubles effective receptive field',
              purpose: 'Provides translation invariance, reduces computation',
              pattern: 'Applied 5 times: 224 → 112 → 56 → 28 → 14 → 7'
            },
            fullyConnected: {
              title: 'Fully Connected Layers',
              description: 'Three large FC layers at the end for classification.',
              structure: '7×7×512 = 25088 features → 4096 → 4096 → 1000',
              purpose: 'Aggregates spatial features into class predictions',
              note: 'These layers contain most of the parameters (~124M of 138M total)'
            }
          },
          learningPoints: [
            'Small filters (3×3) are more efficient than large filters when stacked',
            'Depth is crucial - VGG showed that deeper networks perform better',
            'Uniform architecture makes VGG easy to understand and modify',
            'Most parameters are in FC layers (90%+), not convolutional layers',
            'VGG-16 has 5 max pooling operations, reducing image from 224×224 to 7×7'
          ]
        }
      default:
        return null
    }
  }

  const info = getModelInfo()
  if (!info) return null

  return (
    <div className="model-info-overlay" onClick={onClose} ref={modalRef}>
      <div className="model-info-content" onClick={(e) => e.stopPropagation()}>
        <div className="model-info-header">
          <div>
            <h2>{info.name}</h2>
            <p className="model-subtitle">{info.fullName} • {info.year} • {info.authors}</p>
          </div>
          <button className="model-info-close" onClick={onClose}>✕</button>
        </div>

        <div className="model-info-body">
          <div className="info-section">
            <h3>Overview</h3>
            <p>{info.description}</p>
            <div className="model-stats">
              <div className="stat-item">
                <strong>Total Parameters:</strong> {info.architecture.totalParams}
              </div>
              <div className="stat-item">
                <strong>Key Innovation:</strong> {info.architecture.keyInnovation}
              </div>
            </div>
          </div>

          <div className="info-section">
            <h3>Architecture & Data Flow</h3>
            <div className="data-flow-diagram">
              <div className="flow-item input">Input: {info.architecture.input}</div>
              {info.dataFlow.map((step, idx) => (
                <div key={idx} className="flow-item">
                  {step}
                </div>
              ))}
            </div>
          </div>

          <div className="info-section">
            <h3>Layer Details</h3>
            <div className="layers-table">
              {info.architecture.layers.map((layer, idx) => (
                <div key={idx} className="layer-row">
                  <div className="layer-name">{layer.name}</div>
                  <div className="layer-type">{layer.type}</div>
                  <div className="layer-details">{layer.details}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="info-section">
            <h3>Key Operations Explained</h3>
            {Object.entries(info.operations).map(([key, op]: [string, any]) => (
              <div key={key} className="operation-detail">
                <h4>{op.title}</h4>
                <p>{op.description}</p>
                {op.steps && (
                  <ul>
                    {op.steps.map((step: string, idx: number) => (
                      <li key={idx}>{step}</li>
                    ))}
                  </ul>
                )}
                {op.structure && (
                  <div className="operation-structure">
                    <strong>Structure:</strong> <code>{op.structure}</code>
                  </div>
                )}
                {op.diagram && (
                  <div className="operation-diagram">
                    <strong>Flow:</strong> <code>{op.diagram}</code>
                  </div>
                )}
                {op.benefits && (
                  <ul>
                    {op.benefits.map((benefit: string, idx: number) => (
                      <li key={idx}>{benefit}</li>
                    ))}
                  </ul>
                )}
                {op.example && (
                  <div className="operation-example">
                    <strong>Example:</strong> {op.example}
                  </div>
                )}
                {op.comparison && (
                  <div className="operation-comparison">
                    <strong>Comparison:</strong> {op.comparison}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="info-section">
            <h3>Key Learning Points</h3>
            <ul className="learning-points">
              {info.learningPoints.map((point, idx) => (
                <li key={idx}>{point}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelInfoModal

