import { useState, useEffect } from 'react'
import ModelLoader from './components/ModelLoader'
import ImageUploader from './components/ImageUploader'
import ModelArchitecture from './components/ModelArchitecture'
import ImageLab from './components/ImageLab'
import ActivationVisualizer from './components/ActivationVisualizer'
import PredictionViewer from './components/PredictionViewer'
import TransformerLoader from './components/TransformerLoader'
import TransformerWorkspace from './components/TransformerWorkspace'
import DiffusionLoader from './components/DiffusionLoader'
import DiffusionWorkspace from './components/DiffusionWorkspace'
import './App.css'

type ModelType = 'cnn' | 'transformer' | 'diffusion'

function App() {
  const [modelType, setModelType] = useState<ModelType>('cnn')
  const [modelId, setModelId] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  
  // Transformer state
  const [transformerModelId, setTransformerModelId] = useState<string | null>(null)
  const [transformerModelName, setTransformerModelName] = useState<string>('bert-base')
  const [transformerType, setTransformerType] = useState<'text' | 'image'>('text')
  
  // Diffusion state
  const [diffusionModelId, setDiffusionModelId] = useState<string | null>(null)
  const [diffusionModelName, setDiffusionModelName] = useState<string>('stable-diffusion-v1-4')

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

  // Reset state when switching model types
  const handleModelTypeChange = (type: ModelType) => {
    setModelType(type)
    setModelId(null)
    setTransformerModelId(null)
    setDiffusionModelId(null)
    setSelectedLayer(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-brand">
          <h1>Trinetr</h1>
          <span className="tagline">Vision AI Visualization</span>
        </div>
        <div className="header-controls">
          <div className="model-type-selector">
            <button 
              className={`type-btn ${modelType === 'cnn' ? 'active' : ''}`}
              onClick={() => handleModelTypeChange('cnn')}
            >
              🔲 CNN
            </button>
            <button 
              className={`type-btn ${modelType === 'transformer' ? 'active' : ''}`}
              onClick={() => handleModelTypeChange('transformer')}
            >
              🔀 Transformer
            </button>
            <button 
              className={`type-btn ${modelType === 'diffusion' ? 'active' : ''}`}
              onClick={() => handleModelTypeChange('diffusion')}
            >
              🎨 Diffusion
            </button>
          </div>
          <div className="header-divider" />
          {modelType === 'cnn' ? (
            <>
              <ModelLoader onModelLoaded={setModelId} />
              <div className="header-divider" />
              <ImageUploader onImageUpload={setImageFile} imageFile={imageFile} />
            </>
          ) : modelType === 'transformer' ? (
            <TransformerLoader 
              onModelLoaded={(id, name) => {
                setTransformerModelId(id)
                setTransformerModelName(name)
              }}
              onTypeChange={setTransformerType}
              transformerType={transformerType}
            />
          ) : (
            <DiffusionLoader 
              onModelLoaded={(id, name) => {
                setDiffusionModelId(id)
                setDiffusionModelName(name)
              }}
            />
          )}
        </div>
      </header>
      
      <main className="app-main">
        {modelType === 'cnn' ? (
          <div className="main-content">
            <div className="architecture-panel">
              {modelId ? (
                <ModelArchitecture
                  modelId={modelId}
                  selectedLayer={selectedLayer}
                  onLayerSelect={setSelectedLayer}
                />
              ) : (
                <div className="info-box">
                  <h3>🎓 Get Started</h3>
                  <p>Load a model from the header to begin exploring Vision AI architectures!</p>
                </div>
              )}
            </div>
            
            <div className="visualization-panel">
              <div className="panel-tabs">
                <div className="tab-content">
                  <ImageLab imageFile={imageFile} modelId={modelId} />
                </div>
                
                <div className="tab-content">
                  {imageFile && selectedLayer && modelId ? (
                    <ActivationVisualizer
                      modelId={modelId}
                      layerName={selectedLayer}
                      imageFile={imageFile}
                    />
                  ) : (
                    <div className="info-box">
                      <h3>📊 Layer Activations</h3>
                      <p>Select a layer from the architecture view to see how it processes the image!</p>
                    </div>
                  )}
                </div>
                
                <div className="tab-content">
                  {imageFile && modelId ? (
                    <PredictionViewer
                      modelId={modelId}
                      imageFile={imageFile}
                    />
                  ) : (
                    <div className="info-box">
                      <h3>🎯 Predictions</h3>
                      <p>Load a model and upload an image to see final layer predictions!</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : modelType === 'transformer' ? (
          <TransformerWorkspace 
            modelId={transformerModelId}
            modelName={transformerModelName}
            transformerType={transformerType}
            imageFile={imageFile}
            onImageUpload={setImageFile}
          />
        ) : (
          <DiffusionWorkspace 
            modelId={diffusionModelId}
            modelName={diffusionModelName}
          />
        )}
      </main>
    </div>
  )
}

export default App

