import { useState, useEffect } from 'react'
import ModelLoader from './components/ModelLoader'
import ImageUploader from './components/ImageUploader'
import ModelArchitecture from './components/ModelArchitecture'
import ActivationVisualizer from './components/ActivationVisualizer'
import PredictionViewer from './components/PredictionViewer'
import './App.css'

function App() {
  const [modelId, setModelId] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [activations, setActivations] = useState<any>(null)

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

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <ModelLoader onModelLoaded={setModelId} />
        </div>
        <div className="header-center">
          <h1>Trinetr</h1>
          <p>CNN Visualization Platform - Learn how CNNs process images</p>
        </div>
        <div className="header-right">
          <div className="image-upload-section">
            <h3>Upload Image</h3>
            <ImageUploader onImageUpload={setImageFile} imageFile={imageFile} />
          </div>
        </div>
      </header>
      
      <main className="app-main">
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
                <p>Load a model from the header to begin exploring CNN architectures!</p>
              </div>
            )}
          </div>
          
          <div className="visualization-panel">
            <div className="panel-tabs">
              <div className="tab-content">
                {imageFile && selectedLayer && modelId ? (
                  <ActivationVisualizer
                    modelId={modelId}
                    layerName={selectedLayer}
                    imageFile={imageFile}
                    onActivationsLoaded={setActivations}
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
      </main>
    </div>
  )
}

export default App

