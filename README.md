# Trinetr - Neural Network Visualization Platform

An interactive visualization platform for understanding deep learning models - from CNNs to Transformers.

> *Trinetr (त्रिनेत्र)* - The third eye that sees beyond the surface, into the inner workings of neural networks.

## What is this?

A tool to visualize and understand how neural networks work internally. Watch data flow through layers, see attention patterns, explore embeddings, and understand what your model is actually learning.

## Features

### 🖼️ CNN Visualization
- **Model Architecture**: Interactive layer-by-layer view of CNN structure
- **Image Preprocessing**: Real-time kernel visualization (blur, sharpen, edge detection)
- **Custom Kernels**: Modify convolution kernels and see immediate effects
- **Activation Maps**: Visualize feature maps at each layer
- **Supported Models**: VGG16, VGG19, ResNet18, ResNet50, DenseNet121

### 🤖 Transformer Visualization
- **Text Models**: GPT-2, BERT - see how language models process text
- **Vision Models**: ViT (Vision Transformer) - understand image classification
- **Attention Visualization**: 
  - Multi-head attention matrices with real values
  - Q, K, V weight matrices and their computation
  - Layer-by-layer attention flow
- **QKV Deep Dive**:
  - Actual Q, K, V values for each token/patch
  - Statistics (mean, std, min, max)
  - Scrollable value tables
- **KV Cache**: Shape, size, and memory usage visualization
- **Token Embeddings**: See how inputs become vectors
- **Next Token Prediction**: Step-by-step generation process

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ CNN Workspace│  │Transformer  │  │ Model Loader   │  │
│  │             │  │ Workspace   │  │                │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ PyTorch     │  │ HuggingFace │  │ Image Processing│  │
│  │ CNN Models  │  │ Transformers│  │ & Kernels       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

- **Backend**: Python FastAPI + PyTorch + HuggingFace Transformers
- **Frontend**: TypeScript + React + Vite

## Quick Start

```bash
# Clone and run
git clone <repo>
cd trinetr
./start.sh
```

The script will:
1. Set up Python environment
2. Install dependencies
3. Build the frontend
4. Start the server at http://localhost:8000

## Manual Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (for development)
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. **Select Model Type**: Choose CNN or Transformer from the dropdown
2. **Load a Model**: Pick from available pretrained models
3. **Provide Input**: Upload an image (CNN/ViT) or enter text (GPT-2/BERT)
4. **Explore**: Navigate through tabs to see different visualizations

### CNN Mode
- View model architecture on the left
- Use Image Lab to preprocess and apply kernels
- Run inference to see activations and predictions

### Transformer Mode
- **Input Tab**: See tokenization and embedding process
- **Attention Tab**: Explore attention patterns, QKV values, layer flow
- **Embeddings Tab**: Visualize token representations
- **Generation Tab**: Watch step-by-step token generation
- **Output Tab**: See final predictions with confidence scores

## Requirements

- Python 3.10+
- Node.js 18+
- 4GB+ RAM (8GB recommended for larger models)

## License

Apache 2.0
