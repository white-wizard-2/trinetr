# Trinetr - Vision AI Visualization Platform

A browser-based Vision AI visualization platform for image and video models.

I am trying to make AI generate a visual representation of models. So I can learn from them.

## Architecture

- **Backend**: Python FastAPI (for ML operations and model inference)
- **Frontend**: TypeScript/React (for interactive visualizations)

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Features

- Upload and visualize Vision AI models
- Image and video model support
- Layer activation visualization
- Feature map visualization
- Gradient-based visualizations (Grad-CAM, saliency maps)

