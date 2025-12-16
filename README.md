# Trinetr - CNN Visualization Platform

A browser-based CNN visualization platform for image and video models.

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

- Upload and visualize CNN models
- Image and video model support
- Layer activation visualization
- Feature map visualization
- Gradient-based visualizations (Grad-CAM, saliency maps)

