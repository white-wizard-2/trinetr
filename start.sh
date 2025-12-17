#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.conda
# Start the backend
(cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload) &
BACKEND_PID=$!

# Start the frontend
(cd frontend && npm run dev) &
FRONTEND_PID=$!

# Trap EXIT (including ctrl+c) and kill both backend and frontend
cleanup() {
    echo "Stopping processes..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
}
trap cleanup EXIT

# Wait for both processes to exit (will return when both are killed)
wait $BACKEND_PID
wait $FRONTEND_PID
