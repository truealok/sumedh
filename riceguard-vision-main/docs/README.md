# RiceGuard AI

RiceGuard AI is now organized into clearly separated frontend, backend, machine learning, documentation, configuration, and asset modules. The project is structured for maintainability while preserving the current prediction workflow and API contract.

## Project Overview

This application provides a foundation for:
- uploading a rice leaf image from the frontend,
- sending the image to a FastAPI backend,
- returning a prediction response from the API layer,
- preparing the codebase for future dataset integration and model training.

> The model is not trained yet, so the prediction endpoint currently returns a development-ready response until a trained model is available.

## Current Structure

```text
backend/
  api/
  core/
  schemas/
  services/
  utils/
frontend/
  public/
  src/
ml/
  dataset/
  inference/
  training/
  checkpoints/
  logs/
  models/
config/
docs/
assets/
```

## Quick Start

### Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- GET /
- GET /health
- GET /classes
- GET /model-info
- POST /predict

## Next Steps

1. Add the dataset under the ML dataset folder.
2. Train the model in the ML training module.
3. Save the trained model under the ML models directory.
4. Let the backend load the model automatically.
5. Replace the current placeholder response with real predictions.
