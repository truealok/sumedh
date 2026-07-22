# RiceGuard AI Project Workflow

## Current Project Status

- Frontend and backend are connected through a shared API contract
- Backend routing, validation, and service layers are organized
- ML scripts are separated into training and inference modules
- Documentation and configuration are now grouped by concern
- Dataset and model training remain pending
- Deployment remains pending

## Development Workflow

### 1. Dataset Preparation
- Add the rice disease dataset under the ML dataset folder
- Organize training, validation, and test data separately

### 2. Model Training
- Train the deep learning model in the ML training module
- Save the trained model under the ML models directory
- Generate class labels for inference

### 3. Backend Integration
- Allow the FastAPI backend to load the saved model
- Replace the current development response with real predictions
- Keep the API contract stable for frontend usage

### 4. Frontend Integration
- Continue using the same upload and prediction workflow
- Display real prediction results after model training
- Keep loading and error states active

### 5. Deployment
- Deploy the backend and frontend from their dedicated directories
- Prepare environment variables in the configuration directory
- Validate the full prediction flow end to end
