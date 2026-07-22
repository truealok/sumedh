import logging
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.core.config import settings
from backend.core.exceptions import InvalidImageError, ModelUnavailableError
from backend.schemas.prediction import HealthResponse, ModelInfoResponse, PredictionResponse
from backend.services.database_service import database_service
from backend.services.model_service import model_service
from backend.utils.file_utils import build_upload_path, ensure_upload_directory, is_supported_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=Dict[str, str], tags=["Health"])
async def root() -> Dict[str, str]:
    return {"message": "RiceGuard AI API is running"}


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_service.model_loaded)


@router.get("/classes", tags=["Prediction"])
async def get_classes() -> Dict[str, Any]:
    return {"classes": model_service.classes}


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info() -> ModelInfoResponse:
    return model_service.get_model_info()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload a PNG, JPG, or JPEG image.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise InvalidImageError("Uploaded file is empty.")
        if len(contents) > settings.max_upload_size:
            raise InvalidImageError("File exceeds the maximum allowed size.")

        upload_dir = ensure_upload_directory(settings.upload_folder)
        upload_path = build_upload_path(settings.upload_folder, file.filename)
        with upload_path.open("wb") as handle:
            handle.write(contents)

        if not model_service.model_loaded:
            raise ModelUnavailableError("Train the Deep Learning model first.")

        prediction_result = model_service.predict(str(upload_path))
        response_payload = PredictionResponse(
            status="success",
            disease=prediction_result["disease"],
            confidence=prediction_result["confidence"],
            message="Prediction completed successfully.",
        )

        database_service.store_prediction(
            {
                "disease": response_payload.disease,
                "confidence": response_payload.confidence,
                "image_name": file.filename,
            }
        )
        return response_payload
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        return PredictionResponse(
            status="Model not trained",
            disease="Not available",
            confidence=0.0,
            message="Train the Deep Learning model first.",
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed due to an internal error.") from exc
