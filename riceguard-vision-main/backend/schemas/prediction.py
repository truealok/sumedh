from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    status: str = Field(default="success", description="API status")
    disease: str = Field(..., description="Predicted rice disease")
    confidence: float = Field(..., description="Prediction confidence percentage")
    message: str = Field(default="Prediction completed successfully.", description="Human-readable message")


class ModelInfoResponse(BaseModel):
    model_name: str
    input_size: int
    number_of_classes: int
    framework: str
    version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
