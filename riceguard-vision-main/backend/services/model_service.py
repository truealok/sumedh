import json
import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from backend.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.classes: List[str] = ["Healthy", "Leaf Blast", "Brown Spot", "Bacterial Blight"]
        self.input_size = 224
        self.framework = "TensorFlow"
        self.version = "1.0.0"
        self.model_loaded = False
        self.model_path = Path(settings.model_path)
        self.classes_path = self.model_path.parent / "classes.json"

    def _load_classes(self) -> None:
        if self.classes_path.exists():
            with self.classes_path.open("r", encoding="utf-8") as handle:
                loaded_classes = json.load(handle)
                if isinstance(loaded_classes, list) and loaded_classes:
                    self.classes = loaded_classes

    def initialize(self) -> None:
        self._load_classes()
        if not self.model_path.exists():
            logger.warning("Model file not found at %s. The API will return a 503 until training is completed.", self.model_path)
            self.model_loaded = False
            return

        try:
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover
            logger.warning("TensorFlow is not installed. Model loading skipped: %s", exc)
            self.model_loaded = False
            return

        try:
            self.model = keras.models.load_model(str(self.model_path))
            self.model_loaded = True
            logger.info("Model loaded successfully from %s", self.model_path)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load model: %s", exc)
            self.model_loaded = False

    def preprocess_image(self, image_path: str):
        if Image is None:
            raise RuntimeError("Pillow is required to preprocess images.")
        if np is None:
            raise RuntimeError("NumPy is required to preprocess images.")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size, self.input_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict(self, image_path: str) -> Dict[str, Any]:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Train the model and place the file at the configured path.")

        processed = self.preprocess_image(image_path)
        prediction = self.model.predict(processed, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index] * 100.0)
        return {
            "disease": self.classes[predicted_index],
            "confidence": round(confidence, 2),
            "probabilities": [round(float(value), 4) for value in prediction],
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_path.name,
            "input_size": self.input_size,
            "number_of_classes": len(self.classes),
            "framework": self.framework,
            "version": self.version,
        }


model_service = ModelService()
