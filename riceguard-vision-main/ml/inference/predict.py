import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ML_DIR / "models" / "rice_model.keras"


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def load_class_names(model_path: Path) -> list[str]:
    classes_path = Path(model_path).parent / "classes.json"
    if classes_path.exists():
        with classes_path.open("r", encoding="utf-8") as handle:
            classes = json.load(handle)
            if isinstance(classes, list) and classes:
                return classes
    return ["Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast"]


def preprocess_image(image_path: str, input_size: int = 224) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(image_path: str, model_path: str = str(DEFAULT_MODEL_PATH)) -> dict:
    model = load_model(model_path)
    class_names = load_class_names(Path(model_path))
    processed = preprocess_image(image_path)
    probabilities = model.predict(processed, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {
        "disease": class_names[index],
        "confidence": round(float(probabilities[index] * 100), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rice disease from an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the trained Keras model")
    args = parser.parse_args()

    result = predict(args.image_path, args.model)
    print(result)
