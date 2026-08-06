import json
import time
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.applications import MobileNetV2
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
for directory in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Accepted folder names per split, so the dataset works whether it is laid out
# as train/valid/test or "Training data"/"Validation data"/"Testing data".
SPLIT_CANDIDATES = {
    "train": ("train", "training", "training data", "train data"),
    "validation": ("valid", "validation", "val", "validation data", "valid data"),
    "test": ("test", "testing", "testing data", "test data"),
}


def resolve_split_dir(split: str) -> Path:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    existing = {entry.name.lower(): entry for entry in DATASET_DIR.iterdir() if entry.is_dir()}
    for candidate in SPLIT_CANDIDATES[split]:
        if candidate in existing:
            return existing[candidate]
    raise FileNotFoundError(
        f"Could not find a '{split}' split under {DATASET_DIR}. "
        f"Expected one of: {', '.join(SPLIT_CANDIDATES[split])}"
    )


TRAIN_DIR = resolve_split_dir("train")
VALIDATION_DIR = resolve_split_dir("validation")
TEST_DIR = resolve_split_dir("test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 10
FINE_TUNE_FROM_LAYER = 100
SEED = 42


def build_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list, dict]:
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    validation_ds = image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    sample_counts = {
        "train": len(train_ds.file_paths),
        "validation": len(validation_ds.file_paths),
        "test": len(test_ds.file_paths),
    }

    # The model (and the backend inference code) expects inputs scaled to [0, 1].
    rescale = layers.Rescaling(1.0 / 255.0)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.055),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomZoom(0.2),
        ],
        name="data_augmentation",
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (rescale(data_augmentation(x, training=True)), y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    validation_ds = validation_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, validation_ds, test_ds, class_names, sample_counts


def build_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    # Inputs arrive in [0, 1] (same contract as the backend inference code);
    # MobileNetV2's ImageNet weights expect [-1, 1].
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_history(history: dict, output_path: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train_accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset, class_names: list) -> dict:
    y_true = np.concatenate([np.argmax(labels, axis=1) for _, labels in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = float(model.evaluate(test_ds, verbose=0)[1])
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def save_metrics(metrics: dict, output_dir: Path) -> None:
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    print(f"Train dir:      {TRAIN_DIR}")
    print(f"Validation dir: {VALIDATION_DIR}")
    print(f"Test dir:       {TEST_DIR}")

    train_ds, validation_ds, test_ds, class_names, sample_counts = build_datasets()
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Samples: {sample_counts}")

    model, base_model = build_model(len(class_names))
    model.summary()

    csv_path = LOG_DIR / "training_log.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Shared across both phases: ModelCheckpoint keeps its running best val_loss,
    # while EarlyStopping/CSVLogger reset per fit() call as intended.
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
        ModelCheckpoint(str(OUTPUT_DIR / "rice_model.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        ModelCheckpoint(
            str(CHECKPOINT_DIR / "epoch_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        CSVLogger(str(csv_path), append=True),
        TensorBoard(log_dir=str(LOG_DIR)),
    ]

    start_time = time.time()
    print("Phase 1: training classifier head (base model frozen)")
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Phase 2: fine-tuning base model from layer {FINE_TUNE_FROM_LAYER}")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_FROM_LAYER]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    last_epoch = len(history.history["loss"])
    history_ft = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=last_epoch + FINE_TUNE_EPOCHS,
        initial_epoch=last_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    training_seconds = time.time() - start_time
    print(f"Training took {training_seconds / 60.0:.1f} minutes")

    merged_history = {
        key: [float(v) for v in history.history.get(key, [])] + [float(v) for v in history_ft.history.get(key, [])]
        for key in set(history.history) | set(history_ft.history)
    }
    with (OUTPUT_DIR / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_history, handle, indent=2)
    plot_history(merged_history, OUTPUT_DIR / "training_history.png")

    # Evaluate the best checkpoint (lowest val_loss across both phases), which is
    # the exact model the backend will serve.
    best_model = keras.models.load_model(OUTPUT_DIR / "rice_model.keras")
    metrics = evaluate_model(best_model, test_ds, class_names)
    metrics["training_seconds"] = round(training_seconds, 1)
    metrics["train_samples"] = sample_counts["train"]
    metrics["validation_samples"] = sample_counts["validation"]
    metrics["test_samples"] = sample_counts["test"]
    save_metrics(metrics, OUTPUT_DIR)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names, OUTPUT_DIR / "confusion_matrix.png")

    with (OUTPUT_DIR / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)
    with (OUTPUT_DIR / "class_indices.json").open("w", encoding="utf-8") as handle:
        json.dump({name: index for index, name in enumerate(class_names)}, handle, indent=2)

    print(f"Test accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall:    {metrics['recall']:.4f}")
    print(f"Test F1 score:  {metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("Training complete. Model saved to", OUTPUT_DIR / "rice_model.keras")


if __name__ == "__main__":
    main()
