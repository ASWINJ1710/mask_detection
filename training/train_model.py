"""Train the mask classifier (MobileNetV2 transfer learning, frozen base).

Run from the project root:
    python training\train_model.py
The test set is evaluated exactly once, at the end.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODEL_PATH = MODELS / "face_mask_model.keras"

CLASS_NAMES = ["with_mask", "without_mask"]  # index 0, 1 -> sigmoid = P(without_mask)
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
EPOCHS = 15
LR = 1e-3
THRESHOLD = 0.5

keras.utils.set_random_seed(SEED)
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def load(split, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        str(DATA / split),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,), name="rgb_0_255")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.15, value_range=(0, 255))(x)
    # Same scaling as mobilenet_v2.preprocess_input: pixels -> [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_minus1_1")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_without_mask")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    train_ds = load("train", shuffle=True)
    val_ds = load("validation", shuffle=False)
    test_ds = load("test", shuffle=False)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks)
    plot_history(history)
    (RESULTS / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()},
                   indent=2),
        encoding="utf-8",
    )

    # ---- Final, one-time test evaluation using the saved best model ----
    best = keras.models.load_model(str(MODEL_PATH))
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    probs = best.predict(test_ds).ravel()
    y_pred = (probs >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    plot_confusion(cm)
    metrics = {
        "test_images": int(len(y_true)),
        "threshold": THRESHOLD,
        "class_names": CLASS_NAMES,
        "sigmoid_output_means": "probability of without_mask",
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "report": report,
        "epochs_run": len(history.history["loss"]),
    }
    (RESULTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2),
                                               encoding="utf-8")

    print("\n=== TEST SET RESULTS ===")
    print(f"Images: {len(y_true)}   Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
