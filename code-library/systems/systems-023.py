#!/usr/bin/env python3
"""
tensorflow_attachment.py
-------------------------------------------------
A light‑weight TensorFlow/Keras wrapper that can be
plugged into the fibre‑optic QA pipeline to
 * train a simple image‑classification network
   (default: Fashion‑MNIST example from transcript)
 * save / load the model as an .h5 file
 * perform single‑image or batch predictions
 * expose a tiny REST‑style API to other modules
     - build_model()     -> returns untrained model
     - train(data_dir)   -> trains and saves a .h5
     - load(model_path)  -> returns a ready model
     - predict(img_arr)  -> returns class + prob
-------------------------------------------------
The baseline topology literally matches the
"TensorFlow in 100 seconds" video:
    Flatten -> Dense(128, ReLU) -> Dense(10, softmax)
but you can pass a list of extra Keras layers
via build_model(..., extra_layers=[...]) if you
decide to grow the network later.
"""

import os
import pathlib
import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# 1. Top‑level helpers
# ----------------------------------------------------------------------

CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9"
]

DEFAULT_MODEL_PATH = pathlib.Path("models") / "tf_classifier_v1.h5"


def build_model(input_shape: Tuple[int, int] = (28, 28),
                num_classes: int = 10,
                extra_layers: List[layers.Layer] = None) -> keras.Model:
    """Return a compiled Keras model following the transcript recipe."""
    model = keras.Sequential(name="simple_mlp")
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation="relu"))
    if extra_layers:
        for lyr in extra_layers:
            model.add(lyr)
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def _prepare_fashion_mnist(split: str = "train",
                           batch: int = 64) -> tf.data.Dataset:
    """Utility loader for the Fashion‑MNIST showcase."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    if split == "train":
        ds = tf.data.Dataset.from_tensor_slices(
            (x_train.astype("float32") / 255.0, y_train)
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (x_test.astype("float32") / 255.0, y_test)
        )
    return ds.shuffle(10_000).batch(batch)


def train(output_path: str | pathlib.Path = DEFAULT_MODEL_PATH,
          epochs: int = 10,
          batch: int = 64) -> pathlib.Path:
    """
    Train on Fashion‑MNIST (demonstration) or swap for your own dataset
    by editing _prepare_fashion_mnist.

    Returns the path to the saved model file.
    """
    tic = time.time()
    model = build_model()
    ds_train = _prepare_fashion_mnist("train", batch)
    ds_val = _prepare_fashion_mnist("test", batch)

    model.fit(ds_train,
              validation_data=ds_val,
              epochs=epochs,
              verbose=2)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path, include_optimizer=False)
    print(f"✓ Model saved to {output_path} ({time.time() - tic:.1f}s)")
    return output_path


def load(model_path: str | pathlib.Path = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load and return a previously‑saved .h5 model."""
    return keras.models.load_model(model_path)


def predict(img: np.ndarray,
            model: keras.Model | None = None,
            model_path: str | pathlib.Path = DEFAULT_MODEL_PATH,
            return_prob: bool = False) -> Tuple[str, float] | str:
    """
    Predict the class of a single greyscale 28 × 28 image (0‑1 float).

    If *model* is None the saved model file is loaded on‑demand.
    """
    if model is None:
        model = load(model_path)

    if img.ndim == 2:               # (H, W)
        img = img[None, ..., None]  # -> (1, H, W, 1)
    elif img.ndim == 3 and img.shape[-1] in (1, 3):
        img = img[None, ...]
    else:
        raise ValueError("Image should be [28×28] or [H×W×1/3]")

    pred = model(img, training=False).numpy()[0]
    idx = int(np.argmax(pred))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)

    return (label, float(pred[idx])) if return_prob else label


# ----------------------------------------------------------------------
# 2. CLI convenience
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or use the simple TensorFlow classifier."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch", type=int, default=64)
    p_train.add_argument("--out", default=str(DEFAULT_MODEL_PATH))

    # predict one file
    p_pred = sub.add_parser("predict", help="Predict a single 28×28 image")
    p_pred.add_argument("image_path", help="Path to greyscale PNG/JPG")
    p_pred.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p_pred.add_argument("--prob", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.out, epochs=args.epochs, batch=args.batch)

    elif args.cmd == "predict":
        import cv2
        img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) / 255.0
        out = predict(img, model_path=args.model, return_prob=args.prob)
        print(out)