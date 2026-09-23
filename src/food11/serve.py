"""Serve the champion Food-11 model through a FastAPI application."""

from contextlib import asynccontextmanager
from io import BytesIO
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
import mlflow
import mlflow.pyfunc
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


CLASS_NAMES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)
MODEL_URI = "models:/food11@champion"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
IMAGE_SIZE = (128, 128)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registered model once when the API process starts."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(title="Food-11 API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def preprocess_image(contents: bytes) -> np.ndarray:
    """Convert an uploaded image into a normalized NCHW model input."""
    try:
        with Image.open(BytesIO(contents)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from error

    normalized = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str | float]:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    logits = np.asarray(app.state.model.predict(preprocess_image(contents)))
    if logits.shape != (1, len(CLASS_NAMES)):
        raise HTTPException(status_code=500, detail="Model returned an unexpected output shape")

    shifted = logits[0] - np.max(logits[0])
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    predicted_index = int(np.argmax(probabilities))
    return {
        "category": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }
