"""
FastAPI Inference Service for Bright Future School Image Classification.
Provides RESTful endpoints for image upload, real-time prediction, health checks,
and interactive OpenAPI documentation via /docs.
"""

import io
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bright_future_api")

# Paths and Class Definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "improved_resnet18_best.pt"

CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# High-level school scene taxonomy mapping
SCENE_MAPPING: Dict[str, str] = {
    "airplane": "playground",     # Outdoor / field
    "automobile": "playground",   # Outdoor / parking lot
    "bird": "playground",         # Outdoor nature
    "cat": "classroom",           # Indoor / pet corner
    "deer": "playground",         # Outdoor wildlife
    "dog": "playground",          # Outdoor sports / activity
    "frog": "classroom",          # Science lab / indoor terrarium
    "horse": "playground",        # Outdoor sports ground
    "ship": "playground",         # Outdoor water area
    "truck": "playground",        # Outdoor logistics / field
}

DEVICE = torch.device("cpu")

# Image preprocessing pipeline (matches training evaluation transform)
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_inference_model() -> nn.Module:
    """
    Loads ResNet-18 with adapted classification head and restores fine-tuned weights.
    Falls back to pre-trained weights if custom checkpoint is not available.
    """
    logger.info("Initializing ResNet-18 model architecture...")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))

    if MODEL_PATH.exists():
        logger.info(f"Loading fine-tuned model checkpoint from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded fine-tuned checkpoint weights.")
    else:
        logger.warning(
            f"Checkpoint not found at {MODEL_PATH}. Initializing with standard pre-trained ImageNet weights."
        )
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_dict = base_model.state_dict()
        # Drop fc layer to match adapted 10-class head
        base_dict = {k: v for k, v in base_dict.items() if not k.startswith("fc.")}
        model.load_state_dict(base_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# Global model reference
app_model: nn.Module = load_inference_model()

# Initialize FastAPI Application
app = FastAPI(
    title="Bright Future School - Image Classification API",
    description=(
        "Production RESTful API serving transfer-learning ResNet-18 computer vision classifier "
        "for Bright Future School. Supports image tagging into classroom and playground categories."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Schemas
class TopPrediction(BaseModel):
    class_name: str
    confidence_percentage: float


class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    scene_tag: str
    confidence: float
    confidence_percentage: str
    top_3_predictions: List[TopPrediction]
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    device: str
    target_classes: int


@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Root endpoint welcoming users and providing navigation links."""
    return {
        "message": "Welcome to Bright Future School Image Classifier API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Service health monitoring endpoint."""
    return HealthResponse(
        status="healthy",
        service="bright-future-image-classifier",
        model_loaded=app_model is not None,
        device=str(DEVICE),
        target_classes=len(CLASS_NAMES),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accepts an uploaded image file (PNG, JPG, JPEG, WEBP), performs inference,
    and returns predicted class, school scene tag (classroom vs playground), and confidences.
    """
    # 1. Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}",
        )

    # 2. Read image bytes
    start_time = time.perf_counter()
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.error(f"Failed to decode image: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid image format. Could not decode image bytes.",
        )

    # 3. Preprocess image
    input_tensor = PREPROCESS_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    # 4. Perform Model Inference
    with torch.no_grad():
        logits = app_model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    inference_time = (time.perf_counter() - start_time) * 1000.0

    # 5. Extract top-3 predictions
    top_probs, top_indices = torch.topk(probabilities, k=3)
    top_3: List[TopPrediction] = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        top_3.append(
            TopPrediction(
                class_name=class_name,
                confidence_percentage=round(prob.item() * 100.0, 2),
            )
        )

    primary_idx = top_indices[0].item()
    primary_class = CLASS_NAMES[primary_idx]
    primary_confidence = float(top_probs[0].item())
    scene_tag = SCENE_MAPPING.get(primary_class, "playground")

    logger.info(
        f"Prediction: {primary_class} ({primary_confidence:.4f}) -> Tag: {scene_tag} "
        f"| Latency: {inference_time:.1f}ms"
    )

    return PredictionResponse(
        success=True,
        predicted_class=primary_class,
        scene_tag=scene_tag,
        confidence=round(primary_confidence, 4),
        confidence_percentage=f"{primary_confidence * 100.0:.2f}%",
        top_3_predictions=top_3,
        inference_time_ms=round(inference_time, 2),
        model_version="ResNet-18-v2-Augmented",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
