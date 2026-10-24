
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

MODEL_REPO_ID = "ledaaaa/invoice-ai-models"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Invoice AI API",
    version="0.2.0",
    description="Invoice layout detection and OCR using YOLO and TrOCR.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

yolo_model = None
ocr_processor = None
ocr_model = None
model_load_errors: list[str] = []


@app.on_event("startup")
def load_models() -> None:
    global yolo_model, ocr_processor, ocr_model

    model_load_errors.clear()

    if not HF_TOKEN:
        model_load_errors.append(
            "HF_TOKEN is missing. Set it in the backend environment variables."
        )
        return

    try:
        # Download only the model files needed by this API.
        model_dir = Path(
            snapshot_download(
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
                allow_patterns=[
                    "yolo/best.pt",
                    "trocr/*",
                ],
            )
        )

        yolo_path = model_dir / "yolo" / "best.pt"
        trocr_path = model_dir / "trocr"

        # Load YOLO
        try:
            if not yolo_path.is_file():
                raise FileNotFoundError(f"YOLO file not found: {yolo_path}")

            yolo_model = YOLO(str(yolo_path))
        except Exception as exc:
            model_load_errors.append(f"YOLO load failed: {exc}")

        # Load TrOCR and its processor
        try:
            if not (trocr_path / "config.json").is_file():
                raise FileNotFoundError(
                    f"TrOCR config not found: {trocr_path}"
                )

            ocr_processor = TrOCRProcessor.from_pretrained(
                str(trocr_path),
                local_files_only=True,
            )

            ocr_model = VisionEncoderDecoderModel.from_pretrained(
                str(trocr_path),
                local_files_only=True,
            )

            ocr_model.to(DEVICE)
            ocr_model.eval()

        except Exception as exc:
            ocr_processor = None
            ocr_model = None
            model_load_errors.append(f"TrOCR load failed: {exc}")

    except Exception as exc:
        model_load_errors.append(
            f"Hugging Face download failed: {type(exc).__name__}: {exc}"
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Invoice AI API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    ready = (
        yolo_model is not None
        and ocr_model is not None
        and ocr_processor is not None
    )

    return {
        "status": "ok" if ready else "models_missing",
        "device": DEVICE,
        "model_repo": MODEL_REPO_ID,
        "yolo_loaded": yolo_model is not None,
        "trocr_loaded": ocr_model is not None,
        "processor_loaded": ocr_processor is not None,
        "details": model_load_errors,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if (
        yolo_model is None
        or ocr_model is None
        or ocr_processor is None
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Models are not loaded.",
                "errors": model_load_errors,
            },
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload an image file (JPG or PNG). Convert PDF pages to images first.",
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 15 MB.",
        )

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Try a JPG or PNG.",
        )

    with torch.inference_mode():
        prediction = yolo_model.predict(
            image,
            conf=0.25,
            verbose=False,
        )[0]

        items: list[dict[str, Any]] = []

        for box in prediction.boxes:
            x1, y1, x2, y2 = [
                int(v) for v in box.xyxy[0].tolist()
            ]

            x1 = max(0, min(x1, image.width - 1))
            x2 = max(x1 + 1, min(x2, image.width))
            y1 = max(0, min(y1, image.height - 1))
            y2 = max(y1 + 1, min(y2, image.height))

            crop = image.crop((x1, y1, x2, y2))

            pixel_values = ocr_processor(
                images=crop,
                return_tensors="pt",
            ).pixel_values.to(DEVICE)

            generated_ids = ocr_model.generate(
                pixel_values,
                max_new_tokens=128,
            )

            text = ocr_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            class_id = int(box.cls[0].item())
            class_name = prediction.names.get(
                class_id,
                str(class_id),
            )

            items.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    float(box.conf[0].item()), 4
                ),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "text": text,
            })

    return {
        "filename": file.filename,
        "image_size": {
            "width": image.width,
            "height": image.height,
        },
        "device": DEVICE,
        "detections": items,
        "note": (
            "Review OCR output before using it for "
            "accounting or business decisions."
        ),
    }
