from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import string
import cv2
import numpy as np

from lsm_hand_tracker import RAW_DIR, MODELS_DIR
from lsm_hand_tracker.processing.data_extraction import (
    create_landmarker,
    process_one_image
)
from lsm_hand_tracker.processing.flatten import flatten_metadata
from lsm_hand_tracker.processing.cleaning import clean_dataset
from lsm_hand_tracker.processing.transformations import transform_features
from lsm_hand_tracker.model import predict_label_proba

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_LETTERS = set(string.ascii_uppercase) | {"Ñ"}
LANDMARKER = create_landmarker(MODELS_DIR / "hand_landmarker.task")


@app.post("/process/")
async def process_image(
    label: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Process an image upload and extract hand landmarks.
    - `label`: The label for the image, must be a single uppercase letter or 'Ñ'.
    - `image`: The image file to process.
    Returns a JSON response with the detected label, predicted label, confidence, and metadata.
    """
    letter = label.strip().upper()
    if letter not in VALID_LETTERS:
        raise HTTPException(400, detail=f"Invalid label: '{label}'")
    content = await image.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        raise HTTPException(422, detail="Could not decode image.")
    try:
        metadata = process_one_image(letter, bgr, LANDMARKER)
    except Exception as e:
        raise HTTPException(500, detail=f"Error during detection: {e}")
    if metadata is None:
        raise HTTPException(422, detail="No hands detected in image.")

    df_flat = flatten_metadata([metadata])
    clean_df = clean_dataset(df_flat).drop(columns=["label"])
    X = transform_features(clean_df)

    pred_label, confidence = predict_label_proba(X)
    print(f"Predicted label: {pred_label}, Confidence: {confidence:.2f}")

    if confidence > 0.85 and pred_label == letter:
        dest_dir = RAW_DIR / letter
    else:
        dest_dir = RAW_DIR / "review" / letter

    dest_dir.mkdir(parents=True, exist_ok=True)
    save_path = dest_dir / image.filename
    save_path.write_bytes(content)

    return JSONResponse({
        "detected_as":  metadata["label"],
        "predicted_as": pred_label,
        "confidence":   confidence,
        "metadata":     metadata
    })
