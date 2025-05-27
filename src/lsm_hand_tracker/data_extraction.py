import cv2
import mediapipe as mp
import math
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from lsm_hand_tracker.path_config import IMAGES_DIR, MEDIAPIPE_MODEL, METADATA_DIR

# ---------------------- Constants & MediaPipe Initialization ----------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the MediaPipe Hand Landmarker once
MP_OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.70,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.55
)
MP_LANDMARKER = HandLandmarker.create_from_options(MP_OPTIONS)

# Landmark indices for engineered features
FINGERTIP_IDXS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCP_IDXS       = {"thumb": 2, "index": 5, "middle": 9,  "ring": 13, "pinky": 17}
PIP_IDXS       = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load an image and convert from BGR to RGB."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def hand_detection(
    rgb_image: np.ndarray,
    landmarker = MP_LANDMARKER
) -> Optional[Any]:
    """Perform hand-landmark detection on an RGB image."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection = landmarker.detect(mp_image)
    if not detection.hand_landmarks:
        print("No hands detected")
        return None
    return detection


def landmarks_extraction(
    detection: Any,
    rgb_image: np.ndarray
) -> Tuple[Dict[str, List[Dict[str, float]]], int, List[str], Dict[str, Optional[float]], int, int]:
    """Extract raw landmarks and basic metadata from detection."""
    height, width = rgb_image.shape[:2]
    hand_count = len(detection.hand_landmarks)

    handedness_list = [h[0].category_name.lower() for h in detection.handedness]
    hand_confidence: Dict[str, Optional[float]] = {
        "left": next(
            (h[0].score for h in detection.handedness if h[0].category_name.lower() == "left"),
            None
        ),
        "right": next(
            (h[0].score for h in detection.handedness if h[0].category_name.lower() == "right"),
            None
        )
    }

    landmarks_dict: Dict[str, List[Dict[str, float]]] = {"left": [], "right": []}
    for h, lm_list in zip(detection.handedness, detection.hand_landmarks):
        side = h[0].category_name.lower()
        landmarks_dict[side] = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in lm_list
        ]

    return landmarks_dict, hand_count, handedness_list, hand_confidence, height, width


def engineered_features_calculation(
    landmarks_dict: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute distances and angles for each finger relative to the wrist."""
    engineered: Dict[str, Dict[str, Dict[str, float]]] = {"left": {}, "right": {}}

    for side in ("left", "right"):
        lm = landmarks_dict.get(side, [])
        if not lm:
            continue

        # wrist landmark
        wrist = lm[0]

        # Distances wrist → each fingertip
        dists: Dict[str, float] = {}
        for finger, idx in FINGERTIP_IDXS.items():
            tip = lm[idx]
            dists[f"{finger}_dist"] = math.sqrt(
                (tip["x"] - wrist["x"])**2 +
                (tip["y"] - wrist["y"])**2 +
                (tip["z"] - wrist["z"])**2
            )
        # Angles at each MCP joint
        angs: Dict[str, float] = {}
        for finger in MCP_IDXS:
            mcp = np.array([lm[MCP_IDXS[finger]]["x"],
                            lm[MCP_IDXS[finger]]["y"],
                            lm[MCP_IDXS[finger]]["z"]])
            pip = np.array([lm[PIP_IDXS[finger]]["x"],
                            lm[PIP_IDXS[finger]]["y"],
                            lm[PIP_IDXS[finger]]["z"]])
            # Vector from MCP → PIP
            v1 = pip - mcp
            # Vector from MCP → wrist
            v2 = np.array([wrist["x"], wrist["y"], wrist["z"]]) - mcp

            # Compute angle (in degrees) between v1 and v2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angs[f"{finger}_angle"] = float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))

        engineered[side] = {"distances": dists, "angles": angs}

    return engineered


def process_single_image(
    image_path: Path
) -> Optional[Dict[str, Any]]:
    """Process one image: detect hands, extract landmarks, and compute features."""
    rgb = load_image(image_path)
    if rgb is None:
        return None

    detection = hand_detection(rgb)
    if detection is None:
        return None

    landmarks_dict, hand_count, handedness_list, hand_confidence, height, width = landmarks_extraction(detection, rgb)
    engineered = engineered_features_calculation(landmarks_dict)

    return {
        "file_name":       image_path.name,
        "label":           image_path.parent.name,
        "image_size":      [width, height],
        "hand_count":      hand_count,
        "handedness":      handedness_list,
        "hand_confidence": hand_confidence,
        "landmarks":       landmarks_dict,
        "engineered":      engineered
    }


def save_results_to_json(
    results: List[Dict[str, Any]],
    output_path: Path = METADATA_DIR / "gestures.json"
) -> None:
    """Save the collected metadata to a JSON file."""
    # Load existing data if present
    existing: List[Dict[str, Any]] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    # Combine and write back
    combined = existing + results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4)
    print(f"✅ Metadata saved to {output_path} (appended {len(results)} records)")


def generate_metadata_from_images(incremental: bool = False) -> None:
    """
    Walk all images in IMAGES_DIR, extract metadata, and append to gestures.json.
    If incremental=True, skip files whose metadata already exists.
    Returns the list of newly-processed records.
    """
    all_results: List[Dict[str, Any]] = []
    images = sorted(IMAGES_DIR.glob("*.jpg"))

    if not images:
        print(f"⚠️ No images found in {IMAGES_DIR}")
        return

    for image_path in images:
        print(f"Processing {image_path.name}...")
        result = process_single_image(image_path)
        if result:
            all_results.append(result)

    if all_results:
        save_results_to_json(all_results)
    else:
        print("⚠️ No valid data processed.")


def main() -> None:
    """Entry point for script execution."""
    generate_metadata_from_images()


if __name__ == "__main__":
    main()