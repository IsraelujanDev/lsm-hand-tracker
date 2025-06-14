import cv2
import json
import uuid
import numpy as np
import math
import mediapipe as mp

# Custom scripts
from lsm_hand_tracker.utils.path_config  import RAW_DIR, INTERIM_DIR, MODELS_DIR, REPORTS_DIR
from lsm_hand_tracker.processing.image_loader import gather_image_records

def generate_metadata(
    image_records,
    model_path    = MODELS_DIR / "hand_landmarker.task"):
    """
    Generate metadata for hand gesture recognition using MediaPipe.
    This script extracts hand landmarks, generates engineered features,
    and saves the results in JSON format.
    """

    # ---------------------- MediaPipe Initialization ----------------------

    # Validate that the model exists
    if not model_path.is_file():
        raise FileNotFoundError(f"HandLandmarker model not found at {model_path}")

    # Aliases from the Tasks API
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    # Build the options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.55
    )

    # Create the landmarker using a context manager for proper cleanup
    landmarker = HandLandmarker.create_from_options(options)

    # ---------------------- Landmark Extraction ----------------------

    print(f"Total images to process: {len(image_records)}")

    # DYNAMIC_LETTERS = {"J", "K", "Ñ", "Q", "X", "Z"}
    failed_count = 0
    failed_log: list = []  # collect failed or undetected filenames
    results = []

    print("\nProcessing Images...")

    with landmarker:
        for i, (letter, img_path) in enumerate(image_records, start=1):
            print(f"Processing {i}/{len(image_records)}: {img_path.name} ({letter})", end="\r")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                failed_count += 1
                failed_log.append(img_path.name)
                continue

            # Prepare image for MediaPipe
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect(mp_image)

            # Skip if no hands detected
            if not detection.hand_landmarks:
                failed_count += 1
                failed_log.append(img_path.name)
                continue

            # Otherwise, handle successful detection
            out_name = f"{letter}_{uuid.uuid4().hex[:8]}{img_path.suffix}"

            # --- Build metadata & landmark dicts ---
            height, width = bgr.shape[:2]
            hand_count = len(detection.hand_landmarks)

            # List of handedness strings
            handedness_list = [
                h[0].category_name.lower()
                for h in detection.handedness
            ]

            # Map each side to its confidence (or None)
            hand_confidence = {
                "left": next(
                    (h[0].score for h in detection.handedness
                    if h[0].category_name.lower() == "left"),
                    None
                ),
                "right": next(
                    (h[0].score for h in detection.handedness
                    if h[0].category_name.lower() == "right"),
                    None
                ),
            }

            # Gather raw landmarks per hand
            landmarks_dict = {"left": [], "right": []}
            for h, lm_list in zip(detection.handedness, detection.hand_landmarks):
                side = h[0].category_name.lower()
                landmarks_dict[side] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list
                ]

            # --- Compute engineered features per hand ---
            engineered = {"left": {}, "right": {}}

            # fingertip and joint indices as per MediaPipe
            fingertip_idxs = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
            mcp_idxs       = {"thumb": 2, "index": 5, "middle": 9,  "ring": 13, "pinky": 17}
            pip_idxs       = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

            for side in ("left", "right"):
                lm = landmarks_dict.get(side, [])
                if not lm:
                    continue

                # wrist landmark
                w = lm[0]

                # 1) Distances wrist → each fingertip
                dists = {}
                for finger, idx in fingertip_idxs.items():
                    tip = lm[idx]
                    d = math.sqrt(
                        (tip["x"] - w["x"])**2 +
                        (tip["y"] - w["y"])**2 +
                        (tip["z"] - w["z"])**2
                    )
                    dists[f"{finger}_dist"] = d

                # 2) Angles at each MCP joint
                angs = {}
                for finger in mcp_idxs:
                    mcp = np.array([lm[mcp_idxs[finger]]["x"],
                                    lm[mcp_idxs[finger]]["y"],
                                    lm[mcp_idxs[finger]]["z"]])
                    pip = np.array([lm[pip_idxs[finger]]["x"],
                                    lm[pip_idxs[finger]]["y"],
                                    lm[pip_idxs[finger]]["z"]])
                    # Vector from MCP → PIP
                    v1 = pip - mcp
                    # Vector from MCP → wrist
                    v2 = np.array([w["x"], w["y"], w["z"]]) - mcp

                    # Compute angle (in degrees) between v1 and v2
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    angle = float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))
                    angs[f"{finger}_angle"] = angle

                engineered[side] = {
                    "distances": dists,
                    "angles":    angs
                }


            # Append the full record
            results.append({
                "file_name":        out_name,
                "label":            letter,
                "image_size":       [width, height],
                "hand_count":       hand_count,
                "handedness":       handedness_list,
                "hand_confidence":  hand_confidence,
                "landmarks":        landmarks_dict,
                "engineered":       engineered
            })

    print(f"\nDone. Processed {len(image_records)} images: {len(results)} extracted, {failed_count} failures.")

    # Write results to JSON
    output_path = INTERIM_DIR / "gestures.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=4), encoding="utf-8")

    # Write failure log
    log_path = REPORTS_DIR / "failed_images_log.txt"
    log_path.write_text("\n".join(failed_log), encoding="utf-8")
    print(f"Saved failure log to {log_path}")



def main():
    image_records = gather_image_records(RAW_DIR)
    generate_metadata(image_records)

if __name__ == "__main__":
    main()
