import os
import cv2
import json
import shutil
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import mediapipe as mp

# ---------------------- Configuration & Paths ----------------------

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

base_path = os.getenv("LSM_BASE")
if not base_path:
    raise ValueError("❌ Environment variable 'LSM_BASE' is not set!")

# Directories
raw_dir       = Path(base_path) / "data" / "raw"
images_dir    = Path(base_path) / "data" / "processed" / "images"
videos_dir    = Path(base_path) / "data" / "processed" / "videos"
not_det_dir   = Path(base_path) / "data" / "processed" / "not_detected"
metadata_dir  = Path(base_path) / "data" / "metadata"
metadata_file = metadata_dir / "gestures.json"

for d in (images_dir, videos_dir, not_det_dir, metadata_dir):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------- Preprocessing Functions ----------------------

def adjust_lighting(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    table = table.astype("uint8")
    return cv2.LUT(image, table)

def attenuate_red(image):
    b, g, r = cv2.split(image)
    r = (r * 0.3).astype(np.uint8)
    return cv2.merge([b, g, r])

def resize_and_pad(image, size=(640, 480)):
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (nw, nh))
    top = (size[1]-nh)//2
    bottom = size[1]-nh-top
    left = (size[0]-nw)//2
    right = size[0]-nw-left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0,0,0))

def full_preprocess(image):
    img = adjust_lighting(image)
    img = gamma_correction(img)
    img = attenuate_red(img)
    return resize_and_pad(img)

# ---------------------- MediaPipe Initialization ----------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.35
)

# ---------------------- Gather Records ----------------------

exts_img   = {".jpg", ".jpeg", ".png"}
exts_vid   = {".mp4", ".avi", ".mov"}

image_records = []
video_records = []

for letter_dir in sorted(raw_dir.iterdir()):
    if not letter_dir.is_dir(): continue
    letter = letter_dir.name
    for file in sorted(letter_dir.iterdir()):
        if file.suffix.lower() in exts_img:
            image_records.append((letter, file))
        elif file.suffix.lower() in exts_vid:
            video_records.append((letter, file))

# ---------------------- Unified Processing ----------------------

DYNAMIC_LETTERS = {"J", "K", "M", "Q", "X", "Z"}
all_entries = []

# Process Images
for letter, img_path in image_records:
    img = cv2.imread(str(img_path))
    proc = resize_and_pad(img)
    # First pass
    result = hands.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    # Fallback
    if not result.multi_hand_landmarks:
        proc2 = full_preprocess(img)
        result = hands.process(cv2.cvtColor(proc2, cv2.COLOR_BGR2RGB))
        final_img = proc2 if result.multi_hand_landmarks else None
    else:
        final_img = proc

    if final_img is None:
        dest = not_det_dir / img_path.name
        shutil.copy(str(img_path), str(dest))
        continue

    out_name = f"{letter}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = images_dir / out_name
    cv2.imwrite(str(out_path), final_img)

    # Build metadata
    landmarks_dict  = {"right_hand": None, "left_hand": None}
    confidence_dict = {"right_hand": None, "left_hand": None}
    handedness_list = []

    if result.multi_hand_landmarks and result.multi_handedness:
        for lm_list, hand_h in zip(result.multi_hand_landmarks, result.multi_handedness):
            side = hand_h.classification[0].label.lower()
            handedness_list.append(side)
            landmarks_dict[f"{side}_hand"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in lm_list.landmark
            ]
            confidence_dict[f"{side}_hand"] = hand_h.classification[0].score

    # Determine the one hand’s landmarks (or empty list if none)
    if handedness_list:
        first_side = handedness_list[0]               # e.g. "right"
        pts = landmarks_dict[f"{first_side}_hand"]    # pull that list of 21 points
    else:
        pts = []

    entry = {
        "image":                out_name,
        "video":                None,
        "letter":               letter,
        "gesture_type":         "dynamic" if letter in DYNAMIC_LETTERS else "static",
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "image_size":           [proc.shape[1], proc.shape[0]],
        "hand_count":           len(handedness_list),
        "handedness_detected":  handedness_list,
        "hand_confidence":      confidence_dict,
        "landmarks":            [{"frame": 0, "landmarks": pts}]
    }
    all_entries.append(entry)

# Process Videos
for letter, vid_path in video_records:
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        continue

    out_name = f"{letter}_dyn_{uuid.uuid4().hex[:8]}.mp4"
    shutil.copy(str(vid_path), str(videos_dir / out_name))

    fps = cap.get(cv2.CAP_PROP_FPS)
    seq = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        proc = resize_and_pad(frame)
        res  = hands.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            lms = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in res.multi_hand_landmarks[0].landmark
            ]
        else:
            lms = [None] * 21

        seq.append({"frame": frame_idx, "landmarks": lms})
        frame_idx += 1

    cap.release()

    if all(all(pt is None for pt in f["landmarks"]) for f in seq):
        shutil.copy(str(vid_path), str(not_det_dir / vid_path.name))
        continue

    entry = {
        "image":        None,
        "video":        out_name,
        "letter":       letter,
        "gesture_type": "dynamic",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "fps":          fps,
        "total_frames": frame_idx,
        "image_size":   [640, 480],
        "hand_count":   sum(1 for f in seq if any(pt is not None for pt in f["landmarks"])),
        "landmarks":    seq
    }
    all_entries.append(entry)

# ---------------------- Save Unified JSON ----------------------

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(all_entries, f, indent=2)

print(f"✅ Saved {len(all_entries)} entries to {metadata_file}")

