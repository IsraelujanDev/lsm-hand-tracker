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

# Load .env (optional, keep if you use it)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass # dotenv not installed or not needed

base_path = os.getenv("LSM_BASE")
if not base_path:
    # Fallback if LSM_BASE is not set (adjust as needed)
    base_path = Path.cwd() # Use current working directory
    print("⚠️ Environment variable 'LSM_BASE' not set. Using current directory:", base_path)
    # raise ValueError("❌ Environment variable 'LSM_BASE' is not set!") # Or keep raising error

# Directories
raw_dir       = Path(base_path) / "data" / "raw"
images_dir    = Path(base_path) / "data" / "processed" / "images"
videos_dir    = Path(base_path) / "data" / "processed" / "videos"
not_det_dir   = Path(base_path) / "data" / "processed" / "not_detected"
metadata_dir  = Path(base_path) / "data" / "metadata"
metadata_file = metadata_dir / "gestures.json" # Changed to save all in one file

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
    r = (r * 0.3).astype(np.uint8) # Reduce red intensity
    return cv2.merge([b, g, r])

def resize_and_pad(image, size=(640, 480)):
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w*scale), int(h*scale)
    if nw == 0 or nh == 0: # Avoid zero dimensions
        return None # Cannot resize if image is empty
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking
    top = (size[1]-nh)//2
    bottom = size[1]-nh-top
    left = (size[0]-nw)//2
    right = size[0]-nw-left
    # Ensure padding values are non-negative
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        # This might happen if the scaled image is larger than the target size,
        # which shouldn't occur with min(scale) but good to handle.
        # Recalculate resize to fit exactly if necessary, or just return None/error.
        print(f"⚠️ Padding calculation error for image shape {image.shape[:2]} -> scaled {(nw, nh)}")
        # Option: Resize exactly to the target size if padding fails?
        # resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # return resized
        return None # Indicate failure
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0,0,0))

def full_preprocess(image):
    if image is None: return None
    img = adjust_lighting(image)
    img = gamma_correction(img)
    img = attenuate_red(img)
    return resize_and_pad(img)

# ---------------------- MediaPipe Initialization ----------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # Process images independently
    max_num_hands=2,              # Detect up to two hands
    min_detection_confidence=0.35 # Lower confidence threshold
)
# Separate instance for videos if needed, might tune differently
hands_video = mp.solutions.hands.Hands(
    static_image_mode=False,      # Process video stream
    max_num_hands=1,              # Often only one hand matters for dynamic gestures? Adjust if needed
    min_detection_confidence=0.5, # Potentially higher confidence for videos
    min_tracking_confidence=0.5
)

# ---------------------- Gather Records ----------------------

exts_img   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"} # Added more image types
exts_vid   = {".mp4", ".avi", ".mov", ".wmv", ".mkv"}  # Added more video types

image_records = []
video_records = []

print(f"Scanning for media in: {raw_dir}")
if not raw_dir.exists():
    print(f"❌ Raw data directory not found: {raw_dir}")
else:
    for letter_dir in sorted(raw_dir.iterdir()):
        if not letter_dir.is_dir(): continue
        letter = letter_dir.name
        print(f"Processing letter: {letter}")
        count_img = 0
        count_vid = 0
        for file in sorted(letter_dir.iterdir()):
            if file.suffix.lower() in exts_img:
                image_records.append((letter, file))
                count_img += 1
            elif file.suffix.lower() in exts_vid:
                video_records.append((letter, file))
                count_vid += 1
        print(f"  Found {count_img} images, {count_vid} videos.")

print(f"\nTotal images found: {len(image_records)}")
print(f"Total videos found: {len(video_records)}")

# ---------------------- Unified Processing ----------------------

DYNAMIC_LETTERS = {"J", "K", "M", "Q", "X", "Z"} # Assuming these letters require motion
all_entries = []
processed_count = 0
failed_count = 0
not_detected_count = 0

# --- Process Images ---
print("\nProcessing Images...")
for i, (letter, img_path) in enumerate(image_records):
    print(f"  Image {i+1}/{len(image_records)}: {img_path.name} ({letter})", end='\r')
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"\n⚠️ Failed to read image: {img_path.name}")
        failed_count += 1
        continue

    # Initial resize/pad (less aggressive preprocessing)
    proc = resize_and_pad(img)
    if proc is None:
         print(f"\n⚠️ Failed to resize/pad (initial): {img_path.name}")
         failed_count += 1
         continue

    # --- First detection attempt ---
    result = hands.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    final_img_to_save = proc
    detection_successful = bool(result.multi_hand_landmarks)

    # --- Fallback with full preprocessing if needed ---
    if not detection_successful:
        proc2 = full_preprocess(img) # Apply more aggressive preprocessing
        if proc2 is None:
             print(f"\n⚠️ Failed to resize/pad (fallback): {img_path.name}")
             failed_count += 1
             continue
        result = hands.process(cv2.cvtColor(proc2, cv2.COLOR_BGR2RGB))
        detection_successful = bool(result.multi_hand_landmarks)
        if detection_successful:
            final_img_to_save = proc2 # Use the preprocessed image if detection worked
        # else: keep final_img_to_save as the original 'proc'

    # --- Handle No Detection ---
    if not detection_successful:
        dest = not_det_dir / img_path.name
        try:
            shutil.copy(str(img_path), str(dest))
        except Exception as e:
             print(f"\n⚠️ Failed to copy to not_detected: {img_path.name} - {e}")
        not_detected_count += 1
        continue # Skip saving metadata if no hands detected

    # --- Save Processed Image ---
    out_name = f"{letter}_img_{uuid.uuid4().hex[:8]}.jpg" # Clearer naming
    out_path = images_dir / out_name
    try:
        cv2.imwrite(str(out_path), final_img_to_save)
    except Exception as e:
        print(f"\n⚠️ Failed to write image: {out_path} - {e}")
        failed_count += 1
        continue

    # --- Build Metadata (Corrected Structure) ---
    # Initialize with None for both hands
    landmarks_dict  = {"right_hand": None, "left_hand": None}
    confidence_dict = {"right_hand": None, "left_hand": None}
    handedness_list = []

    if result.multi_hand_landmarks and result.multi_handedness:
        # Iterate through detected hands and populate the dictionaries
        for lm_list, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Mediapipe gives handedness as a list, take the first classification
            classification = hand_info.classification[0]
            side = classification.label.lower() # Should be "left" or "right"
            handedness_list.append(side)

            # Store landmarks for the correct hand
            landmarks_dict[f"{side}_hand"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in lm_list.landmark
            ]
            # Store confidence for the correct hand
            confidence_dict[f"{side}_hand"] = classification.score

    # Create the final JSON entry for the image
    entry = {
        "id":             out_name.split('.')[0], # Use filename without ext as ID
        "image":          out_name,               # Filename in processed/images
        "video":          None,                   # Explicitly None for images
        "letter":         letter,
        "gesture_type":   "dynamic" if letter in DYNAMIC_LETTERS else "static",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_file":    str(img_path.relative_to(base_path)), # Original path relative to base
        "image_size":     [final_img_to_save.shape[1], final_img_to_save.shape[0]], # W, H
        "hand_count":     len(handedness_list),   # Number of detected hands
        "handedness_detected": handedness_list,   # List of detected hands ["left", "right"]
        "hand_confidence": confidence_dict,       # Dict with scores {"left_hand": score, "right_hand": score}
        "landmarks":      landmarks_dict          # <<< CORRECTED STRUCTURE for images
                                                  # e.g., {"left_hand": [...], "right_hand": None}
    }
    all_entries.append(entry)
    processed_count += 1

print(f"\nFinished processing {len(image_records)} images.")

# --- Process Videos ---
print("\nProcessing Videos...")
for i, (letter, vid_path) in enumerate(video_records):
    print(f"  Video {i+1}/{len(video_records)}: {vid_path.name} ({letter})", end='\r')
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"\n⚠️ Failed to open video: {vid_path.name}")
        failed_count += 1
        continue

    # --- Prepare Output ---
    out_name = f"{letter}_vid_{uuid.uuid4().hex[:8]}.mp4" # Clearer naming
    out_path = videos_dir / out_name

    # We just copy the video for now, landmark extraction happens next
    try:
        shutil.copy(str(vid_path), str(out_path))
    except Exception as e:
        print(f"\n⚠️ Failed to copy video: {vid_path.name} to {out_path} - {e}")
        failed_count += 1
        cap.release()
        continue

    # --- Extract Landmarks Frame by Frame ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_landmarks_sequence = [] # List to store landmarks per frame
    frame_idx = 0
    detected_frames_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Resize frame to consistent input size for detection (e.g., 640x480)
        # Important: Use the same size as used for padding images if consistency is needed
        # Or resize based on video aspect ratio. Here using fixed size.
        proc_frame = resize_and_pad(frame, size=(640, 480))
        if proc_frame is None:
            # Add empty entry if frame processing failed
            frame_landmarks_sequence.append({"frame": frame_idx, "landmarks": [None] * 21}) # Keep structure consistent
            frame_idx += 1
            continue

        # Process with video hand detector
        result_vid = hands_video.process(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB))

        # --- Get Landmarks for this frame ---
        # Note: This simplified version takes the *first detected hand*.
        # To store left/right, you'd need logic similar to image processing *per frame*.
        if result_vid.multi_hand_landmarks:
            detected_frames_count += 1
            # Take landmarks of the first detected hand
            hand_landmarks = result_vid.multi_hand_landmarks[0]
            frame_lms = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            # TODO: If needed, add handedness check here per frame
            # handedness = result_vid.multi_handedness[0].classification[0].label.lower()
            # frame_lms_dict = {f"{handedness}_hand": frame_lms, "other_hand": None} # Example
        else:
            # No hand detected in this frame, add placeholder
            frame_lms = [None] * 21 # Use list of Nones to indicate no detection

        frame_landmarks_sequence.append({"frame": frame_idx, "landmarks": frame_lms})
        frame_idx += 1

    cap.release()

    # --- Check if *any* hand was detected in the video ---
    if detected_frames_count == 0:
        # If no hands detected in any frame, move original video to not_detected
        dest = not_det_dir / vid_path.name
        try:
            # Move the *original* raw video if no hands ever detected
            if out_path.exists(): os.remove(out_path) # remove the copy we made
            shutil.copy(str(vid_path), str(dest)) # copy original
        except Exception as e:
             print(f"\n⚠️ Failed to move to not_detected: {vid_path.name} - {e}")
        not_detected_count += 1
        continue # Skip adding metadata for videos with no detections

    # --- Build Metadata for Video ---
    entry = {
        "id":             out_name.split('.')[0], # Use filename without ext as ID
        "image":          None,                   # Explicitly None for videos
        "video":          out_name,               # Filename in processed/videos
        "letter":         letter,
        "gesture_type":   "dynamic",              # Videos are assumed dynamic? Or check letter?
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_file":    str(vid_path.relative_to(base_path)), # Original path relative to base
        "fps":            fps,
        "total_frames":   frame_idx,             # Actual number of frames processed
        # Size refers to the *processed* size landmarks relate to
        "image_size":     [640, 480], # The size used for processing (resize_and_pad target)
        # Hand count for videos is tricky. This counts frames with *any* detection.
        "hand_count":     detected_frames_count,
        # Handedness/Confidence not stored per-frame in this simplified version
        # "handedness_detected": [], # Could add if tracked per frame
        # "hand_confidence": {},     # Could add if tracked per frame
        "landmarks":      frame_landmarks_sequence # List of {"frame": i, "landmarks": [...]}
                                                   # where [...] is list of 21 points or Nones
    }
    all_entries.append(entry)
    processed_count += 1

print(f"\nFinished processing {len(video_records)} videos.")

# ---------------------- Save Unified JSON ----------------------

print(f"\nAttempting to save metadata to: {metadata_file}")
try:
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Successfully saved metadata for {len(all_entries)} entries to {metadata_file}")
    print(f"  - Processed media: {processed_count}")
    print(f"  - Failed reads/writes: {failed_count}")
    print(f"  - No hands detected: {not_detected_count}")
except Exception as e:
    print(f"\n❌ Failed to save JSON metadata: {e}")

print("\nPreprocessing finished.")