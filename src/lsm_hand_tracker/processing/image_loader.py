from pathlib import Path
from lsm_hand_tracker.utils.path_config import RAW_DIR

def gather_image_records(raw_dir: Path = RAW_DIR):
    exts_img = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    records = []
    for letter_dir in sorted(raw_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name
        for img_path in sorted(letter_dir.iterdir()):
            if img_path.suffix.lower() in exts_img:
                records.append((letter, img_path))
    return records

if __name__ == "__main__":
    records = gather_image_records()
    print(f"Found {len(records)} images.")
