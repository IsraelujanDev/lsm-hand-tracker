import json
import pandas as pd
from pathlib import Path

# custom scripts
from .path_config import METADATA_DIR

def flatten_metadata_to_csv(
        metadata_json_path: Path = METADATA_DIR / "gestures.json",
        flattened_csv_path: Path = METADATA_DIR / "gestures_flat.csv"
):
    """
    Load the JSON metadata file and flatten it into a CSV format.
    The output CSV will contain one row per image, with columns for
    each feature, including engineered features.
    """

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Build a flat list of dicts
    flat_rows = []
    for rec in records:
        # Base metadata
        row = {
            "file_name": rec["file_name"],
            "label":     rec["label"],
            "width":     rec["image_size"][0],
            "height":    rec["image_size"][1],
            "hand_count":rec["hand_count"],
            # Handedness as one-hot flags
            "hadedness_left": int("left" in rec["handedness"]),
            "hadedness_right": int("right" in rec["handedness"]),
            # Confidence per side
            "confidence_left":  rec["hand_confidence"]["left"],
            "confidence_right": rec["hand_confidence"]["right"],
        }

        # Flatten raw landmarks: left_x0,left_y0,left_z0,…, right_x20,right_y20,right_z20
        for side in ("left", "right"):
            lms = rec["landmarks"].get(side, [])
            for idx, lm in enumerate(lms):
                row[f"{side}_x{idx}"] = lm["x"]
                row[f"{side}_y{idx}"] = lm["y"]
                row[f"{side}_z{idx}"] = lm["z"]

        # Flatten engineered distances and angles
        for side, feats in rec["engineered"].items():
            if feats is None:
                # fill NaNs for missing hand
                for finger in ("thumb","index","middle","ring","pinky"):
                    row[f"{side}_{finger}_dist"]  = float("nan")
                    row[f"{side}_{finger}_angle"] = float("nan")
                continue

            # Distances Data
            for dist_name, dist_val in feats["distances"].items():
                row[f"{side}_{dist_name}"] = dist_val
            
            # Angles Data
            for ang_name, ang_val in feats["angles"].items():
                row[f"{side}_{ang_name}"] = ang_val

        flat_rows.append(row)

    # 4) Create DataFrame and write CSV
    df = pd.DataFrame(flat_rows)
    df.to_csv(flattened_csv_path, index=False)
    print(f"Wrote flat CSV with {len(df)} rows to {flattened_csv_path}")

def main():
    flatten_metadata_to_csv()

if __name__ == "__main__":
    main()