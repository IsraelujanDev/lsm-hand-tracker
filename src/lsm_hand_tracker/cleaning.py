import pandas as pd
import numpy as np
from pathlib import Path

from .path_config import METADATA_DIR

def clean_dataset(
    input_csv_path:  Path = METADATA_DIR / "gestures_flat.csv",
    output_csv_path: Path = METADATA_DIR / "gestures_clean.csv",
) -> pd.DataFrame:
    """
    1) Read the flattened gestures CSV.
    2) Replace NaNs in confidence and choose the hand with higher confidence.
    3) Pull in only that hand’s landmarks, distances, and angles.
    4) Drop any rows where two hands were detected.
    5) Drop columns that are no longer needed.
    6) Write out the cleaned CSV.
    """
    # 1) Load
    df_raw = pd.read_csv(input_csv_path)

    # Work on a copy
    df = df_raw.copy()

    # 2) Fill missing confidences & choose preferred hand
    df["confidence_left_f"]  = df["confidence_left"].fillna(0)
    df["confidence_right_f"] = df["confidence_right"].fillna(0)

    df["preferred_hand"] = np.where(
        df["confidence_left_f"] >= df["confidence_right_f"],
        "left",
        "right"
    )

    # 3) Build new DataFrame with only the selected hand’s data
    cols_base = ["file_name", "label", "width", "height", "hand_count"]
    clean = df[cols_base].copy()

    # unified handedness & confidence
    clean["handedness"] = df["preferred_hand"]
    clean["confidence"]  = np.where(
        clean["handedness"] == "left",
        df["confidence_left"],
        df["confidence_right"]
    )

    # landmarks
    for axis in ("x", "y", "z"):
        for i in range(21):
            left_col  = f"left_{axis}{i}"
            right_col = f"right_{axis}{i}"
            out_col   = f"{axis}{i}"
            clean[out_col] = np.where(
                df["preferred_hand"] == "left",
                df[left_col],
                df[right_col]
            )

    # engineered features
    fingers = ["thumb","index","middle","ring","pinky"]
    for finger in fingers:
        clean[f"{finger}_dist"]  = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_dist"],
            df[f"right_{finger}_dist"]
        )
        clean[f"{finger}_angle"] = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_angle"],
            df[f"right_{finger}_angle"]
        )

    # 4) Filter out rows with two hands
    clean = clean[clean["hand_count"] != 2].reset_index(drop=True)

    # 5) Drop columns no longer needed
    clean.drop(columns=["file_name", "width", "height", "hand_count"], inplace=True)

    # 6) Save
    clean.to_csv(output_csv_path, index=False)
    print(f"Wrote cleaned data with {len(clean)} rows to {output_csv_path}")

    return clean


def main():
    clean_dataset()


if __name__ == "__main__":
    main()
