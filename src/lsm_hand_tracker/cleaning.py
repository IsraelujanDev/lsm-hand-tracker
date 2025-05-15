import pandas as pd
import numpy as np
from pathlib import Path

from .path_config import METADATA_DIR

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with no predictive value based on EDA.
    """
    return df.drop(columns=["file_name", "width", "height"])

def mark_preferred_hand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing confidence values and add preferred_hand column.
    """
    df["confidence_left"] = df["confidence_left"].fillna(0)
    df["confidence_right"] = df["confidence_right"].fillna(0)
    df["preferred_hand"] = np.where(
        df["confidence_left"] >= df["confidence_right"],
        "left",
        "right"
    )
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame selecting the more confident hand's landmarks, distances, and angles.
    """
    mask_left = df["preferred_hand"] == "left"

    clean_df = pd.DataFrame({
        "label": df["label"],
        "hand_count": df["hand_count"],
        "handedness": df["preferred_hand"],
        "confidence": df["confidence_left"].where(mask_left, df["confidence_right"])
    })

    # landmarks    
    for axis in ("x", "y", "z"):
        for i in range(21):
            left_col  = f"left_{axis}{i}"
            right_col = f"right_{axis}{i}"
            output_col   = f"{axis}{i}"
            clean_df[output_col] = np.where(
                df["preferred_hand"] == "left",
                df[left_col],
                df[right_col]
            )

    # engineered distances and angles
    fingers = ["thumb","index","middle","ring","pinky"]
    for finger in fingers:
        clean_df[f"{finger}_dist"]  = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_dist"],
            df[f"right_{finger}_dist"]
        )
        clean_df[f"{finger}_angle"] = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_angle"],
            df[f"right_{finger}_angle"]
        )

    return clean_df

def finalize_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove samples with two hands and drop temporary columns.
    """
    df = df[df["hand_count"] != 2].reset_index(drop=True)
    
    # Droping confidence because our dataset doesn't contain so much variance in this column but in the future is possible to recover it
    # 'hand_count' has a similar case, where our model is just trained with one hand, if we solve this issue in the future we can include it again
    return df.drop(columns=["confidence", "hand_count"])

def clean_dataset(
    input_csv_path: Path = METADATA_DIR / "gestures_flat.csv",
    output_csv_path: Path = METADATA_DIR / "gestures_clean.csv",
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline: drop unused columns, choose preferred hand,
    extract features, finalize, then save the cleaned CSV.
    """
    df = pd.read_csv(input_csv_path)
    df = drop_unused_columns(df)
    df = mark_preferred_hand(df)
    df_feat = extract_features(df)
    clean_df = finalize_clean_df(df_feat)

    clean_df.to_csv(output_csv_path, index=False)
    print(f"Wrote cleaned data with {len(clean_df)} rows to {output_csv_path}")
    return clean_df

def main():
    clean_dataset()

if __name__ == "__main__":
    main()
