import pandas as pd
from pathlib import Path

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from .path_config import METADATA_DIR

def engineer_features(
    input_csv:        Path = METADATA_DIR / "gestures_clean.csv",
    output_pca_csv:   Path = METADATA_DIR / "gestures_pca.csv",
    output_balanced:  Path = METADATA_DIR / "gestures_balanced.csv",
    pca_variance:     float = 0.95,
    smote_k:          int   = 2,
) -> pd.DataFrame:
    """
    1) Read cleaned CSV.
    2) Binary-encode handedness.
    3) Power-transform skewed features.
    4) MinMax-scale all numeric cols.
    5) Drop 'confidence'.
    6) PCA (retain pca_variance of total variance).
    7) SMOTE to balance labels.
    8) Save PCA-dataframe and balanced dataframe.
    Returns the final balanced DataFrame.
    """
    # 1) Load
    df = pd.read_csv(input_csv)

    # 2) Categorical → numeric
    df["handedness"] = df["handedness"].map({"left": 0, "right": 1})

    # 3) Identify columns to transform
    z_cols     = [f"z{i}" for i in range(21)]
    dist_cols  = [c for c in df.columns if c.endswith("_dist")]
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    to_transform = z_cols + dist_cols + angle_cols

    # 4) PowerTransformer
    pt = PowerTransformer(method="yeo-johnson")
    df[to_transform] = pt.fit_transform(df[to_transform])

    # 5) Scale all numeric columns
    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 6) Drop confidence
    if "confidence" in df.columns:
        df.drop(columns=["confidence"], inplace=True)

    # 7) PCA
    X = df[num_cols]
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    df_pca["label"] = df["label"]
    df_pca.to_csv(output_pca_csv, index=False)
    print(f"Saved PCA dataset ({len(pca_cols)} components) to {output_pca_csv}")

    # 8) SMOTE
    smote = SMOTE(random_state=42, k_neighbors=smote_k)
    X_res, y_res = smote.fit_resample(df_pca[pca_cols], df_pca["label"])
    df_balanced = pd.DataFrame(X_res, columns=pca_cols)
    df_balanced["label"] = y_res
    df_balanced.to_csv(output_balanced, index=False)
    print(f"Saved balanced dataset ({len(df_balanced)}) to {output_balanced}")

    return df_balanced

def main():
    engineer_features()

if __name__ == "__main__":
    main()
