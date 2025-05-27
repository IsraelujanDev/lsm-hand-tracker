import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from .path_config import INTERIM_DIR, PROCESSED_DIR

def encode_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'left'/'right' in handedness to 0/1.
    """
    df['handedness'] = df['handedness'].map({'left': 0, 'right': 1})
    return df

def power_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Yeo-Johnson PowerTransformer to skewed features.
    """
    z_cols = [f'z{i}' for i in range(21)]
    dist_cols = [c for c in df.columns if c.endswith('_dist')]
    angle_cols = [c for c in df.columns if c.endswith('_angle')]
    cols = z_cols + dist_cols + angle_cols

    pt = PowerTransformer(method='yeo-johnson')
    df[cols] = pt.fit_transform(df[cols])
    return df

def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-Max scale all numeric columns.
    """
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def apply_pca(df: pd.DataFrame, variance: float) -> pd.DataFrame:
    """
    Fit PCA retaining given variance and return DataFrame of PC components plus label.
    """
    X = df.select_dtypes(include=['int64','float64'])
    pca = PCA(n_components=variance, random_state=42)
    X_pca = pca.fit_transform(X)
    cols = [f'PC{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=cols, index=df.index)
    df_pca['label'] = df['label']
    print(f'PCA: {len(cols)} components retain {variance:.0%} variance')
    return df_pca

def balance_dataset(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Oversample minority classes using SMOTE.
    """
    smote = SMOTE(random_state=42, k_neighbors=k)
    features = df.columns.drop('label')
    X_res, y_res, *_ = smote.fit_resample(df[features], df['label'])
    df_bal = pd.DataFrame(X_res, columns=features)
    df_bal['label'] = y_res
    print(f'SMOTE: balanced to {len(df_bal)} samples')
    return df_bal

def transform_and_balance_dataset(
    input_csv: Path = INTERIM_DIR / 'gestures_clean.csv',
    output_balanced: Path = PROCESSED_DIR / 'gestures_balanced.csv',
    pca_variance: float = 0.95,
    smote_k: int = 2,
) -> pd.DataFrame:
    """
    Orchestrates loading, encoding, transforming, scaling, PCA, and SMOTE,
    then saves the balanced dataset.
    """

    df = pd.read_csv(input_csv)
    df = encode_handedness(df)
    df = power_transform_features(df)
    df = scale_numeric(df)
    df_pca = apply_pca(df, pca_variance)
    df_balanced = balance_dataset(df_pca, smote_k)

    df_balanced.to_csv(output_balanced, index=False)
    print(f'Saved balanced data to {output_balanced}')
    return df_balanced

def main():
    transform_and_balance_dataset()

if __name__ == '__main__':
    main()
