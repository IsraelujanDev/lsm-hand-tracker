import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Optional

from .path_config import PROCESSED_DIR, MODELS_DIR

def load_balanced_data(path: Path = PROCESSED_DIR / 'gestures_balanced.csv') -> pd.DataFrame:
    """
    Load the balanced dataset CSV.
    """
    return pd.read_csv(path)

def train_model(
    data: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Path = MODELS_DIR / 'gesture_classifier.joblib'
) -> None:
    """
    Train a RandomForest classifier on the gesture dataset, evaluate, and save the model.

    Steps:
      1) Split features and labels, then train/test split.
      2) Initialize and fit RandomForestClassifier.
      3) Evaluate with classification report and confusion matrix.
      4) Serialize the trained model to disk.
    """
    # Load data if not provided
    df = data if data is not None else load_balanced_data()

    # Separate features and label
    X = df.drop(columns=['label'])
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Model initialization
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(clf, model_path)
    print(f"Saved trained model to {model_path}")


def main():
    train_model()


if __name__ == '__main__':
    main()
