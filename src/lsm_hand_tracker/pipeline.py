from lsm_hand_tracker.processing.data_extraction import generate_metadata
from lsm_hand_tracker.processing.flatten import flatten_metadata_to_csv
from lsm_hand_tracker.processing.cleaning import clean_dataset
from lsm_hand_tracker.processing.features import transform_and_balance_dataset
from lsm_hand_tracker.training import train_model


def run_pipeline():
    """
    Execute the full processing pipeline:
      1) Generate metadata (JSON with landmarks and engineered features).
      2) Flatten metadata into a CSV file.
      3) Clean the dataset (drop unused columns, select preferred hand, remove NaNs).
      4) Transform features and balance classes (PowerTransformer, PCA, SMOTE).
      5) (Optional) Train and evaluate the predictive model.
    """
    print("1) Generating metadata…")
    generate_metadata()

    print("2) Flattening metadata to CSV…")
    
    flatten_metadata_to_csv()

    print("3) Cleaning the dataset…")
    clean_dataset()

    print("4) Transforming and balancing features…")
    transform_and_balance_dataset()

    print("5) Training the model…")
    train_model()

    print("✅ Pipeline complete!")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
