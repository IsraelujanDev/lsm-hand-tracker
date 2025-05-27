#!/usr/bin/env python3
import argparse
from typing import List, Optional, Set, Callable, Any

from lsm_hand_tracker.data_extraction import generate_metadata_from_images
from lsm_hand_tracker.json_to_csv       import flatten_metadata_to_csv
from lsm_hand_tracker.cleaning          import clean_dataset
from lsm_hand_tracker.features          import transform_and_balance_dataset
from lsm_hand_tracker.training          import train_model


def run_pipeline(
    incremental: bool = False,
    skip_steps: Optional[List[str]] = None
) -> None:
    """
    Execute the full processing pipeline with optional incremental updates
    and selective skipping:
      1) Generate metadata
      2) Flatten to CSV
      3) Clean
      4) Transform & balance
      5) Train
    """
    skip: Set[str] = set(skip_steps or [])

    steps: List[tuple[str, Callable[[], Any]]] = [
        ("metadata", lambda: generate_metadata_from_images(incremental=incremental)),
        ("csv",      flatten_metadata_to_csv),
        ("clean",    clean_dataset),
        ("features", transform_and_balance_dataset),
        ("train",    train_model),
    ]

    for idx, (name, func) in enumerate(steps, start=1):
        if name in skip:
            print(f"{idx}) Skipping {name} step.")
            continue

        print(f"{idx}) {name.capitalize()} step…")
        result = func()
        if name == "metadata":
            # we expect a List of new records
            print(f"   ↳ {len(result)} new records processed.")

    print("✅ Pipeline complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full LSM hand-tracker pipeline."
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new images not yet in JSON."
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["metadata", "csv", "clean", "features", "train"],
        help="Steps to skip."
    )
    args = parser.parse_args()

    run_pipeline(
        incremental=args.incremental,
        skip_steps=args.skip
    )


if __name__ == "__main__":
    main()
