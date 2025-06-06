"""Build final training dataset by combining and flattening merged data files."""

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
from utils.data_transformers import flatten_extractor_outputs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Combine merged data files into a single flattened training dataset.
    
    Reads all merged CSV files, combines them, and flattens the data structure
    for training. Features are selected based on config settings.
    
    Note:
        Outputs extractor win rates if 'best_extractor' column is present.
        Directory is cleared before new files are written.
    """
    config = load_config()

    merged_dir = Path(config["data_paths"]["DB_merged_data"])
    output_dir = Path(config["data_paths"]["DB_final_training_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    merged_files = list(merged_dir.glob("*_merged.csv"))
    logger.info(f"Found {len(merged_files)} merged files to stack.")

    all_rows = []
    for file in merged_files:
        try:
            df = pd.read_csv(file)
            all_rows.append(df)
        except Exception as e:
            logger.error(f"Failed to read {file.name}: {e}")

    if not all_rows:
        logger.error("No data to combine.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # Extract config values
    feature_cols = config["features_to_use"]
    extractor_name_map = config["extractor_name_map"]

    # Flatten
    final_df = flatten_extractor_outputs(
        full_df,
        feature_cols,
        extractor_name_map,
    )

    # Save to disk
    final_file = output_dir / "training_data.csv"
    final_df.to_csv(final_file, index=False)
    logger.info(f"✓ Saved flattened training dataset to {final_file}")

    # Log extractor distribution
    if "best_extractor" in final_df.columns:
        extractor_counts = final_df["best_extractor"].value_counts(normalize=True).round(3)
        for extractor, proportion in extractor_counts.items():
            logger.info(f"Extractor win rate: {extractor} — {proportion:.1%}")
    else:
        logger.warning("No 'best_extractor' column found — skipping distribution log.")


if __name__ == "__main__":
    main()