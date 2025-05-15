# scripts/build_final_training_dataset

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
from utils.data_transformers import flatten_extractor_outputs  # <-- new import
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    merged_dir = Path(config["data_paths"]["DB_merged_data"])
    output_dir = Path(config["data_paths"]["DB_final_training_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)
    
    training_cfg = config["training_config"]
    resolve_ties_randomly = training_cfg.get("resolve_ties_randomly", True)
    random_state = training_cfg.get("seed", 42)

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
        logger.error("❌ No data to combine.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # Extract config values
    feature_cols = config["features_to_use"]
    extractor_name_map = config["extractor_name_map"]

    # Flatten the dataset
    final_df = flatten_extractor_outputs(
        full_df,
        feature_cols,
        extractor_name_map,
        resolve_ties_randomly=resolve_ties_randomly,
        random_state=random_state
    )

    # Save
    final_file = output_dir / "training_data.csv"
    final_df.to_csv(final_file, index=False)
    logger.info(f"Saved flattened training dataset to {final_file}")


if __name__ == "__main__":
    main()
