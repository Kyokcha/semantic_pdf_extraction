# scripts/build_final_training_dataset

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    merged_dir = Path(config["data_paths"]["merged_data"])
    output_dir = Path(config["data_paths"]["final_training_data"])
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
        logger.error("❌ No data to combine.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_file = output_dir / "training_data.csv"
    final_df.to_csv(final_file, index=False)
    logger.info(f"✅ Saved final training dataset to {final_file}")


if __name__ == "__main__":
    main()
