# scripts/extract_features.py

import pandas as pd
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from utils import sentence_features
from utils.file_operations import clear_directory
from utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_file(args):
    """
    Process a single CSV file to extract linguistic and structural features from each sentence.

    Args:
        args (tuple): (csv_path, feature_dir)
    """
    csv_path, feature_dir = args
    try:
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if "extracted_sentence" not in df.columns or "extractor" not in df.columns:
            logger.warning(f"Skipping {csv_path.name}: missing required columns")
            return

        # Extract features for each sentence using the extractor name
        features = [
            sentence_features.sentence_features(row["extracted_sentence"], row["extractor"])
            for _, row in df.iterrows()
        ]
        features_df = pd.DataFrame(features)

        # Merge original metadata with features
        enriched_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

        # Save to features directory
        output_path = feature_dir / csv_path.name
        enriched_df.to_csv(output_path, index=False)
        logger.info(f"✓ Processed: {csv_path.name}")
    except Exception as e:
        logger.error(f"✗ Failed on {csv_path.name}: {e}")


def main():
    """
    Main control flow for batch feature extraction from sentence CSVs.
    Uses multiprocessing for efficiency.
    """
    config = load_config()

    extracted_dir = Path(config["data_paths"]["extracted_sentence"])
    feature_dir = Path(config["data_paths"]["features"])
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Optional: clear any existing outputs
    clear_directory(feature_dir)

    csv_files = list(extracted_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} sentence files to process.")

    # Decide on number of cores to use
    usable_cores = min(8, cpu_count() - 1)
    logger.info(f"Using {usable_cores} CPU cores.")

    args_list = [(csv_path, feature_dir) for csv_path in csv_files]

    # Run in parallel
    with Pool(processes=usable_cores) as pool:
        pool.map(process_file, args_list)


if __name__ == "__main__":
    main()
