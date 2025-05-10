# scripts/extract_features

import pandas as pd
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from utils import sentence_features
from utils.file_operations import clear_directory

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_file(args):
    csv_path, feature_dir = args
    try:
        df = pd.read_csv(csv_path)
        extractor_name = csv_path.stem.split("-")[-1]

        features = [sentence_features(row["sentence"], extractor_name) for _, row in df.iterrows()]
        features_df = pd.DataFrame(features)

        output_path = feature_dir / csv_path.name
        features_df.to_csv(output_path, index=False)
        logger.info(f"Processed: {csv_path.name}")
    except Exception as e:
        logger.error(f"Failed on {csv_path.name}: {e}")


def main():
    extracted_dir = Path("data/extracted")
    feature_dir = Path("data/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure directories exist and are empty
    clear_directory(feature_dir)

    csv_files = list(extracted_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSVs to process.")

    usable_cores = min(8, cpu_count() - 1)
    logger.info(f"Using {usable_cores} CPU cores.")

    # Build arguments list with csv_path + feature_dir
    args_list = [(csv_path, feature_dir) for csv_path in csv_files]

    with Pool(processes=usable_cores) as pool:
        pool.map(process_file, args_list)


if __name__ == "__main__":
    main()
