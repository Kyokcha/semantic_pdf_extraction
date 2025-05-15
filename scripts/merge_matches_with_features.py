# scripts/merge_macthes_with_features

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
import logging
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_matched_with_features(args):
    matched_file, feature_dir, output_dir = args
    base_name = matched_file.stem.replace("_matched", "")
    feature_file = feature_dir / f"{base_name}.csv"

    if not feature_file.exists():
        logger.warning(f"Missing features for: {base_name}")
        return

    try:
        matched_df = pd.read_csv(matched_file)
        feature_df = pd.read_csv(feature_file)

        # Ensure ID fields are strings for consistent merging
        matched_df["extracted_sentence_id"] = matched_df["extracted_sentence_id"].astype(str)
        feature_df["extracted_sentence_id"] = feature_df["extracted_sentence_id"].astype(str)

        # Merge on sentence ID
        merged = pd.merge(matched_df, feature_df, on="extracted_sentence_id", how="left")

        # Rename to avoid suffix confusion
        if "extracted_sentence_x" in merged.columns:
            merged.rename(columns={"extracted_sentence_x": "matched_extracted_sentence"}, inplace=True)
            merged.drop(columns=["extracted_sentence_y"], inplace=True, errors="ignore")
            filter_col = "matched_extracted_sentence"
        else:
            logger.warning(f"{base_name}: 'extracted_sentence_x' missing after merge — skipping filtering.")
            filter_col = None

        # Filter rows with no sentence
        initial_count = len(merged)
        if filter_col:
            merged = merged[merged[filter_col].notnull()]
        final_count = len(merged)

        logger.info(f"{base_name}: kept {final_count}/{initial_count} rows with valid matches.")

        # Save output
        output_path = output_dir / f"{base_name}_merged.csv"
        merged.to_csv(output_path, index=False)
        logger.info(f"✓ Merged: {output_path.name}")

    except Exception as e:
        logger.error(f"✗ Failed on {matched_file.name}: {e}")


def main():
    config = load_config()

    matched_dir = Path(config["data_paths"]["DB_matched_sentences"])
    feature_dir = Path(config["data_paths"]["DB_features"])
    output_dir = Path(config["data_paths"]["DB_merged_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    matched_files = list(matched_dir.glob("*_matched.csv"))
    logger.info(f"Found {len(matched_files)} matched files to merge.")

    args_list = [(f, feature_dir, output_dir) for f in matched_files]

    cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {cores} CPU cores.")

    with Pool(processes=cores) as pool:
        pool.map(merge_matched_with_features, args_list)


if __name__ == "__main__":
    main()
