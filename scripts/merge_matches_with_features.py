# scripts/merge_matches_with_features.py

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
    doc_id = matched_file.stem.replace("_matched", "")  # includes extractor, e.g., "doc_038-ocr"

    try:
        matched_df = pd.read_csv(matched_file)

        if matched_df.empty or "extracted_sentence_id" not in matched_df.columns:
            logger.warning(f"{doc_id}: No extracted_sentence_id column or empty data.")
            return

        # Load the corresponding feature file
        feature_files = list(feature_dir.glob(f"{doc_id}.csv"))
        if not feature_files:
            logger.warning(f"Missing features for: {doc_id}")
            return

        logger.info(f"{doc_id}: Found {len(feature_files)} feature file(s).")

        feature_dfs = []
        for f in feature_files:
            try:
                df = pd.read_csv(f)
                feature_dfs.append(df)
            except Exception as e:
                logger.warning(f"Couldn't read feature file {f.name}: {e}")

        if not feature_dfs:
            logger.warning(f"No readable features for: {doc_id}")
            return

        feature_df = pd.concat(feature_dfs, ignore_index=True)

        # Ensure ID fields are strings for consistent merging
        matched_df["extracted_sentence_id"] = matched_df["extracted_sentence_id"].astype(str)
        feature_df["extracted_sentence_id"] = feature_df["extracted_sentence_id"].astype(str)

        # Merge on sentence ID
        merged = pd.merge(
            matched_df,
            feature_df,
            on="extracted_sentence_id",
            how="left"
        )

        # Clean up columns
        if "extracted_sentence_x" in merged.columns:
            merged.rename(columns={"extracted_sentence_x": "matched_extracted_sentence"}, inplace=True)
            merged.drop(columns=["extracted_sentence_y"], inplace=True, errors="ignore")
        elif "extracted_sentence" in merged.columns:
            merged.rename(columns={"extracted_sentence": "matched_extracted_sentence"}, inplace=True)

        # Drop rows without matched sentences
        initial_count = len(merged)
        merged = merged[merged["matched_extracted_sentence"].notnull()]
        final_count = len(merged)
        logger.info(f"{doc_id}: kept {final_count}/{initial_count} rows with valid matches.")

        # Save output
        output_path = output_dir / f"{doc_id}_merged.csv"
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
