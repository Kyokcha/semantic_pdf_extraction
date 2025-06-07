"""Merge matched sentences with their extracted features for model training."""

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
import logging
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_matched_with_features(doc_base: str, matched_dir: Path, 
                              feature_dir: Path, output_dir: Path) -> None:
    """Merge matched sentences with their features for a single document.
    
    Args:
        doc_base (str): Base document identifier (e.g., 'doc_001')
        matched_dir (Path): Directory containing matched sentence CSVs
        feature_dir (Path): Directory containing feature CSVs
        output_dir (Path): Directory for saving merged outputs
    
    Note:
        Handles multiple extractors per document.
        Missing similarity scores are filled with 0.
        Maintains clean versions of extracted_sentence and extractor columns.
    """
    try:
        # Gather all extractor-specific matches for this doc
        matched_paths = list(matched_dir.glob(f"{doc_base}-*_matched.csv"))
        if not matched_paths:
            logger.warning(f"{doc_base}: No matched files found.")
            return

        dfs = []
        for path in matched_paths:
            try:
                df = pd.read_csv(path)
                df['extractor'] = path.stem.split("-")[1].replace("_matched", "")
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Couldn't read {path.name}: {e}")

        if not dfs:
            logger.warning(f"{doc_base}: All matched files failed to load.")
            return

        matched_df = pd.concat(dfs, ignore_index=True)
        if "extracted_sentence_id" not in matched_df.columns:
            logger.warning(f"{doc_base}: No 'extracted_sentence_id' column.")
            return

        # Load feature file (per document)
        feature_path = feature_dir / f"{doc_base}.csv"
        if not feature_path.exists():
            logger.warning(f"{doc_base}: No features found at {feature_path}")
            return

        feature_df = pd.read_csv(feature_path)
        feature_df["extracted_sentence_id"] = feature_df["extracted_sentence_id"].astype(str)
        matched_df["extracted_sentence_id"] = matched_df["extracted_sentence_id"].astype(str)

        # Merge on sentence ID
        merged = pd.merge(matched_df, feature_df, on="extracted_sentence_id", how="left")

        # Keep clean version of extracted sentence
        if "extracted_sentence_y" in merged.columns:
            merged.drop(columns=["extracted_sentence_x"], inplace=True, errors="ignore")
            merged.rename(columns={"extracted_sentence_y": "extracted_sentence"}, inplace=True)
        elif "extracted_sentence" not in merged.columns:
            logger.warning(f"{doc_base}: No extracted_sentence column after merge.")
            return
        
        # Keep clean version of extractor column
        if "extractor_x" in merged.columns:
            merged.drop(columns=["extractor_y"], inplace=True, errors="ignore")
            merged.rename(columns={"extractor_x": "extractor"}, inplace=True)
        elif "extractor" not in merged.columns:
            logger.warning(f"{doc_base}: No extractor column after merge.")
            return

        # Fill similarity comparison columns if missing
        sim_cols = [col for col in merged.columns if "jaccard" in col or "cosine_sim" in col]
        merged[sim_cols] = merged[sim_cols].fillna(0)

        logger.info(f"{doc_base}: merged {len(merged)} rows across all extractors.")

        # Save merged file
        output_path = output_dir / f"{doc_base}_merged.csv"
        merged.to_csv(output_path, index=False)
        logger.info(f"✓ Merged: {output_path.name}")

    except Exception as e:
        logger.error(f"✗ Failed on {doc_base}: {e}")


def main() -> None:
    """Merge matched sentences with features for all documents.
    
    Combines matched sentence data with extracted features, processing
    each document independently in parallel.
    
    Note:
        Uses (CPU core count - 1) up to max 8 cores for parallel processing.
        Output directory is cleared before processing starts.
    """
    config = load_config()
    matched_dir = Path(config["data_paths"]["DB_matched_sentences"])
    feature_dir = Path(config["data_paths"]["DB_features"])
    output_dir = Path(config["data_paths"]["DB_merged_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    matched_files = list(matched_dir.glob("*_matched.csv"))
    doc_bases = sorted(set(f.stem.split("-")[0] for f in matched_files))
    logger.info(f"Found {len(doc_bases)} documents with matched data.")

    args_list = [(doc_base, matched_dir, feature_dir, output_dir) for doc_base in doc_bases]

    cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {cores} CPU cores.")

    with Pool(processes=cores) as pool:
        pool.starmap(merge_matched_with_features, args_list)


if __name__ == "__main__":
    main()