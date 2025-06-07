"""Generate features from extracted sentences for model training."""

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


def merge_extractor_files_for_doc(doc_id: str, extractor_files: list[Path]) -> pd.DataFrame:
    """Combine outputs from different extractors for a single document.
    
    Args:
        doc_id (str): Document identifier.
        extractor_files (list[Path]): List of CSV file paths for each extractor.
    
    Returns:
        pd.DataFrame: Combined data from all extractors, or empty DataFrame if no valid data.
        
    Note:
        Requires 'sentence_id', 'extractor', and 'extracted_sentence' columns in CSVs.
    """
    dfs = []
    for path in extractor_files:
        df = pd.read_csv(path)
        if {"sentence_id", "extractor", "extracted_sentence"}.issubset(df.columns):
            dfs.append(df)
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged["article_id"] = doc_id
        return merged
    else:
        return pd.DataFrame()


def process_document(args: tuple) -> None:
    """Process a single document to generate features for each extracted sentence.
    
    Args:
        args (tuple): Contains (doc_id, extractor_paths, feature_dir) where:
            - doc_id (str): Document identifier
            - extractor_paths (list[Path]): Paths to extractor output files
            - feature_dir (Path): Output directory for feature files
            
    Note:
        Generates inter-extractor comparison features.
        Skips invalid or empty sentences.
    """
    doc_id, extractor_paths, feature_dir = args
    try:
        df = merge_extractor_files_for_doc(doc_id, extractor_paths)
        if df.empty:
            logger.warning(f"Skipping {doc_id}: no valid extractor outputs")
            return

        df["extracted_sentence"] = df["extracted_sentence"].astype(str)
        grouped = df.groupby("sentence_id")
        enriched_rows = []

        for sentence_id, group in grouped:
            # Map: extractor → sentence string
            sentence_versions = {
                row["extractor"]: row["extracted_sentence"]
                for _, row in group.iterrows()
                if isinstance(row["extracted_sentence"], str) and row["extracted_sentence"].strip()
            }

            for _, row in group.iterrows():
                text = row["extracted_sentence"]
                if not isinstance(text, str) or text.strip().lower() == "nan":
                    continue

                extractor = row["extractor"]

                # Remove self from comparison list for inter-extractor features
                other_versions = {
                    name: sent for name, sent in sentence_versions.items()
                    if name != extractor
                }

                feats = sentence_features.sentence_features(
                    row=row,
                    all_sentences=other_versions,
                )

                enriched_row = row.to_dict()
                enriched_row.update(feats)
                enriched_rows.append(enriched_row)

        if enriched_rows:
            enriched_df = pd.DataFrame(enriched_rows)
            output_path = feature_dir / f"{doc_id}.csv"
            enriched_df.to_csv(output_path, index=False)
            logger.info(f"✓ Processed: {doc_id}")
        else:
            logger.warning(f"No valid rows for {doc_id}")

    except Exception as e:
        logger.error(f"✗ Failed on {doc_id}: {e}")


def main() -> None:
    """Generate features for all extracted sentences using parallel processing.
    
    Processes each document's extracted sentences to generate features for
    model training, including inter-extractor comparisons.
    
    Note:
        Uses (CPU core count - 1) up to max 8 cores for parallel processing.
        Output directory is cleared before processing starts.
    """
    config = load_config()
    extracted_dir = Path(config["data_paths"]["DB_extracted_sentences"])
    feature_dir = Path(config["data_paths"]["DB_features"])
    feature_dir.mkdir(parents=True, exist_ok=True)

    clear_directory(feature_dir)

    # Collect all CSVs and group by doc_id prefix (before "-ocr.csv")
    extractor_files = list(extracted_dir.glob("*.csv"))
    doc_groups = {}
    for path in extractor_files:
        doc_id = path.name.split("-")[0]
        doc_groups.setdefault(doc_id, []).append(path)

    logger.info(f"Found {len(doc_groups)} documents with extractor outputs.")
    usable_cores = min(8, cpu_count() - 1)
    logger.info(f"Using {usable_cores} CPU cores.")

    args_list = [(doc_id, paths, feature_dir) for doc_id, paths in doc_groups.items()]

    with Pool(processes=usable_cores) as pool:
        pool.map(process_document, args_list)


if __name__ == "__main__":
    main()