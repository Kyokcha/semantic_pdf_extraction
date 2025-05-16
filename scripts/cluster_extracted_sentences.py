import logging
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
from utils.clustering import cluster_sentences

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    input_csv_dir = Path(config["data_paths"]["DB_extracted_sentences"])
    input_emb_dir = Path(config["data_paths"]["DB_embeddings_PDF"])
    output_csv_dir = Path(config["data_paths"]["DB_clustered_sentences"])
    output_pt_dir = Path(config["data_paths"]["DB_clustered_embeddings"])

    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_pt_dir.mkdir(parents=True, exist_ok=True)

    clear_directory(output_csv_dir)
    clear_directory(output_pt_dir)

    # Group by doc_id (e.g., doc_001)
    grouped = {}
    for emb_file in input_emb_dir.glob("*.pt"):
        doc_id = emb_file.stem.split("-")[0]
        grouped.setdefault(doc_id, {"embeddings": [], "csvs": []})
        grouped[doc_id]["embeddings"].append(emb_file)
        grouped[doc_id]["csvs"].append(input_csv_dir / f"{emb_file.stem}.csv")

    logger.info(f"Found {len(grouped)} documents to cluster.")

    for doc_id, files in grouped.items():
        logger.info(f"Clustering for {doc_id}...")
        try:
            cluster_sentences(
                csv_files=files["csvs"],
                embedding_files=files["embeddings"],
                output_csv_dir=output_csv_dir,
                output_pt_dir=output_pt_dir,
                threshold=0.8  # adjustable threshold
            )
        except Exception as e:
            logger.error(f"✗ Failed to cluster {doc_id}: {e}")


if __name__ == "__main__":
    main()
