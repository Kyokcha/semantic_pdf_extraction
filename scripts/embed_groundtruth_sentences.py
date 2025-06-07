"""Generate sentence embeddings for ground truth text using parallel processing."""

import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from utils.config import load_config
from utils.embedding import embed_sentences_from_csv
from utils.file_operations import clear_directory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_csv_file(args: tuple) -> None:
    """Embed sentences from a single ground truth CSV file.
    
    Args:
        args (tuple): Contains (csv_path, output_dir) where:
            - csv_path (Path): Path to input CSV file
            - output_dir (Path): Directory for saving embeddings
    
    Note:
        Saves embeddings in PyTorch format (.pt files).
        Logs success or failure for each file processed.
    """
    csv_path, output_dir = args
    out_path = output_dir / f"{csv_path.stem}.pt"

    try:
        embed_sentences_from_csv(csv_path, out_path)
        logger.info(f"Embedded: {csv_path.name} → {out_path.name}")
    except Exception as e:
        logger.error(f"Failed to embed {csv_path.name}: {e}")


def main() -> None:
    """Process all ground truth files to generate embeddings.
    
    Reads CSV files containing ground truth sentences and generates embeddings
    using parallel processing for efficiency.
    
    Note:
        Uses (CPU core count - 1) up to max 8 cores for parallel processing.
        Output directory is cleared before processing starts.
    """
    config = load_config()

    input_dir = Path(config["data_paths"]["DB_ground_truth"])
    output_dir = Path(config["data_paths"]["DB_embeddings_GT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    csv_files = list(input_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} ground truth sentence files to embed.")

    args = [(csv_path, output_dir) for csv_path in csv_files]
    usable_cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {usable_cores} CPU cores.")

    with Pool(usable_cores) as pool:
        pool.map(embed_csv_file, args)


if __name__ == "__main__":
    main()