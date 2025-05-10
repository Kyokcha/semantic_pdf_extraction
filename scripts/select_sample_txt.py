# scripts/select_sample_txt.py

import random
import logging
from pathlib import Path
from datasets import load_from_disk
from utils.config import load_config
from utils.file_operations import clear_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load config
    config = load_config()
    dump_dir = Path(config["data_paths"]["dump"])
    raw_dir = Path(config["data_paths"]["raw"])
    num_articles = config["wikipedia"]["num_articles"]
    seed = config["wikipedia"].get("seed", 42)

    # Prepare output directory
    clear_directory(raw_dir)

    # Load previously saved dataset
    dataset = load_from_disk(str(dump_dir))

    # Sample randomly with fixed seed
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_articles)
    sampled_dataset = dataset.select(indices)

    # Save each article to a text file
    for i, example in enumerate(sampled_dataset):
        # Retrieve article title and body
        title = example.get("title", "Untitled").strip()
        text = example.get("text", "").strip()

        # Prepend title to body for clarity
        full_text = f"{title}\n\n{text}"

        # Define a zero-padded filename (e.g., article_001.txt)
        file_path = raw_dir / f"article_{i+1:03}.txt"

        # Write the full text (title + body) to disk
        file_path.write_text(full_text, encoding="utf-8")
    
    logger.info(f"Extracted {num_articles} articles to {raw_dir}")


if __name__ == "__main__":
    main()
