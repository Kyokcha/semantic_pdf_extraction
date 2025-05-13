# scripts/select_sample_txt.py

import random
import logging
from pathlib import Path
from datasets import load_from_disk
from utils.config import load_config
from utils.file_operations import clear_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def word_count(text):
    return len(text.split())


def main():
    # Load config
    config = load_config()
    dump_dir = Path(config["data_paths"]["dump"])
    raw_dir = Path(config["data_paths"]["raw"])
    num_articles = config["wikipedia"]["num_articles"]
    min_words = config["wikipedia"].get("min_words", 200)
    seed = config["wikipedia"].get("seed", 42)

    # Prepare output directory
    clear_directory(raw_dir)

    # Load dataset
    dataset = load_from_disk(str(dump_dir))
    total = len(dataset)

    # Shuffle with seed
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)

    # Select articles with sufficient word count
    selected = []
    rejected_count = 0

    for idx in indices:
        example = dataset[idx]
        text = example.get("text", "").strip()
        if word_count(text) >= min_words:
            selected.append(example)
        else:
            rejected_count += 1
        if len(selected) >= num_articles:
            break

    logger.info(f"Selected {len(selected)} articles with ≥ {min_words} words")
    logger.info(f"Skipped {rejected_count} articles due to insufficient length")

    # Save selected articles
    for i, example in enumerate(selected):
        title = example.get("title", "Untitled").strip()
        text = example.get("text", "").strip()
        full_text = f"{title}\n\n{text}"
        file_path = raw_dir / f"article_{i+1:03}.txt"
        file_path.write_text(full_text, encoding="utf-8")

    logger.info(f"Wrote {len(selected)} text files to {raw_dir}")

if __name__ == "__main__":
    main()
