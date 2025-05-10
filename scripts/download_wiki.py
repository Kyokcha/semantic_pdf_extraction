
# scripts/download_wiki.py

from pathlib import Path
from datasets import load_dataset
from utils.config import load_config
from utils.file_operations import clear_directory


def main():

    # Configuration
    config = load_config()
    dump_dir = Path(config["data_paths"]["dump"])
    snapshot = config["wikipedia"]["snapshot"]

    # Make dump directory for .txt files
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Ensure directories exist and are empty
    clear_directory(dump_dir)

    # Load first N articles (sorted order, no randomness)
    dataset = load_dataset("wikipedia", snapshot, split="train", trust_remote_code=True)

    # Save the full dataset to disk for reuse
    dataset.save_to_disk(str(dump_dir))
    print(f" Dataset saved to disk at {dump_dir}")


if __name__ == "__main__":
    main()