
from datasets import load_dataset
from pathlib import Path
from utils.config import load_config


def main():

    # Configuration
    config = load_config()
    dump_dir = Path(config["data_paths"]["dump"])
    raw_dir = Path(config["data_paths"]["raw"])
    num_articles = config["wikipedia"]["num_articles"]
    snapshot = config["wikipedia"]["snapshot"]

    # Make output directory for .txt files
    dump_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load first N articles (sorted order, no randomness)
    dataset = load_dataset("wikipedia", snapshot, split=f"train[:{num_articles}]", trust_remote_code=True)

    # Save the full dataset to disk for reuse
    dataset.save_to_disk(str(dump_dir))
    print(f" Dataset saved to disk at {dump_dir}")
    
    # Write each article to a separate .txt file
    for i, article in enumerate(dataset):
        title = article['title'].replace("/", "-").replace("\\", "-")
        text = article['text'].strip()
        filename = raw_dir / f"{i:04d}_{title[:50]}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{text}")

    print(f" Saved {num_articles} articles to {raw_dir}")


if __name__ == "__main__":
    main()