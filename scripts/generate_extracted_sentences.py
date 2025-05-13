# scripts/generate_extracted_sentences.py

import logging
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count
from utils.config import load_config
from utils.file_operations import clear_directory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_text_file(args):
    txt_path, output_dir = args

    # Parse article ID and extractor from filename
    base_name = txt_path.stem  # e.g., article_001-header_footer-pypdf2
    if base_name.count("-") < 2:
        logger.warning(f"Skipping file with unexpected name format: {txt_path.name}")
        return

    article_id, extractor = base_name.rsplit("-", 1)

    try:
        text = txt_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {txt_path.name}: {e}")
        return

    sentences = sent_tokenize(text)
    if not sentences:
        logger.warning(f"No sentences found in {txt_path.name}")
        return

    data = []
    for i, sentence in enumerate(sentences):
        extracted_sentence_id = f"{article_id}_{extractor}_{i}"
        data.append({
            "extracted_sentence_id": extracted_sentence_id,
            "article_id": article_id,
            "extractor": extractor,
            "sentence_id": i,
            "extracted_sentence": sentence.strip()
        })

    df = pd.DataFrame(data)
    output_file = output_dir / f"{base_name}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Processed {txt_path.name} -> {output_file.name}")


def main():
    config = load_config()
    input_dir = Path(config["data_paths"]["extracted"])
    output_dir = Path(config["data_paths"]["extracted_sentences"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    txt_files = list(input_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} extracted text files to process.")

    args = [(path, output_dir) for path in txt_files]
    usable_cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {usable_cores} CPU cores.")

    with Pool(usable_cores) as pool:
        pool.map(process_text_file, args)


if __name__ == "__main__":
    main()
