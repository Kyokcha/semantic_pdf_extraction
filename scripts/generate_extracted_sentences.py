# scripts/generate_extracted_sentences

import logging
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count
from utils.config import load_config
from utils.file_operations import clear_directory
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_probable_table_line(text):
    tokens = text.split()

    if len(tokens) < 3:
        return False  # too short to be a table row

    # Count PascalCase or all-caps terms
    caps_or_mixed_case = sum(
        1 for w in tokens if re.match(r'^[A-Z][a-zA-Z0-9]*$', w) or w.isupper()
    )

    # Count terms with math-like symbols or digits
    mathy_words = sum(
        1 for w in tokens if any(c in w for c in "*+/=∗−()N0123456789")
    )

    has_no_punctuation = not any(p in text for p in ".!?")

    return (caps_or_mixed_case + mathy_words) / len(tokens) > 0.4 and has_no_punctuation


def process_text_file(args):
    txt_path, output_dir = args

    base_name = txt_path.stem  # e.g., doc_001-ocr
    if "-" not in base_name:
        logger.warning(f"Skipping file with unexpected name format: {txt_path.name}")
        return

    article_id, extractor = base_name.split("-", 1)

    try:
        raw_text = txt_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {txt_path.name}: {e}")
        return

    lines = raw_text.splitlines()
    text_lines = []
    table_lines = []
    sentence_blocks = []
    data = []
    global_id = 0

    for line in lines:
        line = line.replace("-\n", "").strip()
        line = " ".join(line.split())  # Normalize whitespace

        if not line:
            continue

        if is_probable_table_line(line):
            # Flush any collected text block
            if text_lines:
                sentence_blocks.append(" ".join(text_lines))
                text_lines = []

            table_lines.append({
                "type": "table_candidate",
                "text": line
            })
        else:
            text_lines.append(line)

    # Flush remaining text
    if text_lines:
        sentence_blocks.append(" ".join(text_lines))

    # Process full text blocks into sentences
    for block in sentence_blocks:
        for sentence in sent_tokenize(block):
            if sentence.strip():
                data.append({
                    "extracted_sentence_id": f"{article_id}_{extractor}_{global_id}",
                    "article_id": article_id,
                    "extractor": extractor,
                    "sentence_id": global_id,
                    "extracted_sentence": sentence.strip(),
                    "content_type": "text"
                })
                global_id += 1

    # Process table lines
    for table_entry in table_lines:
        data.append({
            "extracted_sentence_id": f"{article_id}_{extractor}_{global_id}",
            "article_id": article_id,
            "extractor": extractor,
            "sentence_id": global_id,
            "extracted_sentence": table_entry["text"],
            "content_type": "table_candidate"
        })
        global_id += 1

    if not data:
        logger.warning(f"No usable content in {txt_path.name}")
        return

    df = pd.DataFrame(data)
    output_file = output_dir / f"{base_name}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Processed {txt_path.name} -> {output_file.name}")


def main():
    config = load_config()
    input_dir = Path(config["data_paths"]["DB_extracted"])
    output_dir = Path(config["data_paths"]["DB_extracted_sentences"])
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
