# scripts/process_sections.py

import os
import logging
from glob import glob
from tqdm import tqdm
from utils.config import load_config
from utils.section_grouper import extract_sections
from utils.file_operations import save_json
from utils.file_operations import clear_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    config = load_config()

    input_dir = config["data_paths"]["DB_raw"]
    output_dir = config["data_paths"]["DB_jsons"]
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)

    y_tol = config["docbank_processing"].get("line_y_tolerance", 5)
    logging.info(f"Y-tolerance set to {y_tol}")

    txt_files = glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        logging.warning(f"No .txt files found in {input_dir}")
        return

    for txt_path in tqdm(txt_files, desc="Processing Documents", unit="file"):
        doc_id = os.path.splitext(os.path.basename(txt_path))[0]
        logging.info(f"Processing {doc_id}...")

        try:
            sections = extract_sections(txt_path)
            out_path = os.path.join(output_dir, f"{doc_id}.json")
            save_json(sections, out_path)
            logging.info(f"Saved → {out_path}")
        except Exception as e:
            logging.error(f"Failed to process {doc_id}: {e}")


if __name__ == "__main__":
    main()
