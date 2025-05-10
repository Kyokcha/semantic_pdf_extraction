# scripts/process_raw_to_JSON.py

import json
import logging
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set up trigger words indicating lists
trigger_words = ["References", "External Links"]


def is_likely_section_header(line):
    """
    Heuristic to determine if a line is likely a section header.
    """
    return (
        len(line) <= 80 and
        line[0].isupper() and
        not line.endswith((".", ":", "!", "?"))
        )


def parse_article(file_path):
    """
    Parse a single text file into a structured JSON format with sections.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    title = lines[0].strip() if lines else "Untitled"
    sections = []
    current_section = {"heading": "Introduction", "paragraphs": []}
    further_info = {"heading": "Further Information", "paragraphs": []}

    i = 1
    merging_further_info = False
    trigger_headers = {"References", "External links"}

    while i < len(lines):
        line = lines[i].strip()

        if is_likely_section_header(line):
            if line in trigger_headers:
                merging_further_info = True
                i += 1
                continue  # Don't add this as a new section
            elif merging_further_info:
                # Likely category-like heading; collect as a line instead
                further_info["paragraphs"].append(line)
                i += 1
                continue
            else:
                sections.append(current_section)
                current_section = {"heading": line, "paragraphs": []}
        else:
            if merging_further_info:
                further_info["paragraphs"].append(line)
            else:
                current_section["paragraphs"].append(line)

        i += 1

    sections.append(current_section)
    if further_info["paragraphs"]:
        sections.append(further_info)

    return {"title": title, "sections": sections}



def main():
    # Load configuration
    config = load_config()
    raw_dir = Path(config["data_paths"]["raw"])
    json_dir = Path(config["data_paths"]["jsons"])

    # Ensure output directory exists
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure directories exist and are empty
    clear_directory(json_dir)

    # Process each .txt file in the raw directory
    for file_path in raw_dir.glob("*.txt"):
        try:
            parsed = parse_article(file_path)
            out_path = json_dir / f"{file_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            logger.info(f"Converted {file_path.name} → {out_path.name}")
        except Exception as e:
            logger.warning(f"Failed to process {file_path.name}: {e}")

    logger.info(f"Finished. JSON files saved to: {json_dir}")

if __name__ == "__main__":
    main()