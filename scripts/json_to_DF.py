# scripts/json_to_DF.py

import json
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
from utils.config import load_config
from utils.file_operations import clear_directory
from multiprocessing import Pool, cpu_count


def process_json(json_path_output_tuple):
    json_path, output_dir = json_path_output_tuple

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article_id = json_path.stem
    article_title = data.get("title", "")
    rows = []
    global_counter = 0

    for section in data.get("sections", []):
        section_title = section.get("heading", "Unknown")
        local_counter = 0

        for para in section.get("paragraphs", []):
            sentences = sent_tokenize(para)
            for sentence in sentences:
                rows.append({
                    "article_id": article_id,
                    "title": article_title,
                    "section": section_title,
                    "sentence_id": local_counter,
                    "global_sentence_id": global_counter,
                    "sentence": sentence.strip()
                })
                local_counter += 1
                global_counter += 1

    df = pd.DataFrame(rows)
    out_file = output_dir / f"{article_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"✓ Saved: {out_file.name}")


def main():
    config = load_config()
    input_dir = Path(config["data_paths"]["jsons"])
    output_dir = Path(config["data_paths"]["ground_truth"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear the directory before writing new files
    clear_directory(output_dir)

    json_paths = list(input_dir.glob("*.json"))
    print(f"Found {len(json_paths)} JSON files.")

    args = [(p, output_dir) for p in json_paths]
    cores = min(cpu_count() - 1, 8)
    print(f"Using {cores} CPU cores...")

    with Pool(processes=cores) as pool:
        pool.map(process_json, args)


if __name__ == "__main__":
    main()
