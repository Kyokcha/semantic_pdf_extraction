# scripts/generate_gt_sentences.py

import json
import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory
from multiprocessing import Pool, cpu_count


def process_json(json_path_output_tuple):
    json_path, output_dir = json_path_output_tuple

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article_id = json_path.stem
    rows = []

    for i, entry in enumerate(data):
        rows.append({
            "gt_sentence_id": f"{article_id}_{i}",
            "article_id": article_id,
            "section": entry["type"],  # Use 'type' field in place of section heading
            "sentence_id": i,
            "sentence": entry["sentence"]
        })

    df = pd.DataFrame(rows)
    out_file = output_dir / f"{article_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"✓ Saved GT: {out_file.name}")


def main():
    config = load_config()
    input_dir = Path(config["data_paths"]["DB_jsons"])
    output_dir = Path(config["data_paths"]["DB_ground_truth"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
