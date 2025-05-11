# scripts/training_data_summary.py

import pandas as pd
from pathlib import Path
from collections import defaultdict
from utils.config import load_config
from utils.file_operations import clear_directory
import re


def extract_key_parts(filename_stem):
    """
    Extract article_id, layout, extractor from filename like article_001-one_column-ocr_training.csv
    """
    match = re.match(r"(article_\d+)-(.*?)-(.*?)_training", filename_stem)
    if match:
        return match.groups()  # (article_id, layout, extractor)
    else:
        return None, None, None


def main():
    config = load_config()
    input_dir = Path(config["data_paths"]["training_data"])
    output_dir = Path(config["data_paths"]["extractor_summary"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)
    output_file = output_dir / "summary.csv"

    all_files = list(input_dir.glob("*_training.csv"))
    if not all_files:
        print(f"No training data files found in {input_dir}.")
        return

    # Group files by (article_id, layout)
    grouped_files = defaultdict(list)
    for file in all_files:
        article_id, layout, extractor = extract_key_parts(file.stem)
        if article_id and layout and extractor:
            grouped_files[(article_id, layout)].append((file, extractor))

    summary_rows = []

    for (article_id, layout), files in grouped_files.items():
        dfs = []
        for file, extractor in files:
            df = pd.read_csv(file)
            if df.empty:
                continue
            df["extractor"] = extractor
            dfs.append(df)

        if not dfs:
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        best_counts = defaultdict(float)
        total_sentences = 0

        for _, group in combined_df.groupby("gt_sentence"):
            max_score = group["similarity_score"].max()
            top_extractors = group[group["similarity_score"] == max_score]["extractor"].tolist()
            share = 1 / len(top_extractors)
            for extractor in top_extractors:
                best_counts[extractor] += share
            total_sentences += 1

        row = {
            "article": article_id,
            "layout": layout
        }
        for extractor in ["plumber", "ocr", "pypdf2"]:
            score = best_counts.get(extractor, 0.0)
            row[extractor] = round((score / total_sentences) * 100, 1) if total_sentences else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.fillna(0)
    summary_df.to_csv(output_file, index=False)
    print(f"✓ Saved extractor summary to {output_file}")


if __name__ == "__main__":
    main()
