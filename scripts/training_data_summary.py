# scripts/training_data_summary.py

import pandas as pd
from pathlib import Path
from collections import defaultdict
from utils.config import load_config


def main():
    config = load_config()

    input_path = Path(config["data_paths"]["DB_final_training_data"]) / "training_data.csv"
    output_dir = Path(config["data_paths"]["DB_extractor_summary"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "extractor_summary.csv"

    df = pd.read_csv(input_path)

    if "gt_sentence_id" not in df.columns:
        print("Missing 'gt_sentence_id' column.")
        return

    df["document"] = df["gt_sentence_id"].apply(lambda x: x.split("_manual_")[0])

    summary_rows = []

    for doc, group in df.groupby("document"):
        best_counts = defaultdict(float)
        total = 0

        for _, row in group.iterrows():
            scores = {
                "ocr": row.get("similarity_score_ocr", 0),
                "pypdf2": row.get("similarity_score_pypdf2", 0),
                "pdfplumber": row.get("similarity_score_plumber", 0),
            }

            if all(pd.isna(v) or v == 0 for v in scores.values()):
                continue

            max_score = max(scores.values())
            best_extractors = [k for k, v in scores.items() if v == max_score and v > 0]
            if not best_extractors:
                continue

            share = 1 / len(best_extractors)
            for extractor in best_extractors:
                best_counts[extractor] += share
            total += 1

        summary_rows.append({
            "document": doc,
            "ocr": round(best_counts["ocr"] / total, 4) if total else 0.0,
            "pypdf2": round(best_counts["pypdf2"] / total, 4) if total else 0.0,
            "pdfplumber": round(best_counts["pdfplumber"] / total, 4) if total else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("document", inplace=True)
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved extractor summary to: {output_path}")


if __name__ == "__main__":
    main()
