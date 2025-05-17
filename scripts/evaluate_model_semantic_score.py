# scripts/evaluate_model_semantic_score.py

import pandas as pd
from pathlib import Path
from utils.config import load_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    pred_path = Path(config["data_paths"]["DB_model_outputs"]) / "model_predictions.csv"
    train_path = Path(config["data_paths"]["DB_final_training_data"]) / "training_data.csv"
    output_dir = Path(config["data_paths"]["DB_evaluation_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "semantic_score_comparison.csv"

    # Load data
    logger.info("Loading model predictions and training data...")
    preds = pd.read_csv(pred_path)
    train = pd.read_csv(train_path)

    # Deduplicate training data in case of tie duplicates
    train = train.drop_duplicates(subset=["gt_sentence_id"], keep="first")

    # Merge on gt_sentence_id
    merged = preds.merge(train, on="gt_sentence_id", how="left")

    # Select relevant columns
    cols = [
        "article_id", "gt_sentence_id", "predicted_extractor",
        "similarity_score_pypdf2", "similarity_score_ocr", "similarity_score_plumber"
    ]
    data = merged[cols].copy()

    # Compute predicted extractor score for each row
    def get_predicted_score(row):
        key = f"similarity_score_{row['predicted_extractor']}"
        return row.get(key, None)

    data["predicted_similarity"] = data.apply(get_predicted_score, axis=1)

    # Group by article and compute means
    summary = data.groupby("article_id").agg({
        "predicted_similarity": "mean",
        "similarity_score_ocr": "mean",
        "similarity_score_plumber": "mean",
        "similarity_score_pypdf2": "mean"
    }).reset_index()

    # Save output
    summary.to_csv(output_path, index=False)
    logger.info(f"✓ Saved semantic similarity comparison to {output_path}")


if __name__ == "__main__":
    main()
