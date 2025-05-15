#scripts/evalute_sentence_level_model_vs_extractors.py

import pandas as pd
import logging
from pathlib import Path
from utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config
config = load_config()
training_data_path = Path(config["data_paths"]["final_training_data"]) / "training_data.csv"
predictions_path = Path(config["data_paths"]["model_outputs"]) / "model_predictions.csv"
output_path = Path(config["data_paths"]["evaluation_outputs"]) / "sentence_level_similarity_evaluation.csv"

# Load data
df_preds = pd.read_csv(predictions_path)
df_full = pd.read_csv(training_data_path)

# Merge on gt_sentence_id
df_merged = pd.merge(df_preds, df_full, on="gt_sentence_id", how="left")

# Extract the model-selected similarity score
def get_selected_score(row):
    extractor = row["predicted_extractor"]
    return row.get(f"similarity_score_{extractor}", None)

df_merged["model_similarity_score"] = df_merged.apply(get_selected_score, axis=1)

# Calculate mean scores
mean_model = df_merged["model_similarity_score"].mean()
mean_pypdf2 = df_merged["similarity_score_pypdf2"].mean()
mean_ocr = df_merged["similarity_score_ocr"].mean()
mean_plumber = df_merged["similarity_score_plumber"].mean()

# Save sentence-level output
output_path.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_csv(output_path, index=False)

# Log summary
logging.info(f"Saved sentence-level evaluation to: {output_path}")
logging.info(f"Average model-picked similarity score: {mean_model:.4f}")
logging.info(f"Average pypdf2 similarity score:       {mean_pypdf2:.4f}")
logging.info(f"Average ocr similarity score:           {mean_ocr:.4f}")
logging.info(f"Average plumber similarity score:       {mean_plumber:.4f}")