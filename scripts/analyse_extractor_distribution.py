# scripts/analyse_extractor_distribution.py

import pandas as pd
from pathlib import Path
import logging
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load paths from config
config = load_config()
base_dir = Path(config["data_paths"]["model_outputs"])

# Define input/output file paths
input_path = base_dir / "model_predictions.csv"
output_path = base_dir / "extractor_distribution_per_doc.csv"

# Read the predictions CSV
df = pd.read_csv(input_path)

# Count extractor usage per document
extractor_counts = df.groupby(['article_id', 'predicted_extractor']).size().unstack(fill_value=0)

# Convert counts to percentages
extractor_percentages = extractor_counts.div(extractor_counts.sum(axis=1), axis=0) * 100

# Save to CSV
extractor_percentages.to_csv(output_path)

logging.info(f"Extractor distribution per document saved to: {output_path}")

