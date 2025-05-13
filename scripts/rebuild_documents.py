# scripts/rebuild_documents.py

import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.file_operations import clear_directory

# Load model predictions
df = pd.read_csv("data/model_outputs/model_predictions.csv")

# Extract article_id, sentence_number, layout
df[['article_id', 'sentence_number', 'layout']] = df['gt_sentence_id'].str.extract(
    r'^(article_\d+)_(\d+)__([a-zA-Z0-9_]+)'
)
df['sentence_number'] = df['sentence_number'].astype(int)

# Group and rebuild
config = load_config()
output_dir = Path(config["data_paths"]["rebuilt_documents"])
output_dir.mkdir(parents=True, exist_ok=True)
clear_directory(output_dir)

for (article_id, layout), group in df.groupby(['article_id', 'layout']):
    group_sorted = group.sort_values('sentence_number')
    sentences = group_sorted['selected_sentence'].tolist()

    full_text = "\n".join(sentences).strip()

    filename = f"{article_id}-{layout}.txt"
    (output_dir / filename).write_text(full_text, encoding='utf-8')

    print(f"Saved {filename}")
