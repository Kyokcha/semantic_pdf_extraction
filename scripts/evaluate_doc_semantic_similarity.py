# scripts/evaluation_doc_semantic_similarity.py

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from utils.config import load_config
from utils.file_operations import clear_directory

# Load paths
config = load_config()
rebuilt_dir = Path(config["data_paths"]["rebuilt_documents"])
gt_dir = Path(config["data_paths"]["raw"])
extracted_dir = Path(config["data_paths"]["extracted"])
output_dir = Path(config["data_paths"]["evaluation_outputs"]) 
output_path = output_dir / "semantic_scores.csv"

# Ensure output directory exists and is clean
output_dir.mkdir(parents=True, exist_ok=True)
clear_directory(output_dir)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

results = []

for rebuilt_file in rebuilt_dir.glob("*.txt"):
    # Format: article_001-header_footer.txt
    stem_parts = rebuilt_file.stem.split("-")
    article_id = stem_parts[0]  # e.g., "article_001"
    layout = "-".join(stem_parts[1:])  # e.g., "header_footer"

    # Ground truth path (raw/article_001.txt)
    gt_path = gt_dir / f"{article_id}.txt"
    if not gt_path.exists():
        print(f"Ground truth not found for {article_id}")
        continue

    # Load ground truth and model-rebuilt text
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    with open(rebuilt_file, "r", encoding="utf-8") as f:
        model_text = f.read().strip()

    sim_model = util.cos_sim(model.encode(gt_text), model.encode(model_text)).item()

    # Check each extractor version
    extractor_scores = {}
    for extractor in ["pypdf2", "ocr", "plumber"]:
        ext_filename = f"{article_id}-{layout}-{extractor}.txt"
        ext_path = extracted_dir / ext_filename
        if ext_path.exists():
            with open(ext_path, "r", encoding="utf-8") as f:
                ext_text = f.read().strip()
            score = util.cos_sim(model.encode(gt_text), model.encode(ext_text)).item()
            extractor_scores[extractor] = score
        else:
            extractor_scores[extractor] = None

    # Determine winner
    scores_all = {"model": sim_model, **extractor_scores}
    winner = max(scores_all.items(), key=lambda x: x[1] if x[1] is not None else -1)[0]

    results.append({
        "article_id": article_id,
        "layout": layout,
        "sim_model": sim_model,
        "sim_pypdf2": extractor_scores["pypdf2"],
        "sim_ocr": extractor_scores["ocr"],
        "sim_plumber": extractor_scores["plumber"],
        "winner": winner,
    })

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_path, index=False)
print(f"Semantic similarity results saved to: {output_path}")

# ---------- Summary Output ----------

print("\n🔢 Overall Win Rate:")
overall = df_results["winner"].value_counts(normalize=True).mul(100).round(2)
print(overall.to_string())

print("\n📊 Win Rate by Layout:")
layout_summary = (
    df_results.groupby("layout")["winner"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .mul(100)
    .round(2)
)
print(layout_summary.to_string())
