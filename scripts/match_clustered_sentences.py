# scripts/match_clustered_sentences.py

import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from utils.config import load_config
from utils.file_operations import clear_directory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(path):
    data = torch.load(path)
    return data["ids"], data["embeddings"].cpu().numpy()


def process_article(article_id, gt_csv_dir, gt_emb_dir, clustered_csv_dir, clustered_emb_dir, output_dir):
    gt_csv = gt_csv_dir / f"{article_id}_manual.csv"
    gt_emb = gt_emb_dir / f"{article_id}_manual.pt"
    clustered_csv = clustered_csv_dir / f"{article_id}_clustered.csv"
    clustered_emb = clustered_emb_dir / f"{article_id}_clustered.pt"

    if not gt_csv.exists() or not gt_emb.exists():
        logger.warning(f"Missing ground truth data for {article_id}")
        return

    if not clustered_csv.exists() or not clustered_emb.exists():
        logger.warning(f"Missing clustered data for {article_id}")
        return

    gt_df = pd.read_csv(gt_csv)
    clustered_df = pd.read_csv(clustered_csv)

    gt_ids, gt_vectors = load_embeddings(gt_emb)
    clus_ids, clus_vectors = load_embeddings(clustered_emb)

    if list(gt_df["gt_sentence_id"].astype(str)) != gt_ids:
        logger.error(f"GT ID mismatch for {article_id}")
        return

    if list(clustered_df["extracted_sentence_id"].astype(str)) != clus_ids:
        logger.error(f"Clustered ID mismatch for {article_id}")
        return

    # Compute similarity matrix
    sim_matrix = cosine_similarity(gt_vectors, clus_vectors)
    cost_matrix = 1.0 - sim_matrix
    gt_idx, clus_idx = linear_sum_assignment(cost_matrix)

    matched_rows = []
    for g, c in zip(gt_idx, clus_idx):
        score = sim_matrix[g, c]
        gt_row = gt_df.iloc[g]
        clus_row = clustered_df.iloc[c]

        matched_rows.append({
            "gt_sentence_id": gt_row["gt_sentence_id"],
            "gt_sentence": gt_row["sentence"],
            "clustered_sentence_id": clus_row["extracted_sentence_id"],
            "clustered_sentence": clus_row["extracted_sentence"],
            "extractor": clus_row["extractor"],
            "similarity_score": score
        })

    out_df = pd.DataFrame(matched_rows)
    output_path = output_dir / f"{article_id}_matched.csv"
    out_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path.name} ({len(out_df)} matches)")


def main():
    config = load_config()

    gt_csv_dir = Path(config["data_paths"]["DB_ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["DB_embeddings_GT"])
    clustered_csv_dir = Path(config["data_paths"]["DB_clustered_sentences"])
    clustered_emb_dir = Path(config["data_paths"]["DB_clustered_embeddings"])
    output_dir = Path(config["data_paths"]["DB_matched_sentences"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    article_ids = sorted({f.stem.replace("_clustered", "") for f in clustered_emb_dir.glob("*.pt")})
    logger.info(f"Found {len(article_ids)} articles to process.")

    for aid in article_ids:
        process_article(aid, gt_csv_dir, gt_emb_dir, clustered_csv_dir, clustered_emb_dir, output_dir)


if __name__ == "__main__":
    main()
