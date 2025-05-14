# scripts/match_sentences_by_similarity.py

import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from utils.config import load_config
from utils.file_operations import clear_directory
import logging
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(path):
    data = torch.load(path)
    return data["ids"], data["embeddings"].cpu().numpy()


def extract_layout_and_extractor(stem):
    parts = stem.split("-")
    return parts[1], parts[2]


def process_article(args):
    article_id, gt_csv_dir, gt_emb_dir, extracted_csv_dir, extracted_emb_dir, output_dir = args

    threshold = 0.85  # Similarity threshold for valid matches

    gt_csv = gt_csv_dir / f"{article_id}.csv"
    gt_emb = gt_emb_dir / f"{article_id}.pt"

    if not gt_csv.exists() or not gt_emb.exists():
        logger.warning(f"Missing GT data for {article_id}")
        return

    gt_df = pd.read_csv(gt_csv)
    gt_ids, gt_vectors = load_embeddings(gt_emb)

    if list(gt_df["gt_sentence_id"].astype(str)) != gt_ids:
        logger.error(f"ID mismatch in GT embeddings for {article_id}")
        return

    extracted_files = list(extracted_emb_dir.glob(f"{article_id}-*.pt"))

    for ext_emb_path in extracted_files:
        base_name = ext_emb_path.stem
        extracted_csv = extracted_csv_dir / f"{base_name}.csv"
        if not extracted_csv.exists():
            logger.warning(f"Missing extracted CSV for {base_name}")
            continue

        extracted_df = pd.read_csv(extracted_csv)
        ext_ids, ext_vectors = load_embeddings(ext_emb_path)

        if list(extracted_df["extracted_sentence_id"].astype(str)) != ext_ids:
            logger.error(f"ID mismatch in extracted embeddings for {base_name}")
            continue

        layout, extractor = extract_layout_and_extractor(base_name)

        sim_matrix = cosine_similarity(gt_vectors, ext_vectors)

        # Hungarian algorithm for optimal matching (minimizing cost = 1 - similarity)
        cost_matrix = 1.0 - sim_matrix
        gt_idx, ext_idx = linear_sum_assignment(cost_matrix)

        matched_rows = []
        used_ext_idx = set()

        for g, e in zip(gt_idx, ext_idx):
            score = sim_matrix[g, e]
            gt_row = gt_df.iloc[g]

            if score >= threshold:
                matched_row = extracted_df.iloc[e]
                used_ext_idx.add(e)

                matched_rows.append({
                    "gt_sentence_id": gt_row["gt_sentence_id"],
                    "gt_sentence": gt_row["sentence"],
                    "extracted_sentence_id": matched_row["extracted_sentence_id"],
                    "extracted_sentence": matched_row["extracted_sentence"],
                    "extractor": extractor,
                    "layout": layout,
                    "similarity_score": score
                })
            else:
                matched_rows.append({
                    "gt_sentence_id": gt_row["gt_sentence_id"],
                    "gt_sentence": gt_row["sentence"],
                    "extracted_sentence_id": None,
                    "extracted_sentence": None,
                    "extractor": extractor,
                    "layout": layout,
                    "similarity_score": 0.0
                })

        out_df = pd.DataFrame(matched_rows)
        output_path = output_dir / f"{base_name}_matched.csv"
        out_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved: {output_path.name}")


def main():
    config = load_config()

    gt_csv_dir = Path(config["data_paths"]["ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["embeddings_GT"])
    extracted_csv_dir = Path(config["data_paths"]["extracted_sentences"])
    extracted_emb_dir = Path(config["data_paths"]["embeddings_PDF"])
    output_dir = Path(config["data_paths"]["matched_sentences"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    article_ids = {f.stem.split("-")[0] for f in extracted_emb_dir.glob("*.pt")}
    logger.info(f"Found {len(article_ids)} articles to process.")

    args_list = [
        (aid, gt_csv_dir, gt_emb_dir, extracted_csv_dir, extracted_emb_dir, output_dir)
        for aid in sorted(article_ids)
    ]

    cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {cores} CPU cores")

    with Pool(processes=cores) as pool:
        pool.map(process_article, args_list)


if __name__ == "__main__":
    main()
