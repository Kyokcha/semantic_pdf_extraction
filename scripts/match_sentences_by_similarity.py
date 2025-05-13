import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
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
    # article_001-two_column-pypdf2 → layout = two_column, extractor = pypdf2
    parts = stem.split("-")
    return parts[1], parts[2]


def process_article(args):
    """
    args is a tuple:
    (article_id, gt_csv_dir, gt_emb_dir, extracted_csv_dir, extracted_emb_dir, output_dir)
    """
    article_id, gt_csv_dir, gt_emb_dir, extracted_csv_dir, extracted_emb_dir, output_dir = args

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
        best_indices = np.argmax(sim_matrix, axis=1)
        best_scores = np.max(sim_matrix, axis=1)

        matched_rows = []
        for i, gt_row in gt_df.iterrows():
            best_idx = best_indices[i]
            best_score = best_scores[i]
            matched_row = extracted_df.iloc[best_idx]

            matched_rows.append({
                "gt_sentence_id": gt_row["gt_sentence_id"],
                "gt_sentence": gt_row["sentence"],
                "extracted_sentence_id": matched_row["extracted_sentence_id"],
                "extracted_sentence": matched_row["extracted_sentence"],
                "extractor": extractor,
                "layout": layout,
                "similarity_score": best_score
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
