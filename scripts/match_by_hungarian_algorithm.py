"""Match extracted sentences to ground truth using Hungarian algorithm and cosine similarity."""

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


def load_embeddings(path: Path) -> tuple[list[str], torch.Tensor]:
    """Load sentence embeddings from a PyTorch file.
    
    Args:
        path (Path): Path to the .pt file containing embeddings.
    
    Returns:
        tuple: Contains (sentence_ids, embeddings) where:
            - sentence_ids (list[str]): List of sentence identifiers
            - embeddings (torch.Tensor): Matrix of sentence embeddings
    """
    data = torch.load(path)
    return data["ids"], data["embeddings"].cpu().numpy()


def extract_layout_and_extractor(stem: str) -> tuple[str, str]:
    """Extract layout type and extractor name from filename.
    
    Args:
        stem (str): Filename stem to parse.
    
    Returns:
        tuple[str, str]: Contains (layout_type, extractor_name).
        Returns ('na', extractor) if no layout info found.
        Returns ('unknown', 'unknown') if format is unexpected.
    """
    parts = stem.split("-")
    if len(parts) == 2:
        return "na", parts[1]
    else:
        logger.warning(f"Unexpected filename format: {stem}")
        return "unknown", "unknown"


def process_article(args: tuple) -> None:
    """Match extracted sentences to ground truth for a single article.
    
    Args:
        args (tuple): Contains (article_id, gt_csv_dir, gt_emb_dir, 
                               extracted_csv_dir, extracted_emb_dir, output_dir)
    
    Note:
        Uses Hungarian algorithm to find optimal matching based on cosine similarity.
        Processes each extractor's output separately.
        Saves matches with similarity scores to CSV files.
    """
    article_id, gt_csv_dir, gt_emb_dir, extracted_csv_dir, extracted_emb_dir, output_dir = args

    gt_csv = gt_csv_dir / f"{article_id}_manual.csv"
    gt_emb = gt_emb_dir / f"{article_id}_manual.pt"

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
        cost_matrix = 1.0 - sim_matrix
        gt_idx, ext_idx = linear_sum_assignment(cost_matrix)

        matched_rows = []

        for g, e in zip(gt_idx, ext_idx):
            score = sim_matrix[g, e]
            gt_row = gt_df.iloc[g]
            matched_row = extracted_df.iloc[e]

            matched_rows.append({
                "gt_sentence_id": gt_row["gt_sentence_id"],
                "gt_sentence": gt_row["sentence"],
                "extracted_sentence_id": matched_row["extracted_sentence_id"],
                "extracted_sentence": matched_row["extracted_sentence"],
                "extractor": extractor,
                "layout": layout,
                "similarity_score": score
            })

        out_df = pd.DataFrame(matched_rows)
        output_path = output_dir / f"{base_name}_matched.csv"
        out_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved: {output_path.name} ({len(out_df)} matches)")


def main() -> None:
    """Match extracted sentences to ground truth for all articles.
    
    Loads embeddings for ground truth and extracted sentences, then finds
    optimal matches using the Hungarian algorithm and cosine similarity.
    
    Note:
        Uses (CPU core count - 1) up to max 8 cores for parallel processing.
        Output directory is cleared before processing starts.
        Skips articles with missing or mismatched data.
    """
    config = load_config()

    gt_csv_dir = Path(config["data_paths"]["DB_ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["DB_embeddings_GT"])
    extracted_csv_dir = Path(config["data_paths"]["DB_extracted_sentences"])
    extracted_emb_dir = Path(config["data_paths"]["DB_embeddings_PDF"])
    output_dir = Path(config["data_paths"]["DB_matched_sentences"])
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