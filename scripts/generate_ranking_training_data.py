# scripts/generate_training_data.py

import torch
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import load_config
from utils.file_operations import clear_directory


def load_dataframe(path):
    return pd.read_csv(path)


def load_embeddings(path):
    return torch.load(path)


def find_best_match(gt_embeddings, extracted_embeddings):
    similarities = cosine_similarity(extracted_embeddings, gt_embeddings)
    best_indices = similarities.argmax(axis=1)
    best_scores = similarities.max(axis=1)
    return best_indices, best_scores


def process_one_file(args):
    extracted_csv_path, gt_df_dir, gt_emb_dir, extracted_emb_dir, output_dir = args

    base_name = extracted_csv_path.stem
    article_id = base_name.split("-")[0]

    gt_csv_path = gt_df_dir / f"{article_id}.csv"
    gt_emb_path = gt_emb_dir / f"{article_id}.pt"
    extracted_emb_path = extracted_emb_dir / f"{base_name}.pt"

    if not all(p.exists() for p in [gt_csv_path, gt_emb_path, extracted_emb_path]):
        print(f"Skipping {base_name} due to missing files.")
        return

    gt_df = load_dataframe(gt_csv_path)
    extracted_df = load_dataframe(extracted_csv_path)
    gt_emb = load_embeddings(gt_emb_path).cpu().numpy()
    extracted_emb = load_embeddings(extracted_emb_path).cpu().numpy()

    best_match_indices, scores = find_best_match(gt_emb, extracted_emb)

    matched_gt_sentences = [gt_df.iloc[i]["sentence"] for i in best_match_indices]
    result_df = pd.DataFrame({
        "article_id": extracted_df["article_id"],
        "extractor": extracted_df["extractor"],
        "extracted_sentence": extracted_df["sentence"],
        "matched_gt_sentence": matched_gt_sentences,
        "similarity_score": scores
    })

    output_path = output_dir / f"{base_name}_training.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved training data for {base_name}")


def main():
    config = load_config()
    gt_df_dir = Path(config["data_paths"]["ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["embeddings_GT"])
    extracted_df_dir = Path("data/extracted_sentences")
    extracted_emb_dir = Path(config["data_paths"]["embeddings_PDF"])
    output_dir = Path(config["data_paths"]["training_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    extracted_files = list(extracted_df_dir.glob("*.csv"))
    print(f"Found {len(extracted_files)} extracted files to process.")

    args = [
        (path, gt_df_dir, gt_emb_dir, extracted_emb_dir, output_dir)
        for path in extracted_files
    ]

    num_workers = min(cpu_count() - 1, 8)
    with Pool(num_workers) as pool:
        pool.map(process_one_file, args)


if __name__ == "__main__":
    main()
