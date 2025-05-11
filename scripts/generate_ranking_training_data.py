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


def process_article(args):
    article_id, gt_df_dir, gt_emb_dir, extracted_df_dir, extracted_emb_dir, output_dir = args

    gt_csv_path = gt_df_dir / f"{article_id}.csv"
    gt_emb_path = gt_emb_dir / f"{article_id}.pt"

    if not gt_csv_path.exists() or not gt_emb_path.exists():
        print(f"Missing GT files for {article_id}")
        return

    gt_df = load_dataframe(gt_csv_path)
    gt_emb = load_embeddings(gt_emb_path).cpu().numpy()

    extracted_files = list(extracted_df_dir.glob(f"{article_id}-*.csv"))

    for extracted_file in extracted_files:
        base_name = extracted_file.stem
        extracted_emb_path = extracted_emb_dir / f"{base_name}.pt"
        if not extracted_emb_path.exists():
            print(f"Missing extracted embedding for {base_name}")
            continue

        extracted_df = load_dataframe(extracted_file)
        extracted_emb = load_embeddings(extracted_emb_path).cpu().numpy()

        similarities = cosine_similarity(gt_emb, extracted_emb)
        best_indices = similarities.argmax(axis=1)
        best_scores = similarities.max(axis=1)

        matched_extracted_sentences = [extracted_df.iloc[i]["sentence"] for i in best_indices]
        extractors = [extracted_df.iloc[i]["extractor"] for i in best_indices]

        result_df = pd.DataFrame({
            "article_id": gt_df["article_id"],
            "gt_sentence": gt_df["sentence"],
            "matched_extracted_sentence": matched_extracted_sentences,
            "extractor": extractors,
            "similarity_score": best_scores
        })

        output_file = output_dir / f"{base_name}_training.csv"
        result_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file.name}")


def main():
    config = load_config()
    gt_df_dir = Path(config["data_paths"]["ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["embeddings_GT"])
    extracted_df_dir = Path(config["data_paths"]["extracted_sentence"])
    extracted_emb_dir = Path(config["data_paths"]["embeddings_PDF"])
    output_dir = Path(config["data_paths"]["training_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    article_ids = {f.stem.split("-")[0] for f in extracted_df_dir.glob("*.csv")}
    print(f"Found {len(article_ids)} articles to process.")

    args = [(aid, gt_df_dir, gt_emb_dir, extracted_df_dir, extracted_emb_dir, output_dir) for aid in article_ids]

    with Pool(min(cpu_count() - 1, 8)) as pool:
        pool.map(process_article, args)


if __name__ == "__main__":
    main()