# scripts/generate_ranking_training_data.py

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
    article_id, gt_df_dir, gt_emb_dir, extracted_df_dir, extracted_emb_dir, feature_dir, output_dir = args

    gt_csv_path = gt_df_dir / f"{article_id}.csv"
    gt_emb_path = gt_emb_dir / f"{article_id}.pt"

    if not gt_csv_path.exists() or not gt_emb_path.exists():
        print(f"Missing GT files for {article_id}")
        return

    gt_df = load_dataframe(gt_csv_path)
    gt_data = load_embeddings(gt_emb_path)
    gt_emb = gt_data["embeddings"].cpu().numpy()
    gt_ids = gt_data["ids"]

    # ✅ Ensure embeddings match the CSV row order
    if list(gt_df["gt_sentence_id"].astype(str)) != gt_ids:
        print(f"❌ GT ID mismatch for {article_id}")
        return

    extracted_files = list(extracted_df_dir.glob(f"{article_id}-*.csv"))
    all_rows = []

    for extracted_file in extracted_files:
        base_name = extracted_file.stem
        extracted_emb_path = extracted_emb_dir / f"{base_name}.pt"
        feature_file = feature_dir / f"{base_name}.csv"

        if not extracted_emb_path.exists() or not feature_file.exists():
            print(f"Missing data for {base_name}")
            continue

        extracted_df = load_dataframe(extracted_file)
        feature_df = load_dataframe(feature_file)
        extracted_data = load_embeddings(extracted_emb_path)
        extracted_emb = extracted_data["embeddings"].cpu().numpy()
        extracted_ids = extracted_data["ids"]

        # ✅ Make sure the sentence IDs match feature/embedding rows
        extracted_df["extracted_sentence_id"] = extracted_df["extracted_sentence_id"].astype(str)
        feature_df["extracted_sentence_id"] = feature_df["extracted_sentence_id"].astype(str)
        full_extracted = pd.merge(extracted_df, feature_df, on="extracted_sentence_id", how="left")

        # Compute cosine similarity between each GT and all extracted
        sim_matrix = cosine_similarity(gt_emb, extracted_emb)
        best_indices = sim_matrix.argmax(axis=1)
        best_scores = sim_matrix.max(axis=1)

        for i, gt_row in gt_df.iterrows():
            best_idx = best_indices[i]
            matched_row = full_extracted.iloc[best_idx]

            row = {
                "article_id": gt_row["article_id"],
                "gt_sentence_id": gt_row["gt_sentence_id"],
                "gt_sentence": gt_row["sentence"],
                "extractor": matched_row["extractor"],
                "matched_extracted_sentence": matched_row["extracted_sentence"],
                "similarity_score": best_scores[i],
            }

            # Add selected features
            feature_cols = [
                "num_chars", "num_words", "avg_word_len", "num_punct",
                "has_verb", "num_verbs", "num_nouns", "num_adjs", "num_advs",
                "gpt2_perplexity"
            ]
            for col in feature_cols:
                row[col] = matched_row.get(col, None)

            all_rows.append(row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        output_file = output_dir / f"{article_id}_training.csv"
        out_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file.name}")
    else:
        print(f"No rows generated for {article_id}")


def main():
    config = load_config()
    gt_df_dir = Path(config["data_paths"]["ground_truth"])
    gt_emb_dir = Path(config["data_paths"]["embeddings_GT"])
    extracted_df_dir = Path(config["data_paths"]["extracted_sentences"])
    extracted_emb_dir = Path(config["data_paths"]["embeddings_PDF"])
    feature_dir = Path(config["data_paths"]["features"])
    output_dir = Path(config["data_paths"]["training_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    article_ids = {f.stem.split("-")[0] for f in extracted_df_dir.glob("*.csv")}
    print(f"Found {len(article_ids)} articles to process.")

    args = [(aid, gt_df_dir, gt_emb_dir, extracted_df_dir, extracted_emb_dir, feature_dir, output_dir)
            for aid in article_ids]

    with Pool(min(cpu_count() - 1, 8)) as pool:
        pool.map(process_article, args)


if __name__ == "__main__":
    main()
