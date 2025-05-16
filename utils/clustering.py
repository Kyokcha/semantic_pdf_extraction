# utils/clustering.py

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import logging

logger = logging.getLogger(__name__)


def cluster_sentences(csv_files, embedding_files, output_csv_dir, output_pt_dir, threshold=0.8):
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_pt_dir.mkdir(parents=True, exist_ok=True)

    for article_id in sorted({f.stem.split("-")[0] for f in csv_files}):
        # Collect all rows and embeddings for this article
        csvs = [f for f in csv_files if f.stem.startswith(article_id)]
        pts = [f for f in embedding_files if f.stem.startswith(article_id)]

        if not csvs or not pts:
            logger.warning(f"Skipping {article_id}: no matching CSV or PT files.")
            continue

        all_df = []
        all_embeddings = []

        for csv, pt in zip(csvs, pts):
            df = pd.read_csv(csv)
            emb = torch.load(pt)

            embeddings_cpu = emb["embeddings"].cpu()
            df["embedding_idx"] = range(len(df))
            df["embedding"] = [v for v in embeddings_cpu]  # individual CPU tensors
            all_df.append(df)
            all_embeddings.append(embeddings_cpu)

        df_combined = pd.concat(all_df, ignore_index=True)
        embeddings = torch.cat(all_embeddings)

        if embeddings.shape[0] < 2:
            logger.warning(f"Skipping {article_id}: not enough data to cluster.")
            continue

        # Compute cosine distance matrix
        embeddings_np = embeddings.numpy()
        sim_matrix = cosine_similarity(embeddings_np)
        distance_matrix = 1.0 - sim_matrix

        # Perform clustering
        clusterer = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - threshold,
            n_clusters=None
        )
        labels = clusterer.fit_predict(distance_matrix)
        df_combined["cluster_id"] = labels

        # For each cluster, select the sentence closest to the cluster centroid
        clustered = []
        clustered_ids = []
        clustered_embeddings = []

        for cid, group in df_combined.groupby("cluster_id"):
            vectors = torch.stack(group["embedding"].tolist())
            centroid = vectors.mean(dim=0, keepdim=True)
            sims = cosine_similarity(centroid.numpy(), vectors.numpy())[0]
            best_idx = sims.argmax()
            selected = group.iloc[best_idx]

            clustered.append(selected)
            clustered_ids.append(selected["extracted_sentence_id"])
            clustered_embeddings.append(vectors[best_idx])

        out_df = pd.DataFrame(clustered)
        out_df = out_df.drop(columns=["embedding", "embedding_idx", "cluster_id"])

        out_csv_path = output_csv_dir / f"{article_id}_clustered.csv"
        out_df.to_csv(out_csv_path, index=False)

        out_pt_path = output_pt_dir / f"{article_id}_clustered.pt"
        torch.save({"ids": clustered_ids, "embeddings": torch.stack(clustered_embeddings)}, out_pt_path)

        logger.info(f"  ✓ {len(out_df)} sentences clustered into {len(set(labels))} clusters (avg {len(labels)/len(set(labels)):.2f} per cluster)")
        logger.info(f"  → Saved clustered results to {out_csv_path}")
        logger.info(f"  → Saved clustered embeddings to {out_pt_path}")
