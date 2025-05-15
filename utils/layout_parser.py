# utils/column_detection.py

from sklearn.cluster import KMeans
import numpy as np


def cluster_columns(tokens, max_columns=3):
    x_vals = np.array([[t["x0"]] for t in tokens])
    kmeans = KMeans(n_clusters=max_columns, random_state=0).fit(x_vals)

    for token, cluster_id in zip(tokens, kmeans.labels_):
        token["column_id"] = cluster_id

    cluster_x0s = {i: [] for i in range(max_columns)}
    for token in tokens:
        cluster_x0s[token["column_id"]].append(token["x0"])

    avg_x0 = {k: sum(v)/len(v) for k, v in cluster_x0s.items()}
    sorted_column_ids = sorted(avg_x0, key=avg_x0.get)
    column_map = {old: new for new, old in enumerate(sorted_column_ids)}

    for token in tokens:
        token["column_id"] = column_map[token["column_id"]]

    return tokens
