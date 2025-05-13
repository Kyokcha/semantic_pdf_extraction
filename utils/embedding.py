# utils/embedding.py

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

_model = None  # Global variable to hold the loaded model


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_sentences_from_csv(csv_path, output_path):
    """
    Loads a CSV with a sentence and sentence ID column, computes sentence embeddings,
    and saves them along with IDs as a .pt file using PyTorch.

    Args:
        csv_path (Path or str): Path to the input CSV.
        output_path (Path or str): Path to save the embeddings dictionary as a .pt file.
    """
    df = pd.read_csv(csv_path)

    # Auto-detect sentence column
    for col in ["sentence", "extracted_sentence", "gt_sentence"]:
        if col in df.columns:
            sentence_col = col
            break
    else:
        raise ValueError(f"No valid sentence column found in {csv_path}")

    # Auto-detect ID column
    for id_col in ["gt_sentence_id", "extracted_sentence_id"]:
        if id_col in df.columns:
            break
    else:
        raise ValueError(f"No valid sentence ID column found in {csv_path}")

    sentences = df[sentence_col].fillna("").tolist()
    ids = df[id_col].tolist()

    model = get_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)

    torch.save({"ids": ids, "embeddings": embeddings}, output_path)


