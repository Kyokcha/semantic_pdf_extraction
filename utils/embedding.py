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
    Loads a CSV with a 'sentence' column, computes sentence embeddings,
    and saves them as a .pt file using PyTorch.

    Args:
        csv_path (Path or str): Path to the input CSV.
        output_path (Path or str): Path to save the embeddings as a .pt file.
    """
    df = pd.read_csv(csv_path)

    if "sentence" not in df.columns:
        raise ValueError(f"'sentence' column not found in {csv_path}")

    sentences = df["sentence"].fillna("").tolist()
    model = get_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    torch.save(embeddings, output_path)

