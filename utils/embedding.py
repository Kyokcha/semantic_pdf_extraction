"""Generate and manage sentence embeddings using sentence-transformers."""

import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

_model = None  # Global variable to hold the loaded model


def get_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer model.
    
    Returns:
        SentenceTransformer: Loaded model instance.
        
    Note:
        Uses singleton pattern to avoid reloading model multiple times.
        Default model is 'all-MiniLM-L6-v2'.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_sentences_from_csv(csv_path: str | Path, output_path: str | Path) -> None:
    """Generate embeddings for sentences from a CSV file.
    
    Args:
        csv_path (str | Path): Path to input CSV file.
        output_path (str | Path): Path to save embeddings as .pt file.
    
    Note:
        Auto-detects sentence column from: 'sentence', 'extracted_sentence', 'gt_sentence'.
        Auto-detects ID column from: 'gt_sentence_id', 'extracted_sentence_id'.
        Empty sentences are replaced with empty string before embedding.
    
    Raises:
        ValueError: If no valid sentence or ID column is found in CSV.
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