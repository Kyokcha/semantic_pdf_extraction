"""Generate linguistic and semantic features from sentences for model training."""

import spacy
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models once (global)
nlp = spacy.load("en_core_web_sm")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_gpt2_perplexity(sentence: str) -> float | None:
    """Calculate GPT-2 perplexity score for a sentence.
    
    Args:
        sentence (str): Input text to evaluate.
    
    Returns:
        float | None: Perplexity score or None if calculation fails.
    
    Note:
        Truncates input to 512 tokens maximum.
    """
    try:
        encodings = gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()
    except Exception:
        return None


def get_sbert_embedding(sentence: str) -> np.ndarray:
    """Generate sentence embedding using SBERT.
    
    Args:
        sentence (str): Input text to embed.
    
    Returns:
        np.ndarray: 384-dimensional embedding vector.
        Returns zero vector if embedding fails.
    """
    try:
        return sbert_model.encode([sentence])[0]
    except Exception:
        return np.zeros(384)


def jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings.
    
    Args:
        a (str): First string to compare.
        b (str): Second string to compare.
    
    Returns:
        float: Similarity score between 0 and 1.
    
    Note:
        Tokenizes by whitespace and converts to lowercase before comparison.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if union else 0


def compute_sentence_features(text: str) -> dict:
    """Extract linguistic features from a sentence.
    
    Args:
        text (str): Input sentence to analyze.
    
    Returns:
        dict: Features including token counts, POS tags, perplexity,
              and first 5 dimensions of SBERT embedding.
    
    Note:
        Ignores spaces and punctuation in token counts.
        Uses spaCy for linguistic analysis and named entity recognition.
    """
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space and not token.is_punct]
    
    features = {
        "num_tokens": len(tokens),
        "num_chars": len(text),
        "num_nouns": sum(1 for t in tokens if t.pos_ == "NOUN"),
        "num_verbs": sum(1 for t in tokens if t.pos_ == "VERB"),
        "num_adjs": sum(1 for t in tokens if t.pos_ == "ADJ"),
        "num_advs": sum(1 for t in tokens if t.pos_ == "ADV"),
        "num_entities": len(doc.ents),
        "num_sentences": len(list(doc.sents)),
        "gpt2_perplexity": get_gpt2_perplexity(text),
        "num_words": len(text.split()),
        "avg_word_len": sum(len(w) for w in text.split()) / (len(text.split()) + 1e-6),
        "num_punct": sum(1 for c in text if c in ".,;!?-—()[]{}\"'"),
        "has_verb": int(any(t.pos_ == "VERB" for t in tokens))
    }

    emb = get_sbert_embedding(text)
    for i in range(5):  # optionally truncate
        features[f"sbert_dim_{i}"] = emb[i]

    return features


def inter_extractor_features(current_sentence: str, other_sentences: dict) -> dict:
    """Compare a sentence against versions from other extractors.
    
    Args:
        current_sentence (str): Sentence to compare.
        other_sentences (dict): Mapping of extractor names to their versions.
    
    Returns:
        dict: Jaccard and cosine similarities with each other extractor.
    
    Note:
        Cosine similarity uses SBERT embeddings for comparison.
        Returns 0.0 for cosine similarity if embedding fails.
    """
    features = {}
    curr_emb = get_sbert_embedding(current_sentence)

    for name, other_sentence in other_sentences.items():
        jacc = jaccard_similarity(current_sentence, other_sentence)
        features[f"jaccard_with_{name}"] = jacc

        try:
            other_emb = get_sbert_embedding(other_sentence)
            cos_sim = cosine_similarity([curr_emb], [other_emb])[0][0]
        except:
            cos_sim = 0.0

        features[f"cosine_sim_with_{name}"] = cos_sim

    return features


def sentence_features(row: dict, all_sentences: dict | None = None) -> dict:
    """Generate all features for a sentence, including inter-extractor comparisons.
    
    Args:
        row (dict): Row containing 'extracted_sentence', 'extractor', etc.
        all_sentences (dict, optional): Mapping of extractor names to sentences.
    
    Returns:
        dict: Combined linguistic and comparison features.
    
    Note:
        Combines basic linguistic features with inter-extractor comparisons
        when all_sentences is provided.
    """
    text = row.get("extracted_sentence", "")
    base_feats = compute_sentence_features(text)

    # Inter-extractor features
    if all_sentences:
        others = {k: v for k, v in all_sentences.items() if k != row.get("extractor")}
        base_feats.update(inter_extractor_features(text, others))

    return base_feats