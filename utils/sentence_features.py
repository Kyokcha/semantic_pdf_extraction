# utils/sentence_features.py

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


def get_gpt2_perplexity(sentence):
    try:
        encodings = gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()
    except Exception:
        return None


def get_sbert_embedding(sentence):
    try:
        return sbert_model.encode([sentence])[0]
    except Exception:
        return np.zeros(384)


def jaccard_similarity(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if union else 0


def compute_sentence_features(text):
    """Basic NLP features for a single sentence."""
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


def inter_extractor_features(current_sentence, other_sentences):
    """Compare this sentence to the same sentence from other extractors."""
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


def sentence_features(row, all_sentences=None):
    """
    Master function that computes all features for a sentence.

    Args:
        row (dict): contains 'extracted_sentence', 'extractor', 'sentence_id', etc.
        all_sentences (dict): mapping {extractor_name: sentence_text}
    """
    text = row.get("extracted_sentence", "")
    base_feats = compute_sentence_features(text)

    # Inter-extractor features
    if all_sentences:
        others = {k: v for k, v in all_sentences.items() if k != row.get("extractor")}
        base_feats.update(inter_extractor_features(text, others))

    return base_feats

