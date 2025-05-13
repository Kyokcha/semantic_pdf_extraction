# utils/sentence_features.py

import string
import spacy
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load spaCy model for POS tagging and tokenization
nlp = spacy.load("en_core_web_sm")

# Load lightweight GPT-2 model and tokenizer for perplexity scoring
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
gpt2_model.eval()  # Set to evaluation mode (no gradient tracking)


def sentence_features(sentence: str, extractor_name: str) -> dict:
    """
    Extract various linguistic and semantic features from a sentence.
    These can be used to assess sentence quality or train a model.

    Args:
        sentence (str): Sentence to extract features from.
        extractor_name (str): Which extractor produced this sentence.

    Returns:
        dict: Dictionary of extracted features.
    """
    doc = nlp(sentence)
    tokens = [token.text for token in doc]

    # Surface-level features
    num_chars = len(sentence)
    num_words = len(tokens)
    avg_word_len = sum(len(w) for w in tokens) / max(len(tokens), 1)
    num_punct = sum(1 for c in sentence if c in string.punctuation)

    # Part-of-speech (POS) feature counts
    pos_counts = {pos: 0 for pos in ['VERB', 'NOUN', 'ADJ', 'ADV']}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

    has_verb = pos_counts["VERB"] > 0

    # GPT-2 perplexity: lower = more semantically plausible sentence
    try:
        encoded = gpt2_tokenizer.encode(sentence, return_tensors='pt')
        with torch.no_grad():
            output = gpt2_model(encoded, labels=encoded)
            loss = output.loss
            perplexity = torch.exp(loss).item()
    except Exception:
        # If model fails (e.g., empty sentence), return high perplexity
        perplexity = 1000

    return {
        "sentence": sentence,
        "extractor": extractor_name,
        "num_chars": num_chars,
        "num_words": num_words,
        "avg_word_len": avg_word_len,
        "num_punct": num_punct,
        "has_verb": has_verb,
        "num_verbs": pos_counts["VERB"],
        "num_nouns": pos_counts["NOUN"],
        "num_adjs": pos_counts["ADJ"],
        "num_advs": pos_counts["ADV"],
        "gpt2_perplexity": perplexity
    }
