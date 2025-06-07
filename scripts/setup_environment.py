"""Download and initialize required NLP models and resources."""

import nltk
import spacy.cli
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def setup_environment() -> None:
    """Download all required models and resources.
    
    Downloads:
        - NLTK punkt tokenizer for sentence splitting
        - MiniLM sentence transformer for embeddings
        - SpaCy English model for text processing
        - DistilGPT2 model and tokenizer for language modeling
    
    Note:
        Models are cached after first download.
        Requires approximately 2GB of disk space.
    """
    print("🔽 Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")

    print("🔽 Downloading sentence-transformers model...")
    SentenceTransformer("all-MiniLM-L6-v2")

    print("🔽 Downloading SpaCy 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")

    print("🔽 Downloading GPT-2 model and tokenizer...")
    GPT2LMHeadModel.from_pretrained("distilgpt2")
    GPT2TokenizerFast.from_pretrained("distilgpt2")

    print("✅ Setup complete.")


if __name__ == "__main__":
    setup_environment()