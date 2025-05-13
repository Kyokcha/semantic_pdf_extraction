# scripts/setup_environment.py

import nltk
import spacy.cli
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

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

