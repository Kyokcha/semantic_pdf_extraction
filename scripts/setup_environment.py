# scripts/setup_environment.py

import nltk
from sentence_transformers import SentenceTransformer

print("Downloading NLTK punkt tokenizer...")
nltk.download("punkt")

print("Downloading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Setup complete.")
