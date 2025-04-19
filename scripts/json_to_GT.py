import json
from pathlib import Path
from utils.config import load_config
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import find
nltk.download('punkt_tab')
import pandas as pd


def json_to_sentence_dataframe(json_data, doc_title="Untitled"):
    """
    Converts a structured JSON document into a flat dataframe of sentence-level ground truth.
    
    Args:
        json_data (dict): JSON document as per your format.
        doc_title (str): Optional document title tag.
        
    Returns:
        pd.DataFrame: A dataframe with sentence-level breakdown.
    """
    rows = []
    section_counter = 0

    for section in json_data.get("sections", []):
        section_heading = section.get("heading", f"Section_{section_counter}")
        paragraphs = section.get("paragraphs", [])
        
        for para_idx, para in enumerate(paragraphs):
            paragraph_id = f"S{section_counter:02d}_P{para_idx:02d}"
            sentences = sent_tokenize(para)

            for sent_idx, sentence in enumerate(sentences):
                sentence_id = f"{paragraph_id}_S{sent_idx:02d}"
                rows.append({
                    "doc_title": doc_title,
                    "section_heading": section_heading,
                    "paragraph_id": paragraph_id,
                    "sentence_id": sentence_id,
                    "sentence_text": sentence.strip()
                })

        section_counter += 1

    return pd.DataFrame(rows)


def main():
    # ==== CONFIGURATION ====
    config = load_config()
    json_dir = Path(config["data_paths"]["jsons"])
    ground_truth_dir = Path(config["data_paths"]["ground_truth"])

    # Make output directory for files
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # Quick check to ensure punkt is downloaded
    try:
        find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Run:")
        print("    poetry run python -c \"import nltk; nltk.download('punkt')\"")
        raise

    # ==== PROCESS JSON FILES ====
    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_title = data.get("title", json_file.stem)
        df = json_to_sentence_dataframe(data, doc_title=doc_title)

        output_file = ground_truth_dir / f"{json_file.stem}_sentences.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Processed {json_file.name} -> {output_file.name}")


if __name__ == "__main__":
    main()