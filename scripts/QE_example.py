"""Run quality estimation on a sample document with semantic similarity comparison."""
import logging
from pathlib import Path
from utils.config import load_config
from extractors import extract_pypdf2, extract_ocr, extract_pdfplumber
import pandas as pd
from utils.sentence_features import get_sbert_embedding
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_all_methods(pdf_path: Path) -> dict:
    extractors = {
        "pypdf2": extract_pypdf2.extract_text,
        "ocr": extract_ocr.extract_text,
        "plumber": extract_pdfplumber.extract_text
    }
    
    results = {}
    for name, extractor in extractors.items():
        try:
            text = extractor(str(pdf_path))
            results[name] = text
            logger.info(f"✓ {name} extraction complete")
        except Exception as e:
            logger.error(f"Failed to extract using {name}: {e}")
            results[name] = ""
    
    return results


def calculate_similarity(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = get_sbert_embedding(text1)
    emb2 = get_sbert_embedding(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0])


def create_line_by_line_comparison(texts: dict, gt_lines: list, base_name: str,
                                 results_dir: Path) -> None:
    results = []

    for i, gt_line in enumerate(gt_lines):
        extracted_lines = {}
        for extractor, full_text in texts.items():
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            extracted_lines[extractor] = lines[i] if i < len(lines) else ""
            similarities = {
            ext: calculate_similarity(gt_line, text)
            for ext, text in extracted_lines.items()
            }
        max_sim = max(similarities.values())
        best_extractors = [ext for ext, sim in similarities.items()
                        if abs(sim - max_sim) < 0.001]
        results.append({
        "line_number": i + 1,
        "ground_truth": gt_line,
        "pypdf2_text": extracted_lines["pypdf2"],
        "ocr_text": extracted_lines["ocr"],
        "plumber_text": extracted_lines["plumber"],
        "pypdf2_similarity": round(similarities["pypdf2"], 3),
        "ocr_similarity": round(similarities["ocr"], 3),
        "plumber_similarity": round(similarities["plumber"], 3),
        "multiple_best": len(best_extractors) > 1,
        "best_extractors": ", ".join(best_extractors)
        })

    pd.DataFrame(results).to_csv(results_dir / f"{base_name}_comparison.csv", index=False)
    with open(results_dir / f"{base_name}_comparison.txt", 'w') as f:
        f.write(f"Quality Estimation Results for {base_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Line {result['line_number']}:\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write("\nExtractions (with similarities):\n")
            f.write(f"PyPDF2 ({result['pypdf2_similarity']:.3f}): {result['pypdf2_text']}\n")
            f.write(f"OCR ({result['ocr_similarity']:.3f}): {result['ocr_text']}\n")
            f.write(f"PDFPlumber ({result['plumber_similarity']:.3f}): {result['plumber_text']}\n")
            if result['multiple_best']:
                f.write(f"\nMultiple best extractors: {result['best_extractors']}\n")
            f.write("-" * 80 + "\n\n")


def main():
    config = load_config()
    
    input_dir = Path(config["data_paths"]["DB_QE"])
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in input directory")
        return
    
    pdf_path = pdf_files[0]
    base_name = pdf_path.stem
    
    txt_path = input_dir / f"{base_name}.txt"
    if not txt_path.exists():
        logger.error(f"No corresponding text file found for {pdf_path.name}")
        return
    
    logger.info(f"Processing: {pdf_path.name}")
    
    results_dir = input_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    gt_lines = [line.strip() for line in txt_path.read_text().split('\n')
                if line.strip()]
    extracted_texts = extract_text_all_methods(pdf_path)
    
    create_line_by_line_comparison(extracted_texts, gt_lines, base_name, results_dir)
    logger.info(f"✓ All results saved to {results_dir}")


if __name__ == "__main__":
    main()