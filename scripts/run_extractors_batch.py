# scripts/run_extractors_batch.py

import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from extractors import extract_pypdf2, extract_pdfminer, extract_ocr
from utils.config import load_config
from utils.file_operations import clear_directory

# Define all available extractors (mapping name to function)
ALL_EXTRACTORS = {
    "pypdf2": extract_pypdf2.extract_text,
    "pdfminer": extract_pdfminer.extract_text,
    "ocr": extract_ocr.extract_text,
    # Add more as needed
}

def get_enabled_extractors(config):
    flags = config.get("extraction", {}).get("extractors", {})
    return {name: func for name, func in ALL_EXTRACTORS.items() if flags.get(name, False)}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pdf(args):
    pdf_path, output_dir, extractors = args
    base_name = pdf_path.stem

    for name, extractor in extractors.items():
        try:
            text = extractor(str(pdf_path)).strip()
            if text:
                output_file = output_dir / f"{base_name}-{name}.txt"
                output_file.write_text(text, encoding="utf-8")
                logger.info(f"[{name}] {output_file.name} ✓")
            else:
                logger.warning(f"[{name}] {base_name}: No text extracted.")
        except Exception as e:
            logger.error(f"[{name}] {base_name}: Failed - {e}")


def main():
    config = load_config()
    pdf_dir = Path(config["data_paths"]["pdfs"])
    output_dir = Path(config["data_paths"]["extracted"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)

    extractors = get_enabled_extractors(config)
    if not extractors:
        logger.warning("No extractors enabled in config.")
        return

    logger.info(f"Enabled extractors: {', '.join(extractors.keys())}")

    pdf_paths = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_paths)} PDF files to process.")

    args = [(path, output_dir, extractors) for path in pdf_paths]

    usable_cores = min(cpu_count() - 1, 8)
    logger.info(f"Using {usable_cores} CPU cores.")

    with Pool(processes=usable_cores) as pool:
        pool.map(process_pdf, args)


if __name__ == "__main__":
    main()