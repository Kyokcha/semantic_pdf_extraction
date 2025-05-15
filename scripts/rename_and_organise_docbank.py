from pathlib import Path
import shutil
import logging
from utils.config import load_config
from utils.pdf_utils import extract_single_pdf_page  # <-- utility for page extraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    source_dir = Path(config["data_paths"]["DB_dump"])
    txt_dest = Path(config["data_paths"]["DB_raw"])
    pdf_dest = Path(config["data_paths"]["DB_pdfs"])

    txt_dest.mkdir(parents=True, exist_ok=True)
    pdf_dest.mkdir(parents=True, exist_ok=True)

    pdf_files = list(source_dir.glob("*_black.pdf"))
    txt_files = list(source_dir.glob("*.txt"))

    matched_pairs = []

    for pdf in sorted(pdf_files):
        base_prefix = pdf.name.replace("_black.pdf", "")
        match = next((txt for txt in txt_files if txt.name.startswith(base_prefix)), None)
        if match:
            matched_pairs.append((pdf, match))

    logger.info(f"Found {len(matched_pairs)} valid .pdf/.txt pairs to copy.")

    for i, (pdf_src, txt_src) in enumerate(matched_pairs, start=1):
        new_name = f"doc_{i:03d}"
        pdf_dst = pdf_dest / f"{new_name}.pdf"
        txt_dst = txt_dest / f"{new_name}.txt"

        # Save only the relevant PDF page
        try:
            page_str = txt_src.stem.split("_")[-1]
            page_num = int(page_str)
            extract_single_pdf_page(pdf_src, page_num, pdf_dst)
        except Exception as e:
            logger.error(f"❌ Failed to extract page from {pdf_src.name}: {e}")
            continue

        shutil.copy2(txt_src, txt_dst)
        logger.info(f"✓ {pdf_src.name} + {txt_src.name} → {new_name}.pdf / .txt")

    logger.info("✅ Done copying and extracting PDF pages.")


if __name__ == "__main__":
    main()
