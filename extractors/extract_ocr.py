#extractors/extract_ocr.py

import pytesseract
from pdf2image import convert_from_path
import tempfile


def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using OCR (pytesseract + pdf2image).

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The extracted text from all pages.
    """
    text_output = []

    try:
        # Render PDF to images
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path, dpi=300, output_folder=temp_dir)

            for i, image in enumerate(images):
                # Perform OCR on each image
                page_text = pytesseract.image_to_string(image)
                text_output.append(page_text)

        return "\n".join(text_output).strip()

    except Exception as e:
        print(f"[ocr] Error processing {pdf_path}: {e}")
        return ""
