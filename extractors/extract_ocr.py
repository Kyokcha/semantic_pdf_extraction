"""Extract text from PDFs using Optical Character Recognition (OCR)."""

import pytesseract
from pdf2image import convert_from_path
import tempfile


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using OCR.
    
    Args:
        pdf_path: Path to the PDF file to process.
        
    Returns:
        str: Extracted text from all pages concatenated with newlines.
             Returns empty string if extraction fails.
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        PermissionError: If the PDF file can't be accessed.
        
    Note:
        Uses a DPI of 300 for image conversion to balance quality and performance.
        Temporary images are automatically cleaned up after processing.
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