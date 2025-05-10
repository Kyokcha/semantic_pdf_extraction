# extractors/extract_pypdf2.py

from PyPDF2 import PdfReader
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF using PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The full extracted text, joined by newlines.
    """
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text).strip()
