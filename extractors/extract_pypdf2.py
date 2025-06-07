"""Extract text from PDFs using PyPDF2."""

from PyPDF2 import PdfReader


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file to process.
    
    Returns:
        str: Extracted text from all pages, joined by newlines.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        PyPDF2.errors.PdfReadError: If the PDF is encrypted or corrupted.
    
    Note:
        Pages without extractable text are skipped.
    """
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text).strip()