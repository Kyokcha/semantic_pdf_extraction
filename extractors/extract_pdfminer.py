"""Extract text from PDFs using pdfminer.six."""

from pdfminer.high_level import extract_text as pdfminer_extract


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using pdfminer.
    
    Args:
        pdf_path: Path to the PDF file to process.
    
    Returns:
        str: Extracted text from the PDF, stripped of leading/trailing whitespace.
             Returns empty string if extraction fails.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
    """
    try:
        text = pdfminer_extract(pdf_path)
        return text.strip() if text else ""
    except Exception as e:
        print(f"[pdfminer] Error processing {pdf_path}: {e}")
        return ""