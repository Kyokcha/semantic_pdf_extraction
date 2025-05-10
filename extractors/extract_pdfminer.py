# extractos/extract_pdfminer.py

from pdfminer.high_level import extract_text

def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfminer.six.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The extracted text from the entire PDF.
    """
    try:
        text = extract_text(pdf_path)
        return text.strip() if text else ""
    except Exception as e:
        print(f"[pdfminer] Error processing {pdf_path}: {e}")
        return ""
