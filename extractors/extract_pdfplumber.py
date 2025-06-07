"""Extract text from PDFs using pdfplumber."""

import pdfplumber


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file to process.
    
    Returns:
        str: Extracted text from all pages, joined by newlines.
             Returns empty string if extraction fails.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        pdfplumber.pdfminer.pdfparser.PDFSyntaxError: If the PDF is corrupted.
    
    Note:
        Pages without extractable text are skipped.
    """
    full_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)

        return "\n".join(full_text).strip()

    except Exception as e:
        print(f"[pdfplumber] Error processing {pdf_path}: {e}")
        return ""