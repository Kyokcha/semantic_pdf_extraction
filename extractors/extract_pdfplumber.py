# extractors/extract_pdf_plumber.py

import pdfplumber


def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from all pages, joined by newlines.
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
