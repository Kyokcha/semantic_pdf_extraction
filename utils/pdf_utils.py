"""PDF manipulation utilities using PyMuPDF."""

import fitz  # PyMuPDF


def extract_single_pdf_page(pdf_path: str | Path, page_num: int, output_path: str | Path) -> None:
    """Extract a single page from a PDF and save it as a new file.
    
    Args:
        pdf_path (str | Path): Path to source PDF file.
        page_num (int): Zero-based page number to extract.
        output_path (str | Path): Path where to save the extracted page.
    
    Raises:
        ValueError: If page_num is out of bounds for the PDF.
    
    Note:
        Uses zero-based page numbering (first page is 0).
    """
    doc = fitz.open(pdf_path)
    if not (0 <= page_num < len(doc)):
        raise ValueError(f"Page number {page_num} out of bounds for {pdf_path.name}.")

    single_page_doc = fitz.open()
    single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    single_page_doc.save(output_path)