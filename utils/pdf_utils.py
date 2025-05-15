# utils/pdf_utils.py

import fitz  # PyMuPDF


def extract_single_pdf_page(pdf_path, page_num, output_path):
    """
    Extract a single page from a PDF using PyMuPDF.
    Page number is 0-indexed.
    """
    doc = fitz.open(pdf_path)
    if not (0 <= page_num < len(doc)):
        raise ValueError(f"Page number {page_num} out of bounds for {pdf_path.name}.")

    single_page_doc = fitz.open()
    single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    single_page_doc.save(output_path)