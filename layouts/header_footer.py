# layouts/header_footer.py

from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from utils.flowables import build_flowables


def add_header_footer(canvas_obj, doc):
    """
    Draws header and footer on every page.
    """
    canvas_obj.saveState()

    # Header
    header_text = "Semantic PDF Extraction – Generated Sample"
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawString(inch, A4[1] - 0.75 * inch, header_text)

    # Footer: Page number
    footer_text = f"Page {doc.page}"
    canvas_obj.drawRightString(A4[0] - inch, 0.75 * inch, footer_text)

    canvas_obj.restoreState()

def render(json_data, output_path):
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=1.25 * inch,  # Extra space for header
        bottomMargin=1.0 * inch  # Extra space for footer
    )

    # Single full-page frame
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id='normal'
    )

    # Page template with header/footer
    template = PageTemplate(id='HeaderFooter', frames=frame, onPage=add_header_footer)
    doc.addPageTemplates([template])

    # Flowable content with title + sections
    flowables = build_flowables(json_data)
    if not flowables:
        flowables = build_flowables({"title": "No content", "sections": []})

    doc.build(flowables)
