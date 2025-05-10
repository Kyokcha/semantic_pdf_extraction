from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

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
    styles = getSampleStyleSheet()
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=1.25 * inch,  # Extra space for header
        bottomMargin=1.0 * inch  # Extra space for footer
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id='normal')

    template = PageTemplate(id='HeaderFooter', frames=frame, onPage=add_header_footer)
    doc.addPageTemplates([template])

    flowables = []

    # Title
    flowables.append(Paragraph(json_data["title"], styles["Title"]))
    flowables.append(Spacer(1, 0.2 * inch))

    # Sections
    for section in json_data.get("sections", []):
        heading = section.get("heading", "")
        paragraphs = section.get("paragraphs", [])

        if heading:
            flowables.append(Paragraph(heading, styles["Heading2"]))
            flowables.append(Spacer(1, 0.1 * inch))

        for para in paragraphs:
            flowables.append(Paragraph(para, styles["BodyText"]))
            flowables.append(Spacer(1, 0.1 * inch))

        flowables.append(Spacer(1, 0.3 * inch))  # Space between sections

    doc.build(flowables)
