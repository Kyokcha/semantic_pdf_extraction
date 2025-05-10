# layouts/layout_one_column.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


def render(json_data, output_path):
    """
    Renders a PDF in a single-column layout using ReportLab.
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()
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
