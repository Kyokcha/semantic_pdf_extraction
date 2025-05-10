# layouts/layout_two_column.py

from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

def render(json_data, output_path):
    styles = getSampleStyleSheet()
    doc = BaseDocTemplate(str(output_path), pagesize=A4,
                          leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                          topMargin=1 * inch, bottomMargin=1 * inch)

    width, height = A4
    column_gap = 0.2 * inch
    column_width = (width - doc.leftMargin - doc.rightMargin - column_gap) / 2
    column_height = height - doc.topMargin - doc.bottomMargin

    # Define two side-by-side frames
    frame1 = Frame(doc.leftMargin, doc.bottomMargin, column_width, column_height, id='col1')
    frame2 = Frame(doc.leftMargin + column_width + column_gap, doc.bottomMargin, column_width, column_height, id='col2')

    doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2])])

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
