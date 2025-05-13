# utils/flowables.py

from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def build_flowables(json_data):
    styles = getSampleStyleSheet()
    flowables = []

    # Title
    title = json_data.get("title", "")
    if title:
        flowables.append(Paragraph(title, styles["Title"]))
        flowables.append(Spacer(1, 0.2 * inch))

    # Sections
    for section in json_data.get("sections", []):
        heading = section.get("heading", "")
        paragraphs = section.get("paragraphs", [])

        if heading:
            flowables.append(Paragraph(heading, styles["Heading2"]))
            flowables.append(Spacer(1, 0.1 * inch))

        for para in paragraphs:
            if para.strip():
                flowables.append(Paragraph(para, styles["BodyText"]))
                flowables.append(Spacer(1, 0.1 * inch))

        flowables.append(Spacer(1, 0.3 * inch))  # Between sections

    return flowables
