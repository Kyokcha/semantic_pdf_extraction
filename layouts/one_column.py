# layouts/layout_one_column.py

from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


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

    flowables = build_flowables(json_data)
    if not flowables:
        flowables = build_flowables({"title": "No content", "sections": []})

    doc.build(flowables)
