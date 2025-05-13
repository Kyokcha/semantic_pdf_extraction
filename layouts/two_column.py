# layouts/layout_two_column.py

from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


def render(json_data, output_path):
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch
    )

    width, height = A4
    column_gap = 0.2 * inch
    column_width = (width - doc.leftMargin - doc.rightMargin - column_gap) / 2
    column_height = height - doc.topMargin - doc.bottomMargin

    # Two columns
    frame1 = Frame(doc.leftMargin, doc.bottomMargin, column_width, column_height, id='col1')
    frame2 = Frame(doc.leftMargin + column_width + column_gap, doc.bottomMargin, column_width, column_height, id='col2')

    doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2])])

    flowables = build_flowables(json_data)
    if not flowables:
        flowables = build_flowables({"title": "No content", "sections": []})

    doc.build(flowables)
