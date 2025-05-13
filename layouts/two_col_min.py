# layouts/two_col_min.py

from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


def render(json_data, output_path):
    width, height = A4
    flowables = build_flowables(json_data)

    col_width = (width - 2 * inch) / 2 - 3
    left_frame = Frame(inch, inch, col_width, height - 2 * inch, id='left')
    right_frame = Frame(inch + col_width + 6, inch, col_width, height - 2 * inch, id='right')

    doc = BaseDocTemplate(str(output_path), pagesize=A4)
    doc.addPageTemplates([
        PageTemplate(id='TwoColMin', frames=[left_frame, right_frame])
    ])
    doc.build(flowables)
