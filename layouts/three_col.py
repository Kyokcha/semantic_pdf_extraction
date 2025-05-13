# layouts/three_col.py

from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


def render(json_data, output_path):
    width, height = A4
    flowables = build_flowables(json_data)

    col_width = (width - 2 * inch - 2 * 6) / 3
    x1 = inch
    x2 = x1 + col_width + 6
    x3 = x2 + col_width + 6

    frame1 = Frame(x1, inch, col_width, height - 2 * inch, id='col1')
    frame2 = Frame(x2, inch, col_width, height - 2 * inch, id='col2')
    frame3 = Frame(x3, inch, col_width, height - 2 * inch, id='col3')

    doc = BaseDocTemplate(str(output_path), pagesize=A4)
    doc.addPageTemplates([
        PageTemplate(id='ThreeCol', frames=[frame1, frame2, frame3])
    ])
    doc.build(flowables)
