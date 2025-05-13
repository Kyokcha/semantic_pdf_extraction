# layouts/mixed_format.py

from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


def render(json_data, output_path):
    width, height = A4
    flowables = build_flowables(json_data)

    if not flowables:
        flowables = build_flowables({"title": "No content", "sections": []})

    # Split into intro vs. main body
    intro_count = 5
    intro_flowables = flowables[:intro_count]
    body_flowables = flowables[intro_count:]

    # ---- Page 1: Full-width intro + columns ----
    top_frame = Frame(inch, height - 2.5 * inch, width - 2 * inch, 1.5 * inch, id='top')
    col_width = (width - 2 * inch - 0.2 * inch) / 2
    left_frame = Frame(inch, inch, col_width, height - 4.1 * inch, id='left')
    right_frame = Frame(inch + col_width + 0.2 * inch, inch, col_width, height - 4.1 * inch, id='right')

    # ---- Page 2+: Two columns only ----
    col_frame1 = Frame(inch, inch, col_width, height - 2 * inch, id='col1')
    col_frame2 = Frame(inch + col_width + 0.2 * inch, inch, col_width, height - 2 * inch, id='col2')

    # Build doc with two templates
    doc = BaseDocTemplate(str(output_path), pagesize=A4)
    doc.addPageTemplates([
        PageTemplate(id='MixedFirstPage', frames=[top_frame, left_frame, right_frame]),
        PageTemplate(id='ColumnsOnly', frames=[col_frame1, col_frame2])
    ])

    # Build: intro content triggers first page, then body flows into 2-col template
    doc.build(intro_flowables + body_flowables)
