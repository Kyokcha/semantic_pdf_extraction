# layouts/side_bar.py

from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from utils.flowables import build_flowables


def render(json_data, output_path):
    width, height = A4
    flowables = build_flowables(json_data)

    if not flowables:
        flowables = build_flowables({"title": "No content", "sections": []})

    # Sidebar = just title or first paragraph
    sidebar_flowable = [flowables[0]]

    # Define frames
    sidebar_frame = Frame(inch, inch, width * 0.25, height - 2 * inch, id='sidebar')
    main_frame = Frame(inch + width * 0.25 + 0.3 * inch, inch,
                       width - (2 * inch + width * 0.25 + 0.3 * inch),
                       height - 2 * inch, id='main')

    def draw_sidebar(canvas, doc):
        sidebar_frame.addFromList(sidebar_flowable, canvas)

    doc = BaseDocTemplate(str(output_path), pagesize=A4)
    doc.addPageTemplates([
        PageTemplate(id='SideBar', frames=[main_frame], onPage=draw_sidebar)
    ])
    doc.build(flowables[1:])
