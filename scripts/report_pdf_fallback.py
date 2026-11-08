"""Render the generated DOCX to PDF with ReportLab when Office export fails.

Preserves text, tables, images, page breaks and Office Math text. The editable
equation objects remain in the DOCX; PDF is a printable companion.
"""
from html import escape
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, PageBreak,
                               Spacer, Table, TableStyle, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
for name, filename in [('TimesVN','times.ttf'), ('TimesVN-Bold','timesbd.ttf'),
                       ('TimesVN-Italic','timesi.ttf'), ('TimesVN-BoldItalic','timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+filename))
pdfmetrics.registerFontFamily('TimesVN', normal='TimesVN', bold='TimesVN-Bold',
                             italic='TimesVN-Italic', boldItalic='TimesVN-BoldItalic')
pdfmetrics.registerFont(TTFont('MathSymbols', 'C:/Windows/Fonts/seguisym.ttf'))
TIMES_GLYPHS = pdfmetrics.getFont('TimesVN').face.charWidths
SYMBOL_GLYPHS = pdfmetrics.getFont('MathSymbols').face.charWidths


def encoded(text):
    """Use a Unicode symbol fallback for mathematical glyphs absent in Times."""
    parts = []
    for char in text:
        value = escape(char)
        if ord(char) not in TIMES_GLYPHS and ord(char) in SYMBOL_GLYPHS:
            value = '<font name="MathSymbols">' + value + '</font>'
        parts.append(value)
    return ''.join(parts)
body = ParagraphStyle('Body', fontName='TimesVN', fontSize=13, leading=16.9,
                      alignment=TA_JUSTIFY, spaceAfter=6, allowWidows=0, allowOrphans=0)
heading1 = ParagraphStyle('H1', parent=body, fontName='TimesVN-Bold', fontSize=16,
                          leading=20, spaceAfter=12, keepWithNext=True, alignment=TA_LEFT)
heading2 = ParagraphStyle('H2', parent=heading1, fontSize=14, leading=18)
title_style = ParagraphStyle('Title', parent=heading1, fontSize=22, leading=28,
                            alignment=TA_CENTER, spaceAfter=20)
cell_style = ParagraphStyle('Cell', parent=body, fontSize=12, leading=13, spaceAfter=0,
                            alignment=TA_LEFT)
caption_style = ParagraphStyle('Caption', parent=body, fontName='TimesVN-Italic',
                               alignment=TA_CENTER, fontSize=12, leading=15)
math_style = ParagraphStyle('Math', parent=body, alignment=TA_CENTER, fontSize=12, leading=17)


def render():
    """Walk DOCX body order into printable text, tables and image objects."""
    document = Document(ROOT / 'docs/BAO_CAO_BTL_LeafLab.docx')
    tmp = ROOT / 'docs/tmp/pdf_assets'
    tmp.mkdir(parents=True, exist_ok=True)
    story = []
    for element in document.element.body:
        if element.tag == qn('w:p'):
            para = WordParagraph(element, document)
            if any(br.get(qn('w:type')) == 'page' for br in element.findall('.//' + qn('w:br'))):
                story.append(PageBreak())
                continue
            drawings = element.findall('.//' + qn('a:blip'))
            if drawings:
                for idx, drawing in enumerate(drawings):
                    rid = drawing.get(qn('r:embed'))
                    part = document.part.related_parts[rid]
                    path = tmp / (rid + '.png')
                    path.write_bytes(part.blob)
                    extent = element.find('.//' + qn('wp:extent'))
                    width = int(extent.get('cx')) / 914400 * 72
                    height = int(extent.get('cy')) / 914400 * 72
                    story.append(Image(str(path), width=width, height=height))
                continue
            maths = element.findall('.//' + qn('m:t'))
            if maths:
                text = ''.join(t.text or '' for t in maths)
                story.append(Paragraph(encoded(text), math_style))
                continue
            if not para.text:
                continue
            style = title_style if para.style.name == 'Title' else heading1 if para.style.name == 'Heading 1' else (
                heading2 if para.style.name.startswith('Heading') else body)
            if para.text.startswith('Hình ') and para.runs and para.runs[0].italic:
                style = caption_style
            text = ''
            for run in para.runs:
                chunk = encoded(run.text).replace('\n', '<br/>')
                if run.bold:
                    chunk = '<b>'+chunk+'</b>'
                if run.italic:
                    chunk = '<i>'+chunk+'</i>'
                text += chunk
            story.append(Paragraph(text or encoded(para.text), style))
        elif element.tag == qn('w:tbl'):
            table = WordTable(element, document)
            data = []
            for row in table.rows:
                values = []
                for cell in row.cells:
                    text = encoded(cell.text)
                    if cell.paragraphs and any(r.bold for r in cell.paragraphs[0].runs):
                        text = '<b>'+text+'</b>'
                    values.append(Paragraph(text, cell_style))
                data.append(values)
            widths = [col.width/914400*72 for col in table.columns]
            t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            commands = [('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),
                        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF1')),
                        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                        ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5),
                        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5)]
            for i in range(2,len(data),2):
                commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#F7F8F9')))
            t.setStyle(TableStyle(commands))
            story.extend([t,Spacer(1,6)])
    out = ROOT / 'docs/BAO_CAO_BTL_LeafLab.pdf'
    def footer(canvas, doc):
        canvas.setFont('TimesVN',11)
        canvas.drawCentredString(11*cm,1.35*cm,str(doc.page))
    pdf = SimpleDocTemplate(str(out), pagesize=(21*cm,29.7*cm), leftMargin=3*cm,
                            rightMargin=2*cm,topMargin=2.5*cm,bottomMargin=2.5*cm,
                            title='Phân tích bệnh lá lúa và cà phê từ ảnh điện thoại',
                            author='Nhóm BTL Xử lý ảnh')
    pdf.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__ == '__main__':
    render()
