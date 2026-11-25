"""Render report_content.json to a Chinese A4 PDF and emit a layout manifest.

Default: bundled-python build_idea_report.py --render
Optional: --input PATH --output PATH --manifest PATH --render-dir PATH
This file contains layout only; it does not supply or change report content.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, Image, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNTIME = Path('C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies')
DEFAULT_PDF = ROOT / 'output/pdf/图像视频推理加速_研究Idea与实验路线_2026-10-15.pdf'
POPPLER = RUNTIME / 'native/poppler/Library/bin'
DEJAVU = RUNTIME / 'native/poppler/Library/share/fonts/DejaVuSans.ttf'
W, H = A4
LEFT = RIGHT = 48
TOP = 62
BOTTOM = 48
CW = W - LEFT - RIGHT
CH = H - TOP - BOTTOM
INK = colors.HexColor('#172B42')
TEAL = colors.HexColor('#087E83')
MUTED = colors.HexColor('#586B7D')
LINE = colors.HexColor('#CEDAE3')
PALE = colors.HexColor('#EDF7F6')

pdfmetrics.registerFont(TTFont('YaHei', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiBold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei', normal='YaHei', bold='YaHeiBold', italic='YaHei', boldItalic='YaHeiBold')
pdfmetrics.registerFont(TTFont('DejaVu', str(DEJAVU)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu', italic='DejaVu', boldItalic='DejaVu')
MISSING_GLYPHS = set()

STYLES = {
    'body': ParagraphStyle('body', fontName='YaHei', fontSize=10.2, leading=16.3,
        textColor=INK, spaceAfter=8, wordWrap='CJK', allowWidows=0, allowOrphans=0),
    'small': ParagraphStyle('small', fontName='YaHei', fontSize=9.1, leading=14.2,
        textColor=MUTED, spaceAfter=7, wordWrap='CJK'),
    'h1': ParagraphStyle('h1', fontName='YaHeiBold', fontSize=20, leading=28,
        textColor=INK, spaceAfter=17, wordWrap='CJK', keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='YaHeiBold', fontSize=12.3, leading=18.5,
        textColor=TEAL, spaceBefore=8, spaceAfter=7, wordWrap='CJK', keepWithNext=True),
    'kicker': ParagraphStyle('kicker', fontName='YaHeiBold', fontSize=9.2, leading=14,
        textColor=TEAL, spaceAfter=9, wordWrap='CJK', keepWithNext=True),
    'cover': ParagraphStyle('cover', fontName='YaHeiBold', fontSize=28, leading=40,
        textColor=INK, spaceAfter=23, wordWrap='CJK', keepWithNext=True),
    'cell': ParagraphStyle('cell', fontName='YaHei', fontSize=8.7, leading=13,
        textColor=INK, wordWrap='CJK'),
    'thead': ParagraphStyle('thead', fontName='YaHeiBold', fontSize=8.7, leading=13,
        textColor=colors.white, wordWrap='CJK'),
    'callout': ParagraphStyle('callout', fontName='YaHei', fontSize=10.2, leading=16.3,
        textColor=INK, wordWrap='CJK'),
    'formula': ParagraphStyle('formula', fontName='DejaVu', fontSize=10.1, leading=17,
        textColor=INK, wordWrap='CJK'),
    'ref': ParagraphStyle('ref', fontName='YaHei', fontSize=9.1, leading=14.1,
        textColor=INK, spaceAfter=12, wordWrap='CJK'),
}


def rich(value, font='YaHei'):
    """Escape arbitrary text, then support bold, numeric citations and line breaks.

    A per-character fallback prevents missing mathematical glyphs in YaHei and
    missing CJK glyphs in DejaVu. This is plain mathematical text, not a TeX parser.
    """
    raw = str(value).replace('\r\n', '\n').replace('\r', '\n')
    raw = raw.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
    main_map = pdfmetrics.getFont(font).face.charToGlyph
    fallback = 'YaHei' if font == 'DejaVu' else 'DejaVu'
    fallback_map = pdfmetrics.getFont(fallback).face.charToGlyph
    chunks = []
    for ch in raw:
        code = ord(ch)
        if ch == '\n':
            chunks.append('<br/>')
        elif code in main_map or ch.isspace():
            chunks.append(escape(ch))
        elif code in fallback_map:
            chunks.append(f'<font name="{fallback}">{escape(ch)}</font>')
        else:
            MISSING_GLYPHS.add(f'U+{code:04X} {ch}')
            chunks.append(escape(ch))
    text = ''.join(chunks)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\[(\d{2,3})\]',
                  lambda m: f'<link href="#ref{m[1]}" color="#087E83">[{m[1]}]</link>', text)
    return text


def paragraph(value, style='body'):
    return Paragraph(rich(value, 'DejaVu' if style == 'formula' else 'YaHei'), STYLES[style])


class LogicalPageStart(Flowable):
    def __init__(self, index, title, cover=False):
        super().__init__()
        self.width = self.height = 0
        self._ZEROSIZE = True
        self.index, self.title, self.cover = index, title, cover

    def draw(self):
        self.canv.bookmarkPage(f'logical{self.index}')
        self.canv.addOutlineEntry(self.title, f'logical{self.index}', level=0)


class ReportDoc(BaseDocTemplate):
    def __init__(self, path, **kwargs):
        super().__init__(str(path), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM, **kwargs)
        frame = Frame(LEFT, BOTTOM, CW, CH, id='body', leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=self.chrome))
        self.logical_pages = []
        self.active_logical = None
        self.max_physical_page = 0

    def chrome(self, canvas, doc):
        self.max_physical_page = max(self.max_physical_page, doc.page)
        canvas.saveState()
        canvas.setFillColor(TEAL)
        canvas.rect(LEFT, H-36, 22, 2.8, fill=1, stroke=0)
        canvas.setFont('YaHei', 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT+31, H-37, '图像 / 视频推理加速  |  研究 Idea 与实验路线')
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.5)
        canvas.line(LEFT, 37, W-RIGHT, 37)
        canvas.setFont('YaHei', 8)
        canvas.drawString(LEFT, 23, '研究截至 2026-10-15  ·  模型与算法方向')
        canvas.drawRightString(W-RIGHT, 23, f'{doc.page:02d}')
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, LogicalPageStart):
            record = dict(logical_page=flowable.index, title=flowable.title,
                          start_physical_page=self.page, end_physical_page=self.page,
                          physical_pages=[self.page], overflow=False,
                          content_bottom_y=None)
            self.logical_pages.append(record)
            self.active_logical = record
        elif (self.active_logical and not isinstance(flowable, PageBreak)
              and (getattr(flowable, 'height', getattr(flowable, '_height', 0)) or 0) > 0):
            # A table or paragraph may have been split; the active logical page
            # stays active until the next explicit start marker.
            rec = self.active_logical
            rec['end_physical_page'] = self.page
            if self.page not in rec['physical_pages']:
                rec['physical_pages'].append(self.page)
            rec['overflow'] = len(rec['physical_pages']) > 1
            rec['content_bottom_y'] = round(self.frame._y, 2)


def make_table(block):
    headers, rows = block['headers'], block['rows']
    n = len(headers)
    if n == 0 or any(len(row) != n for row in rows):
        raise ValueError('Table has zero columns or inconsistent row lengths')
    requested = block.get('widths')
    widths = [CW/n]*n
    if requested:
        if len(requested) != n or min(requested) <= 0:
            raise ValueError('Table widths must be positive and match columns')
        widths = [CW*w/sum(requested) for w in requested]
    cells = [[paragraph(x, 'thead') for x in headers]]
    cells += [[paragraph(x, 'cell') for x in row] for row in rows]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT',
                  spaceAfter=11, splitByRow=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F3F7F9')]),
        ('LINEBELOW', (0, 0), (-1, 0), 1.2, TEAL),
        ('GRID', (0, 1), (-1, -1), .35, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return table


def make_panel(text, kind):
    """Use real cell padding so adjacent panel backgrounds cannot overlap."""
    panel = Table([[paragraph(text, kind)]], colWidths=[CW], hAlign='LEFT',
                  spaceBefore=4, spaceAfter=11)
    commands = [
        ('BACKGROUND', (0, 0), (-1, -1), PALE if kind == 'callout' else colors.HexColor('#F2F5F8')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
    ]
    if kind == 'callout':
        commands.append(('BOX', (0, 0), (-1, -1), .65, colors.HexColor('#BBDCD8')))
    panel.setStyle(TableStyle(commands))
    return panel


def build_story(data, input_path):
    story = []
    references = data.get('refs', {})
    ref_keys_defined = set()
    ref_keys_used = set()
    for index, page in enumerate(data['pages'], 1):
        if story:
            story.append(PageBreak())
        story.append(LogicalPageStart(index, page['title'], page.get('cover', False)))
        if page.get('cover'):
            story.append(Spacer(1, 38))
        if page.get('kicker'):
            story.append(paragraph(page['kicker'], 'kicker'))
        story.append(paragraph(page['title'], 'cover' if page.get('cover') else 'h1'))
        for block in page.get('blocks', []):
            kind = block['type']
            if kind in ('p', 'h2', 'small', 'callout', 'formula'):
                style = 'body' if kind == 'p' else kind
                story.append(make_panel(block['text'], kind) if kind in ('callout', 'formula')
                             else paragraph(block['text'], style))
                ref_keys_used.update(re.findall(r'\[(\d{2,3})\]', block['text']))
            elif kind == 'table':
                story.append(make_table(block))
                ref_keys_used.update(re.findall(r'\[(\d{2,3})\]', json.dumps(block, ensure_ascii=False)))
            elif kind == 'image':
                path = Path(block['path'])
                if not path.is_absolute():
                    path = input_path.parent / path
                img = Image(str(path))
                requested_w = block.get('width')
                requested_h = block.get('height')
                scale = min(CW / img.imageWidth, (CH-65) / img.imageHeight)
                if requested_w:
                    scale = min(scale, float(requested_w)/img.imageWidth)
                if requested_h:
                    scale = min(scale, float(requested_h)/img.imageHeight)
                img.drawWidth, img.drawHeight = img.imageWidth*scale, img.imageHeight*scale
                img.hAlign = 'LEFT'
                story.extend([img, Spacer(1, 10)])
            elif kind == 'refs':
                for key in map(str, block['keys']):
                    if key in ref_keys_defined:
                        raise ValueError(f'Duplicate reference anchor: {key}')
                    ref_keys_defined.add(key)
                    ref = references[key]
                    text = f'<a name="ref{key}"/>[{key}] <b>{rich(ref["title"])}</b>'
                    if ref.get('meta'):
                        text += f'<br/>{rich(ref["meta"])}'
                    text += f'<br/><link href={quoteattr(ref["url"])} color="#087E83">论文 / 来源链接</link>'
                    if ref.get('code'):
                        text += f'  ·  <link href={quoteattr(ref["code"])} color="#087E83">官方代码 / 项目</link>'
                    story.append(Paragraph(text, STYLES['ref']))
            else:
                raise ValueError(f'Unsupported block type {kind!r} on logical page {index}')
    missing = ref_keys_used - ref_keys_defined
    if missing:
        raise ValueError(f'Citations without displayed reference anchors: {sorted(missing)}')
    return story


def text_bbox_qa(pdf_path):
    """Report page-bound violations and replacement glyphs; does not replace visual QA."""
    import pdfplumber
    result = {'engine': 'pdfplumber', 'pages': [], 'page_bound_violations': [],
              'content_frame_violations': [],
              'replacement_characters': [], 'note': 'Text bounds are checked; rendered-page inspection is still required.'}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            chars = page.chars
            outside = [c for c in chars if c['x0'] < -.5 or c['x1'] > page.width+.5
                       or c['top'] < -.5 or c['bottom'] > page.height+.5]
            body = [c for c in chars if c['top'] > 45 and c['bottom'] < page.height-40]
            result['pages'].append({'physical_page': i, 'characters': len(chars),
                'body_text_bbox': [round(min(c['x0'] for c in body), 2),
                                   round(min(c['top'] for c in body), 2),
                                   round(max(c['x1'] for c in body), 2),
                                   round(max(c['bottom'] for c in body), 2)] if body else None})
            if outside:
                result['page_bound_violations'].append({'page': i, 'count': len(outside)})
            margin_chars = [c for c in body if c['x0'] < LEFT-1 or c['x1'] > W-RIGHT+1
                            or c['top'] < TOP-4 or c['bottom'] > H-BOTTOM+3]
            if margin_chars:
                result['content_frame_violations'].append({'page': i, 'count': len(margin_chars),
                    'sample': ''.join(c['text'] for c in margin_chars[:40])})
            if '\ufffd' in (page.extract_text() or ''):
                result['replacement_characters'].append(i)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE/'report_content.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_PDF)
    parser.add_argument('--manifest', type=Path, default=HERE/'layout_manifest.json')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-dir', type=Path, default=HERE/'rendered')
    parser.add_argument('--dpi', type=int, default=110)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    story = build_story(data, args.input)
    doc = ReportDoc(args.output, title='图像视频推理加速：研究 Idea 与实验路线',
                    author='研究报告', pageCompression=1)
    doc.build(story)
    manifest = {'pdf': str(args.output.resolve()), 'input': str(args.input.resolve()),
        'date': '2026-10-15', 'physical_page_count': doc.max_physical_page,
        'logical_page_count': len(data['pages']), 'page_size_pt': [W, H],
        'content_frame_pt': {'left': LEFT, 'bottom': BOTTOM, 'width': CW, 'height': CH},
        'logical_pages': doc.logical_pages,
        'overflow_logical_pages': [p['logical_page'] for p in doc.logical_pages if p['overflow']],
        'missing_font_glyphs': sorted(MISSING_GLYPHS),
        'text_qa': text_bbox_qa(args.output),
        'rendered_pages': [], 'visual_qa_status': 'pending'}
    if args.render:
        args.render_dir.mkdir(parents=True, exist_ok=True)
        render_root = args.render_dir.resolve()
        for old_png in args.render_dir.glob('page-*.png'):
            if old_png.is_file() and old_png.resolve().parent == render_root:
                old_png.unlink()
        subprocess.run([str(POPPLER/'pdftoppm.exe'), '-r', str(args.dpi), '-png',
                        str(args.output), str(args.render_dir/'page')], check=True)
        manifest['rendered_pages'] = [str(p.resolve()) for p in sorted(args.render_dir.glob('page-*.png'))]
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pdf': str(args.output), 'manifest': str(args.manifest),
                     'physical_pages': doc.max_physical_page,
                     'overflow_logical_pages': manifest['overflow_logical_pages'],
                     'missing_font_glyphs': manifest['missing_font_glyphs']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
