"""Build the project report with the retained academic document's format."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(path):
    """Parse the existing editable technical content without changing its metrics."""
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        return {k: tuple(v) for k,v in data['sections'].items()}, data['abstract']
    text = path.read_text(encoding='utf-8')
    # The editable source separates emitted table rows with blank lines.
    text = re.sub(r'(?m)(^\|[^\n]*\|)[ \t]*\n(?:[ \t]*\n)+(?=\|)', r'\1\n', text)
    sections = {}
    preamble, *_ = text.split('\n## ', 1)
    for chunk in re.split(r'\n## ', text)[1:]:
        title, _, content = chunk.partition('\n')
        key = title.split()[0]
        sections[key] = (title, content.strip())
    abstract = next(p for p in preamble.split('\n\n') if p.startswith('Fixed traffic-signal'))
    return sections, abstract


class Builder:
    def __init__(self, template, source):
        self.template = template
        self.doc = Document(template)
        self.parts = {n: b for n, b in self.zip_parts(template).items()}
        self.sections = [deepcopy(s._sectPr) for s in self.doc.sections]
        self.section_index = 0
        self.source, self.abstract = read_source(source)
        self.markdown = []
        self.heading_count = 0
        self.image_count = 0
        self.table_count = 0
        # Preserve source styles and source page patterns; replace only content slots.
        body = self.doc._element.body
        for element in list(body):
            body.remove(element)
        body.append(deepcopy(self.sections[0]))
        self.doc.core_properties.title = 'Adaptive Crossroads Control with Emergency Vehicle Priority'
        self.doc.core_properties.subject = 'Reinforcement Learning - UCS760'
        self.doc.core_properties.author = 'Dyaus Ghangas'
        self.doc.core_properties.last_modified_by = 'Dyaus Ghangas'
        self.doc.core_properties.created = datetime(2026, 11, 18, tzinfo=timezone.utc)
        self.doc.core_properties.modified = datetime(2026, 11, 18, tzinfo=timezone.utc)

    @staticmethod
    def zip_parts(path):
        with ZipFile(path) as archive:
            return {n: archive.read(n) for n in archive.namelist()}

    def paragraph(self, text='', *, align=None, after=None, before=None, bold=False,
                  italic=False, size=None, color=None, keep=False):
        para = self.doc.add_paragraph(style='Normal')
        if align is not None:
            para.alignment = align
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        if before is not None:
            para.paragraph_format.space_before = Pt(before)
        para.paragraph_format.keep_with_next = keep
        for i, segment in enumerate(re.split(r'(\*\*.*?\*\*|<b>.*?</b>)', text)):
            if not segment:
                continue
            strong = (segment.startswith('**') and segment.endswith('**')) or segment.startswith('<b>')
            segment = segment[2:-2] if segment.startswith('**') else segment.replace('<b>', '').replace('</b>', '')
            segment = segment.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
            run = para.add_run(segment)
            run.bold = bold or strong
            run.italic = italic
            if size:
                run.font.size = Pt(size)
            if color:
                run.font.color.rgb = RGBColor.from_string(color)
        self.markdown.append(text)
        return para

    def heading(self, text, level=1, page=False):
        para = self.doc.add_paragraph(text, style=f'Heading {level}')
        para.paragraph_format.page_break_before = bool(page)
        # Keep the source's underlined Times New Roman heading system.
        self.markdown.append('#' * (level+1) + ' ' + text)
        self.heading_count += 1
        return para

    def next_section(self):
        self.section_index += 1
        body = self.doc._element.body
        current = body.find(qn('w:sectPr'))
        body.remove(current)
        para = OxmlElement('w:p')
        props = OxmlElement('w:pPr')
        props.append(current)
        para.append(props)
        body.append(para)
        body.append(deepcopy(self.sections[self.section_index]))

    def numbered(self, items):
        root = self.doc.part.numbering_part.element
        existing = [int(n.get(qn('w:numId'))) for n in root.findall(qn('w:num'))]
        source = root.find(f"{{{NS['w']}}}num[@{{{NS['w']}}}numId='1']")
        abstract = source.find(qn('w:abstractNumId')).get(qn('w:val'))
        num = OxmlElement('w:num')
        num_id = max(existing) + 1
        num.set(qn('w:numId'), str(num_id))
        ref = OxmlElement('w:abstractNumId'); ref.set(qn('w:val'), abstract); num.append(ref)
        override = OxmlElement('w:lvlOverride'); override.set(qn('w:ilvl'), '0')
        start = OxmlElement('w:startOverride'); start.set(qn('w:val'), '1'); override.append(start); num.append(override)
        root.append(num)
        for text in items:
            para = self.paragraph(text)
            pr = para._p.get_or_add_pPr()
            numbering = OxmlElement('w:numPr')
            level = OxmlElement('w:ilvl'); level.set(qn('w:val'), '0')
            identifier = OxmlElement('w:numId'); identifier.set(qn('w:val'), str(num_id))
            numbering.extend([level, identifier]); pr.append(numbering)
            para.paragraph_format.left_indent = Pt(25)
            para.paragraph_format.first_line_indent = Pt(-18)

    def table(self, rows):
        cols = len(rows[0])
        table = self.doc.add_table(rows=1, cols=cols)
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        # Reuse the retained academic format's plain table grid.
        template_xml = E.fromstring(self.parts['word/document.xml'])
        template_table = template_xml.find('.//w:tbl', NS)
        source_props = template_table.find('w:tblPr', NS)
        if source_props is not None:
            table._tbl.remove(table._tbl.tblPr)
            table._tbl.insert(0, deepcopy(source_props))
        width = self.doc.sections[-1].page_width - self.doc.sections[-1].left_margin - self.doc.sections[-1].right_margin
        props = table._tbl.tblPr
        preferred = props.find(qn('w:tblW'))
        if preferred is None: preferred=OxmlElement('w:tblW');props.append(preferred)
        preferred.set(qn('w:type'),'dxa');preferred.set(qn('w:w'),str(round(width/635)))
        indentation = props.find(qn('w:tblInd'))
        if indentation is not None: props.remove(indentation)
        old_borders = props.find(qn('w:tblBorders'))
        if old_borders is not None: props.remove(old_borders)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            border=OxmlElement('w:'+edge)
            for attr,value in [('val','single'),('sz','4'),('color','A6A6A6')]: border.set(qn('w:'+attr),value)
            borders.append(border)
        props.append(borders)
        table.alignment=WD_TABLE_ALIGNMENT.LEFT
        fractions = ([.31,.69] if cols == 2 else [.32,.23,.15,.30] if cols == 4 else [.27]+[.73/(cols-1)]*(cols-1))
        table.autofit = False
        for c, f in zip(table.columns, fractions):
            c.width = int(width * f)
        for index, values in enumerate(rows):
            row = table.rows[0] if index == 0 else table.add_row()
            row.height = None
            for j, value in enumerate(values):
                cell = row.cells[j]
                cell.width = int(width * fractions[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'F2F2F2' if index % 2 else 'FFFFFF')
                cell._tc.get_or_add_tcPr().append(shading)
                para = cell.paragraphs[0]
                para.style = self.doc.styles['Normal']
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.space_before = Pt(3)
                para.paragraph_format.keep_with_next = index == 0
                run = para.add_run(value)
                run.bold = index == 0
                run.font.size = Pt(11 if cols <= 4 else 10.5)
                run.font.color.rgb = RGBColor(0,0,0)
            if index == 0:
                marker = OxmlElement('w:tblHeader'); row._tr.get_or_add_trPr().append(marker)
            # Keep individual rows intact while allowing continuation with repeated headers.
            no_split = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(no_split)
        self.table_count += 1
        self.markdown.append('| ' + ' | '.join(rows[0]) + ' |')
        self.markdown.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
        self.markdown.extend('| ' + ' | '.join(row) + ' |' for row in rows[1:])

    def figure(self, target, caption):
        path = (OUT / target).resolve()
        para = self.doc.add_paragraph(style='Normal')
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        usable = self.doc.sections[-1].page_width - self.doc.sections[-1].left_margin - self.doc.sections[-1].right_margin
        para.add_run().add_picture(str(path), width=min(usable, Inches(6.4)))
        self.paragraph(caption, italic=True, size=11, after=12)
        self.image_count += 1
        self.markdown.append(f'![{caption}]({target})')

    def blocks(self, text, *, subprefix=None, skip_figures=False):
        paragraphs = re.split(r'\n\s*\n', text.strip())
        # Place table labels above their tables, matching the source's table-introduction pattern.
        for index in range(len(paragraphs)-1):
            if paragraphs[index].lstrip().startswith('|') and paragraphs[index+1].startswith('Table '):
                paragraphs[index], paragraphs[index+1] = paragraphs[index+1], paragraphs[index]
        section_number = 0
        for block in paragraphs:
            block = block.strip()
            if not block:
                continue
            if block.startswith('### '):
                section_number += 1
                self.heading((f'{subprefix}.{section_number} ' if subprefix else '') + block[4:], 2)
            elif block.startswith('|'):
                lines = [ln for ln in block.splitlines() if ln.strip().startswith('|')]
                rows = [[v.strip() for v in ln.strip().strip('|').split('|')] for ln in lines]
                rows = [row for row in rows if not all(re.fullmatch(r'[:\- ]+', v) for v in row)]
                self.table(rows)
            elif block.startswith('!['):
                if skip_figures:
                    continue
                match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', block, re.S)
                self.figure(match.group(2), match.group(1))
            elif block.startswith('```'):
                lines = block.splitlines()[1:-1]
                # Editable native text for the source's simple reward/pseudocode notation.
                para = self.doc.add_paragraph(style='Normal')
                para.paragraph_format.space_after = Pt(10)
                run = para.add_run('\n'.join(lines))
                run.font.name = 'Consolas'; run.font.size = Pt(9)
                self.markdown.append('```text\n'+'\n'.join(lines)+'\n```')
            elif block.startswith(('Figure ', 'Table ')):
                self.paragraph(block, italic=True, size=11, keep=block.startswith('Table '))
            else:
                self.paragraph(block)

    def source_body(self, number, **kwargs):
        self.blocks(self.source[str(number)][1], **kwargs)

    def toc(self):
        original = E.fromstring(self.parts['word/document.xml'])
        control = deepcopy(original.find('.//w:body/w:sdt', NS))
        content = control.find('w:sdtContent', NS)
        title = deepcopy(content[0])
        for child in list(content): content.remove(child)
        content.append(title)
        para = OxmlElement('w:p')
        for kind in ['begin', 'instruction', 'separate', 'cached', 'end']:
            run = OxmlElement('w:r')
            if kind == 'instruction':
                element = OxmlElement('w:instrText'); element.set(qn('xml:space'), 'preserve')
                element.text = ' TOC \\o "1-2" \\h \\z \\u '
            elif kind == 'cached':
                element = OxmlElement('w:t'); element.text = ' '
            else:
                element = OxmlElement('w:fldChar'); element.set(qn('w:fldCharType'), kind)
            run.append(element); para.append(run)
        content.append(para)
        body = self.doc._element.body
        body.insert(len(body)-1, control)

    def build(self, output):
        centre = WD_ALIGN_PARAGRAPH.CENTER
        for text in ['A Project Report submitted for', 'Reinforcement Learning - UCS760', 'On']:
            self.paragraph(text, align=centre, after=18)
        title = self.paragraph('Adaptive Crossroads Control with\nEmergency Vehicle Priority', align=centre, after=18)
        title.style = self.doc.styles['Title']
        for run in title.runs:
            run.font.name = 'Times New Roman';run.font.size=Pt(12);run.bold=False;run.underline=False;run.font.color.rgb=RGBColor(0,0,0)
        self.paragraph('By', align=centre, after=18)
        self.paragraph('Dyaus Ghangas          102308054', align=centre, after=45, color='0000FF')
        self.paragraph('Submitted to:', align=centre, bold=True, after=18)
        self.paragraph('Dr. Himika Sharma', align=centre, bold=True, after=80)
        logo = self.doc.add_paragraph(style='Normal');logo.alignment=centre
        with ZipFile(self.template) as archive:
            from io import BytesIO
            logo.add_run().add_picture(BytesIO(archive.read('word/media/image1.jpg')), width=Inches(2.6))
        self.doc.add_page_break()
        self.toc()
        self.heading('Certificate', page=True)
        self.paragraph('This is to certify that the project report entitled “Adaptive Crossroads Control with Emergency Vehicle Priority”, submitted by Dyaus Ghangas (Roll Number 102308054) to Dr. Himika Sharma, Thapar Institute of Engineering and Technology, Patiala, in fulfilment of the course requirement of Reinforcement Learning (UCS760), presents the design, implementation and experimental evaluation of the project.', after=8)
        self.paragraph('The report records the problem formulation, synthetic dataset, trained controllers, simulation application and measured outcomes. The results are supported by the configurations, checkpoints, evaluation records and verification evidence accompanying the project.', after=60)
        self.paragraph('Dr. Himika Sharma', bold=True, after=0)
        self.paragraph('Course Instructor, UCS760', after=75)
        self.paragraph('Dyaus Ghangas', align=WD_ALIGN_PARAGRAPH.RIGHT, after=0)
        self.paragraph('102308054', align=WD_ALIGN_PARAGRAPH.RIGHT)
        self.heading('Acknowledgement', page=True)
        self.paragraph('I would like to express my sincere gratitude to Dr. Himika Sharma, course instructor for Reinforcement Learning (UCS760), for the opportunity to undertake this project and apply the concepts studied in the course. Developing Adaptive Crossroads Control with Emergency Vehicle Priority provided a practical setting for understanding state design, reward formulation, deep Q-learning and experimental evaluation. I am also grateful to Thapar Institute of Engineering and Technology, Patiala, for the academic environment in which this work was carried out. The project strengthened my understanding of reproducibility, fair controller comparison and the importance of reporting limitations alongside measured results. I thank everyone who supported the completion of the project.', after=0)

        self.heading('Introduction', page=True)
        self.heading('1.1 Project Overview',2); self.paragraph(self.abstract)
        self.heading('1.2 Problem Identification',2)
        self.blocks('\n\n'.join(self.source['1'][1].split('\n\n')[:2]))
        self.heading('1.3 Solution Proposed',2)
        self.numbered([
            '**Adaptive traffic service:** Formulate signal selection as a reinforcement learning problem using observed lane demand, queue age and phase duration.',
            '**Emergency awareness:** Include ambulance presence and proximity in the state and penalise red/yellow approach impedance, stopping and waiting.',
            '**Fair comparison:** Evaluate fixed-time, rule-based, standard DQN, emergency-aware DQN and hybrid control on identical held-out traffic inputs.',
            '**Reproducible simulation:** Include route manifests, trained checkpoints, configurations, raw evaluation records and verification commands.',
            '**Interactive demonstration:** Display actual SUMO positions using single or synchronized side-by-side playback, with genuine ambulance injections and clean lifecycle controls.'
        ])
        self.heading('1.4 Objectives and Scope',2)
        self.paragraph(self.source['1'][1].split('\n\n')[-1])
        self.paragraph('The scope is a single four-way junction with synthetic traffic, a fixed evaluation horizon and a small set of trained seeds. The system is an academic simulation rather than a road-infrastructure deployment. No database server, private traffic feed or cloud inference service is required.')
        self.heading('1.5 Reinforcement Learning Background',2);self.source_body(2)
        self.heading('1.6 Intended Use and Expected Benefits',2)
        self.numbered([
            '**Students and evaluators:** Inspect how traffic demand, controller choice and an approaching ambulance affect phase selection and delay.',
            '**Experimentation:** Reproduce trained-policy results and compare emergency service against ordinary-traffic costs under matching inputs.',
            '**Transparent demonstration:** View exact simulator states, pause or replay history and distinguish an injection request from physical vehicle insertion.'
        ])

        self.next_section()
        self.heading('System Design and Dataset')
        self.heading('2.1 System Architecture',2);self.source_body(3)
        self.heading('2.2 Dataset and Traffic Generation',2)
        self.paragraph('The primary dataset contains 96 synthetic route files arranged as 48 normal/emergency pairs. Each background seed generates identical regular-vehicle schedules and simulator seeds for the paired cases. Ambulance timing uses an independent random stream, so adding an emergency vehicle does not change the regular-arrival draws. These are simulation inputs, not measured city traffic.')
        self.paragraph('Table 1. Primary dataset partition. Each pair contains the same regular traffic with and without an ambulance.',italic=True,size=11,keep=True)
        self.table([['Partition','Background seeds','Pairs','Routes'],['Training','10000-10023','24','48'],['Validation','20000-20011','12','24'],['Held-out test','30000-30011','12','24'],['Total','Disjoint partitions','48','96']])
        self.paragraph('Low, medium and rush demand use Poisson rates of 0.055, 0.12 and 0.21 vehicles per second per approach. Patterns are symmetric, north/south-heavy or east/west-heavy. The heavy axis receives a 1.45 multiplier and the other axis 0.55, preserving total expected demand. Vehicles choose straight travel with probability 0.8 and each other destination with probability 0.1. Regular arrivals stop at 900s; episodes end at 1200s.')
        self.paragraph('Requested ambulance departure is between 240s and 420s, with early and late timing windows. Each 12-pair block balances direction, demand and imbalance. It is not a full factorial: all four directions, three levels and three imbalances would require 36 combinations before replication. The compact design cannot identify every interaction.')
        self.paragraph('Learner seeds 0, 1 and 2 are separate from traffic seeds, and train/validation/test inputs are disjoint. Final checkpoints use a fixed 40000-decision budget. Each validation and test split contains 24 routes times (2 deterministic policies + 3 policies times 3 model seeds), giving 264 episodes. The emergency-present results use 132 of these records.')
        self.paragraph('Manifests record configuration, vehicle count, split, seed, input path and SHA-256. DATASET.md also identifies supplementary integrity/regression fixtures. Interactive scenarios are stored separately and excluded from held-out statistics.')

        self.next_section()
        self.heading('Learning Methodology')
        self.blocks(self.source['5'][1],subprefix='3')
        self.heading('3.3 Controllers and Training Configuration',2);self.source_body(6)
        self.heading('3.4 Application Implementation',2);self.source_body(9,skip_figures=True)

        self.next_section()
        self.heading('3.5 Simulation Workflow',2)
        self.paragraph('The application presents the same generated traffic in both views. Each controller runs in its own SUMO process and advances through a common time barrier. Vehicle colours, signal indicators and telemetry describe the actual displayed state.')
        self.figure('../../docs/images/simulation.png', 'Figure 4. Actual application capture showing emergency-aware DQN and rule-based preemption at the same simulation time. This interactive demonstration is separate from the held-out evaluation.')
        self.numbered(['Configure the traffic level, focused approaches, emergency vehicle and controller selections.',
                       'Start the run, use the shared playback controls and inspect the live junction.',
                       'Inject an ambulance at the live edge or pause and replay recorded states.',
                       'Stop the workers cleanly, or restart to apply a new configuration.'])

        self.next_section()
        self.heading('Experimental Evaluation')
        self.heading('4.1 Evaluation Measures and Protocol',2);self.source_body(7)
        self.heading('4.2 Held Out Results and Statistical Analysis',2);self.source_body(8)
        self.heading('4.3 Implementation Difficulties and Resolutions',2);self.source_body(10)
        self.heading('4.4 Verification and Reproducibility',2);self.source_body(11)
        self.heading('4.5 Limitations and Further Work',2);self.source_body(12)

        self.heading('Reflections',page=True)
        self.paragraph('Working on Adaptive Crossroads Control with Emergency Vehicle Priority connected reinforcement learning theory with the practical demands of a complete simulation system. Designing the observation and reward made the control objective concrete: short queues do not always mean fair service, and an ambulance can be impeded before it comes to a complete stop. These considerations motivated explicit waiting-age and emergency-proximity information rather than relying only on total traffic load.')
        self.paragraph('An important lesson was that a learned controller must be assessed against strong and understandable alternatives. Emergency-aware DQN reduced mean ambulance passage relative to standard DQN in the held-out scenarios, while rule-based preemption retained a slightly lower passage mean. The learned policies also showed lower ordinary stopped wait than the rule. Reporting this tradeoff was more informative than presenting one favourable score as a universal improvement.')
        self.paragraph('The implementation also demonstrated why experimental integrity and interface behaviour must be considered together. Matching traffic seeds is not sufficient if two displays drift in time, and acknowledging an injection is not the same as observing the ambulance in the network. Separate simulation workers, state barriers, immutable recordings and explicit request/active counts made these distinctions observable. Smooth playback improved the presentation while preserving actual simulator endpoints and event ordering.')
        self.paragraph('The final outcome is a working academic project with an inspectable dataset, trained models, a simulation application and reproducible results. The project strengthened my understanding of deep Q-learning, controlled evaluation, finite-horizon metrics and software verification. It also made the limits clear: synthetic single-junction evidence and zero observed simulator safety events do not establish real-world safety or guarantee gridlock prevention. Further experiments on independent networks and more diverse traffic remain necessary.')
        self.heading('References',page=True);self.source_body('References')
        # Mark all fields for refresh as a fallback; native Word refresh is the final gate.
        update = self.doc.settings.element.find(qn('w:updateFields'))
        if update is None: update=OxmlElement('w:updateFields');self.doc.settings.element.append(update)
        update.set(qn('w:val'),'true')
        working = ROOT/'cache/report_template/working.docx'
        working.parent.mkdir(parents=True,exist_ok=True)
        self.doc.save(working)
        generated = self.zip_parts(working)
        # Preserve opaque/template parts byte-for-byte; replace only editable parts.
        editable = {'word/document.xml','word/_rels/document.xml.rels','word/settings.xml',
                    'word/numbering.xml','docProps/core.xml','[Content_Types].xml'}
        parts = dict(self.parts)
        for n in editable: parts[n]=generated[n]
        for n, content in generated.items():
            if n.startswith('word/media/') and n not in self.parts: parts[n]=content
        # Keep only images referenced by the rewritten body plus the retained institute logo.
        document = E.fromstring(parts['word/document.xml'])
        used = set(document.xpath('//@r:embed | //@r:link',namespaces=NS))
        rels = E.fromstring(parts['word/_rels/document.xml.rels'])
        keep_media = {'word/media/image1.jpg'}
        for rel in list(rels):
            if rel.get('Type','').endswith('/image'):
                if rel.get('Id') not in used:
                    rels.remove(rel)
                else:
                    keep_media.add('word/'+rel.get('Target'))
        parts['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
        parts = {n:b for n,b in parts.items() if not n.startswith('word/media/') or n in keep_media}
        with ZipFile(output,'w',compression=ZIP_DEFLATED) as archive:
            for n,b in parts.items():archive.writestr(n,b)
        if output.parent.resolve() == OUT.resolve():
            (OUT/'Adaptive_Crossroads_Project_Report.md').write_text('\n\n'.join(self.markdown),encoding='utf-8')
        summary = dict(reference_sha256=digest(self.template),output_sha256=digest(output),
                       sections=len(self.doc.sections),headings=self.heading_count,figures=self.image_count,tables=self.table_count,
                       preserved_parts=[n for n in self.parts if n in parts and parts[n]==self.parts[n]],
                       removed_content_images=[n for n in self.parts if n.startswith('word/media/') and n not in parts])
        verification = OUT/'template_verification.json' if output.parent.resolve()==OUT.resolve() else output.with_suffix('.verification.json')
        verification.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,default=OUT/'Adaptive_Crossroads_Project_Report.docx')
    parser.add_argument('--content',type=Path,default=OUT/'report_content.json')
    parser.add_argument('--output',type=Path,default=OUT/'Adaptive_Crossroads_Project_Report.docx')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    from verify_repository import evidence
    checked=evidence()
    previous=json.loads((OUT/'report_evidence.json').read_text())
    checked.update({k:v for k,v in previous.items() if k.startswith('offline_')})
    (OUT/'report_evidence.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
    Builder(args.template,args.content).build(args.output)


if __name__=='__main__':main()
