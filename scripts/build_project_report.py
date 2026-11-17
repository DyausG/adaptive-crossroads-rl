"""Build the standalone project report from verified, repository-contained results."""
import hashlib
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_repository import evidence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                               PageBreak, Table, TableStyle, Image, KeepTogether, Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = ROOT / 'reports/final'
OUT.mkdir(parents=True, exist_ok=True)
META = json.loads((OUT / 'metadata.json').read_text())
FACTS = evidence()
METRICS = FACTS['summaries']
ORDER = ['fixed', 'rule', 'standard', 'emergency', 'hybrid']
NAMES = ['Fixed-time', 'Rule preemption', 'Standard DQN', 'Emergency DQN', 'Hybrid Rule+DQN']
PALETTE = ['#7B858D', '#187F76', '#4D73AB', '#CE7755', '#8672A3']
BLUE = colors.HexColor('#203E53')
GRAY = colors.HexColor('#64727A')
WIDTH = 6.9 * inch


def fonts():
    windows = Path('C:/Windows/Fonts')
    fallback = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    for name, win, linux in [('Body', 'georgia.ttf', 'DejaVuSerif.ttf'),
                             ('BodyBold', 'georgiab.ttf', 'DejaVuSerif-Bold.ttf'),
                             ('Sans', 'arial.ttf', 'DejaVuSans.ttf'),
                             ('SansBold', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'),
                             ('Mono', 'consola.ttf', 'DejaVuSansMono.ttf')]:
        file = windows / win
        pdfmetrics.registerFont(TTFont(name, str(file if file.exists() else fallback / linux)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='SansBold', italic='Sans', boldItalic='SansBold')


fonts()
STYLES = dict(
    body=ParagraphStyle('Body', fontName='Body', fontSize=10.7, leading=15.2, spaceAfter=9, alignment=TA_JUSTIFY),
    h1=ParagraphStyle('H1', fontName='SansBold', fontSize=16, leading=20, textColor=BLUE, spaceBefore=20, spaceAfter=11, keepWithNext=True),
    h2=ParagraphStyle('H2', fontName='SansBold', fontSize=12, leading=16, textColor=BLUE, spaceBefore=13, spaceAfter=7, keepWithNext=True),
    caption=ParagraphStyle('Caption', fontName='Sans', fontSize=9, leading=12, textColor=GRAY, spaceBefore=6, spaceAfter=12),
    cell=ParagraphStyle('Cell', fontName='Sans', fontSize=9, leading=12, spaceAfter=0),
    header=ParagraphStyle('Header', fontName='SansBold', fontSize=9, leading=12, textColor=colors.white),
    code=ParagraphStyle('Code', fontName='Mono', fontSize=8, leading=11, spaceBefore=5, spaceAfter=10),
    title=ParagraphStyle('Title', fontName='SansBold', fontSize=27, leading=34, textColor=colors.black, spaceAfter=22),
    subtitle=ParagraphStyle('Subtitle', fontName='Sans', fontSize=14, leading=20, textColor=BLUE, spaceAfter=14),
    toc=ParagraphStyle('TOC', fontName='Sans', fontSize=10, leading=16, leftIndent=0, firstLineIndent=0, spaceBefore=3),
)
STORY, MARKDOWN = [], []


class Report(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=letter, leftMargin=.8*inch, rightMargin=.8*inch,
                         topMargin=.7*inch, bottomMargin=.7*inch,
                         title=META['title'], author=META['author'], subject=META['course'])
        self.addPageTemplates(PageTemplate('report', [Frame(self.leftMargin, self.bottomMargin,
                            self.width, self.height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont('Sans', 8)
            canvas.setFillColor(GRAY)
            canvas.drawString(self.leftMargin, .35*inch, 'Adaptive Crossroads | Reinforcement Learning UCS760')
            canvas.drawRightString(letter[0] - self.rightMargin, .35*inch, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'H1':
            title = flowable.getPlainText()
            key = 'section_' + hashlib.sha1(title.encode()).hexdigest()[:12]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0)
            self.notify('TOCEntry', (0, title, self.page, key))


def p(text):
    STORY.append(Paragraph(text, STYLES['body']))
    MARKDOWN.append(text.replace('<b>', '**').replace('</b>', '**').replace('&gt;', '>').replace('&lt;', '<'))


def h(text, level=1):
    STORY.append(Paragraph(text, STYLES['h1' if level == 1 else 'h2']))
    MARKDOWN.append('#' * (level + 1) + ' ' + text)


def table(headers, rows, widths=None, caption=None, keep=False):
    data = [[Paragraph(escape(str(c)), STYLES['header']) for c in headers]]
    data += [[Paragraph(escape(str(c)), STYLES['cell']) for c in row] for row in rows]
    item = Table(data, colWidths=widths or [WIDTH/len(headers)]*len(headers), repeatRows=1, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), .45, colors.HexColor('#D9D9D9')),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F1F4F6')]),
    ]))
    STORY.extend([Spacer(1, 4), KeepTogether([item]) if keep else item, Spacer(1, 7)])
    MARKDOWN.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'])
    MARKDOWN.extend('| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows)
    if caption:
        STORY.append(Paragraph(caption, STYLES['caption']))
        MARKDOWN.append(caption)


def figure(path, caption, width=WIDTH):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        height = width * im.height / im.width
    STORY.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, STYLES['caption'])]))
    MARKDOWN.append(f'![{caption}]({Path(path).relative_to(OUT).as_posix() if Path(path).is_relative_to(OUT) else "../../" + Path(path).relative_to(ROOT).as_posix()})')


def code(text):
    STORY.append(Preformatted(text, STYLES['code']))
    MARKDOWN.append('```text\n' + text + '\n```')


def charts():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#203E53', 'text.color': '#203E53', 'savefig.dpi': 190})
    labels = ['Fixed-time', 'Rule\npreemption', 'Standard\nDQN', 'Emergency\nDQN', 'Hybrid\nRule+DQN']
    for key, ylabel, name, sd in [('passage_mean','Mean detection-to-clear time (s)','passage.png',True),
                                 ('regular_wait','Mean stopped wait per inserted vehicle (s)','regular_wait.png',False)]:
        fig, ax = plt.subplots(figsize=(6.9, 3.25), layout='constrained')
        vals = [METRICS[c][key] for c in ORDER]
        errors = [METRICS[c]['seed_sd'] or 0 for c in ORDER] if sd else None
        bars = ax.bar(labels, vals, color=PALETTE, yerr=errors, capsize=4 if sd else 0, width=.65)
        ax.bar_label(bars, fmt='%.1f', padding=9 if sd else 3, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals)*1.25)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5E9EC')
        fig.savefig(OUT/name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.9, 2.5), layout='constrained')
    ax.set_axis_off()
    boxes = [('Scenario\nconfiguration', .01), ('Shared route\nand seed', .25), ('Isolated SUMO\nworkers', .49), ('Shared clock\nand canvas', .73)]
    for text, x in boxes:
        ax.add_patch(plt.Rectangle((x,.35), .22,.4, facecolor='#EDF3F5', edgecolor='#8297A4', linewidth=1.1))
        ax.text(x+.11,.55,text,ha='center',va='center',fontsize=10)
        if x < .7:
            ax.annotate('',xy=(x+.245,.55),xytext=(x+.22,.55),arrowprops={'arrowstyle':'->','color':'#203E53'})
    ax.text(.5,.13,'Raw states and metadata are recorded before visual interpolation',ha='center',fontsize=10)
    fig.savefig(OUT/'architecture.png')
    plt.close(fig)


def main():
    charts()
    MARKDOWN.append('# ' + META['title'])
    STORY.append(Spacer(1, .85*inch))
    STORY.append(Paragraph(META['title'], STYLES['title']))
    STORY.append(Paragraph('Project Report', STYLES['subtitle']))
    STORY.append(Spacer(1, .35*inch))
    for line in [META['author'], 'Roll number ' + META['roll_number'], META['course'], META['date']]:
        STORY.append(Paragraph(escape(line), STYLES['subtitle']))
        MARKDOWN.append(line)
    STORY.append(Spacer(1, .45*inch))
    p('A reproducible single-intersection study of adaptive signal control, ambulance priority and ordinary-traffic delay, with trained DQN policies and a synchronized desktop simulator.')
    p('Repository: ' + META['repository'])
    STORY.append(PageBreak())
    STORY.append(Paragraph('Abstract', STYLES['h2']))
    p('Fixed traffic-signal schedules cannot respond directly to changing demand or an approaching ambulance. This project models a four-way signalized junction in SUMO, formulates signal selection as a Markov decision process, and compares fixed-time control, legal rule-based preemption, standard deep Q-learning, emergency-aware deep Q-learning and an explicitly identified hybrid controller. The study uses 96 synthetic routes arranged as 48 matched normal/emergency pairs, with separate training, validation and held-out test partitions. Six DQN policies are trained for 40000 decisions each. The 264 test episodes include 12 emergency routes per controller and three learner seeds for learned policies. Mean detection-to-clear passage is 98.3 seconds for fixed-time, 38.2 for rule preemption, 75.6 for standard DQN, 41.4 for emergency-aware DQN and 33.5 for hybrid Rule+DQN. Emergency-aware DQN improves the standard DQN mean by 45.2 percent on this test set, but does not outperform rule preemption on that mean. Ordinary-traffic wait and remaining backlog reveal the corresponding tradeoffs. A native desktop application supports reproducible scenarios, genuine live injections, smooth position interpolation and fair synchronized comparisons. All inputs, checkpoints and evaluation records are file-based and included with the project. Results demonstrate behavior in a small synthetic simulation, not real-world safety or general gridlock prevention.')
    STORY.append(Paragraph('Contents', STYLES['h2']))
    toc = TableOfContents()
    toc.levelStyles = [STYLES['toc']]
    STORY.append(toc)
    STORY.append(PageBreak())

    h('1 Problem statement and objectives')
    p('The project addresses the conflict between predictable ordinary-traffic service and rapid ambulance passage at a signalized crossroads. A fixed cycle may leave an ambulance facing red despite an otherwise serviceable approach. Giving an ambulance priority can shorten its delay while making other approaches wait. An adaptive controller therefore needs information about both network demand and the emergency vehicle, together with a measurable definition of service quality.')
    p('The central question is whether an emergency-aware learned policy can reduce ambulance impedance relative to a standard learned traffic controller under identical traffic and legal phase-transition constraints. Fixed-time and rule-based controllers provide interpretable reference points. The hybrid tests the distinct choice of using deterministic emergency priority together with a learned policy for ordinary traffic. It is not treated as a purely learned solution.')
    p('The implementation objectives are to build a valid simulator interface, produce reproducible traffic inputs, train and reload actual policies, evaluate them on reserved scenarios, record ordinary-traffic costs and safety events, and expose the resulting behavior in a usable simulation. Submission includes code, a dataset, models, raw results, test evidence and this report. There is no database service; a file-based structure makes the complete evidence inspectable offline.')
    h('2 Technical background')
    p('Reinforcement learning trains an agent through state observations, actions and rewards rather than through labeled correct actions. In this project the agent requests one of four movement groups at a single junction. Traffic evolves inside SUMO; the environment returns a new observation and reward after a decision interval. Deep Q-learning approximates the value of an action using a neural network. Experience replay and a separate target network reduce the instability associated with correlated observations and a moving training target [1].')
    p('SUMO-RL provides the Gymnasium-compatible environment and legal signal transition interface [3]. Stable-Baselines3 supplies the DQN implementation [2]. These are implementation dependencies; the project-specific work defines the traffic scenarios, emergency and fairness observations, reward terms, controller comparisons, evaluation measures and interactive application. SUMO supplies microscopic vehicle dynamics and actual signal, insertion and crossing events [4].')
    p('Rule-based preemption is an essential comparator because ambulance priority has a simple, observable trigger: an emergency vehicle approaches a red movement. A learned policy should not be declared successful simply because it beats a fixed cycle. It should also be compared with this direct rule, while measuring how the rule affects regular traffic. The reported results preserve that distinction.')

    STORY.append(PageBreak())
    h('3 System design and implementation')
    figure(OUT/'architecture.png', 'Figure 1. Runtime data flow. Side-by-side operation sends identical traffic inputs to separate SUMO processes and presents their actual states using one playback clock.')
    p('The system has three parts. The experiment pipeline generates traffic, trains policies and writes evaluation records. The simulator adapter constructs observations, applies the shared phase constraints and records telemetry. The native Tk application configures and displays a live run. Separating these parts prevents visual controls or smoothing from changing scientific evaluation data.')
    table(['Component','Implementation role'], [
        ['Scenario generation','generate_routes.py for fixed benchmark inputs; ui_scenarios.py for composable interactive inputs'],
        ['Environment and policies','environment.py, controllers.py, train.py and evaluate.py'],
        ['Desktop presentation','app.py and ui_playback.py'],
        ['Simulation process','sim_worker.py loads an actual checkpoint, advances SUMO and streams JSON states'],
        ['Evidence checks','tests/, verify_repository.py and validate_ui.py'],
    ], [1.5*inch,5.4*inch])
    p('libsumo has process-level simulation state, so a comparison runs one child process per junction. Worker commands and stdout states travel through queues and pipes. The application only accepts a comparison batch when all workers reach the same simulated second. It also checks scenario identity, route SHA-256 and simulator seed. A failure stops the run instead of silently showing unmatched states.')
    p('The simulator network contains a single four-way junction with incoming approaches extended to approximately 389.6m. This places a complete 250m emergency detection window inside the network. Connections and signal links define the actual permitted turns. The canvas reads network lane geometry and real vehicle coordinates rather than placing illustrative markers on arbitrary paths.')

    h('4 Dataset and experimental protocol')
    p('The primary dataset contains 96 synthetic XML route files, not measured city traffic. Each background random seed generates two files: one without an ambulance and one with the same regular-vehicle schedule plus an ambulance. The two files share a simulator seed. Ambulance timing uses an independent random stream, so generating an emergency vehicle does not consume or change regular-traffic draws. This permits within-scenario normal/emergency comparisons under controlled inputs.')
    table(['Partition','Background seeds','Pairs','Route files'],[
        ['Training','10000-10023','24','48'],['Validation','20000-20011','12','24'],['Held-out test','30000-30011','12','24'],['Total','Disjoint partitions','48','96']],
        [1.6*inch,2.6*inch,1.0*inch,1.7*inch], 'Table 1. Primary dataset partition. A pair contains the same regular arrivals with and without an ambulance.')
    p('Per-approach Poisson arrival rates are 0.055, 0.12 and 0.21 vehicles per second for low, medium and rush conditions. Directional patterns are symmetric, north/south-heavy or east/west-heavy. The heavy axis receives a 1.45 multiplier and the other axis 0.55, keeping total expected demand constant across these benchmark patterns. Vehicles choose straight movements with probability 0.8 and each other permitted destination with probability 0.1. Regular arrivals stop at 900s and all episodes continue to 1200s, leaving a 300s drain period.')
    p('Ambulance requested departure lies between 240s and 420s, stratified into early and late windows. Direction, demand and imbalance have balanced coverage in each 12-pair block. This is a compact balanced design, not a full factorial: four directions, three demand levels and three imbalances would require 36 combinations before replication. The design therefore cannot identify every interaction or substitute for testing on different networks.')
    p('Training and evaluation seeds are disjoint. Models use learner seeds 0, 1 and 2, which are separate from traffic seeds. Checkpoints are selected at the fixed 40000-decision budget rather than by test performance. Each validation and test split runs both deterministic controllers once and each of the three seeded learned/hybrid controllers three times: 24 routes times (2 + 3 times 3) equals 264 episodes. The result table below restricts attention to ambulance-present cases, leaving 132 episode records.')
    p('Manifest files identify split, configuration, vehicle count, input path and SHA-256. The repository also retains an additional experiment fixture set used by integrity and regression checks; it is not pooled into the primary results reported here. DATASET.md identifies every included input and its purpose. Interactive configurations have separate storage and are excluded from the benchmark.')

    h('5 Reinforcement learning formulation')
    h('State action and transition',2)
    p('The standard observation has 26 normalized features. The base 21 features encode the active phase, whether its minimum green time has elapsed, incoming-lane density and halting queues. Four additional features encode maximum regular-vehicle waiting age on each approach, capped and normalized at 120s. One feature encodes elapsed phase duration relative to the maximum-green limit. Queue age makes low-flow approaches visible even when their queues are short.')
    p('The emergency-aware observation adds eight features, producing 34 dimensions. Each direction contributes an ambulance-presence flag and maximum proximity inside 250m. Proximity is one minus distance divided by 250, clipped to zero through one. Only incoming ambulances with the junction as their next signal are considered. Both policies use the same discrete action space: north/south through and right, north/south protected left, east/west through and right, or east/west protected left.')
    p('Decisions occur every 5 simulated seconds. Requests pass through a shared state machine with 3s yellow, 10s minimum green and a 60s maximum-green setting. A request does not instantly bypass an ongoing phase. These timing settings constrain every controller identically. Ambulances retain speedMode 31, and the project does not enable a bluelight device or red-light bypass.')
    h('Reward and training objective',2)
    p('The standard reward is the upstream difference in accumulated waiting-time reward, minus 0.15 times the sum of normalized maximum approach waits. The emergency-aware reward subtracts an additional penalty for each detected ambulance: 1.5 times (1 + proximity) when its movement is not green, plus 2 when stopped, plus 0.04 times its waiting time capped at 120s. The red/yellow approach penalty supplies a signal before the ambulance is fully stopped. These coefficients are design choices and are not established as globally optimal.')
    code('standard reward = difference in waiting-time reward\n                  - 0.15 * sum(normalized approach waits)\nemergency reward = standard reward\n                   - sum(red/yellow approach + stop + wait penalties)\nDQN nonterminal target = reward + gamma * max Q_target(next_state, action)')
    p('DQN minimizes the discrepancy between its estimated action value and the target based on observed reward and the target network. Epsilon-greedy exploration allows phase choices beyond the current greedy estimate during training. Evaluation uses deterministic model prediction. Reward curves of standard and emergency policies cannot be compared as domain performance scores because their reward definitions differ; passage, traffic delay and backlog provide the common measures.')

    STORY.append(PageBreak())
    h('6 Controllers and training configuration')
    table(['Controller','Decision rule'],[
        ['Fixed-time','Requests four movement groups in an 80s cycle with nominal slots 25/15/25/15s, subject to the common legal transition state machine.'],
        ['Rule preemption','Requests a green-compatible phase for the nearest approaching ambulance within 250m; uses the fixed cycle when none is detected.'],
        ['Standard DQN','Learns from the 26-feature state and ordinary-traffic reward without emergency-specific features or penalties.'],
        ['Emergency-aware DQN','Learns from the 34-feature state and emergency-aware reward.'],
        ['Hybrid Rule+DQN','Uses rule preemption within the detection window and the matched-seed standard DQN otherwise. No separate hybrid network is trained.'],
    ], [1.45*inch,5.45*inch], 'Table 2. Evaluated policies. Rule-based priority embedded in the hybrid is explicitly part of that controller.', keep=True)
    table(['Parameter','Recorded value'],[
        ['Network','MLP with two hidden layers of 64 units'],['Learning rate / discount','0.0005 / 0.99'],
        ['Replay capacity / learning starts','50000 transitions / 1000 decisions'],['Batch / training frequency','64 / every 4 decisions, one gradient step'],
        ['Target update interval','500 decisions'],['Exploration','Epsilon 1.0 to 0.05 over the first 35% of the budget'],
        ['Budget / learner seeds','40000 decisions per model / 0, 1, 2'],['Execution','CPU; one PyTorch training thread; deterministic evaluation'],
    ], [2.5*inch,4.4*inch], 'Table 3. Configuration recorded with each of the six primary checkpoints.')
    times = [m['seconds'] for m in FACTS['models'] if m['experiment']=='v2']
    p(f'The six primary models completed a total of 240000 decisions. Their recorded training times range from {min(times):.2f}s to {max(times):.2f}s, totaling {sum(times):.2f} process-seconds. These are machine-specific recorded run timings, not a promise for a fresh machine. Every final checkpoint was reloaded and used for a valid prediction. Configuration files record observation schema, feature count, learner seed, input-manifest hash and library version to reject incompatible checkpoint/state combinations.')

    h('7 Evaluation measures and statistical treatment')
    p('<b>Ambulance passage</b> is the interval between first detection inside 250m and first observation on an outgoing road. It is neither total journey time nor requested-departure-to-clear time. The latter also includes insertion delay and travel before detection. Reporting both prevents a controller from appearing better simply because an ambulance entered the simulated network later.')
    p('<b>Ordinary-traffic stopped wait</b> is accumulated stopped vehicle-seconds divided by inserted regular vehicles within an episode, then averaged equally across episodes. It includes observed waiting of unfinished inserted trips, but excludes vehicles that never entered. The stopped-plus-pending congestion measure integrates stopped and pending counts over the horizon and divides by requested vehicles. Unfinished and uninserted counts remain visible to expose horizon and congestion effects.')
    p('<b>Queue recovery</b> begins at clearance and uses a threshold of max(1, 1.1 times the mean queue in the preceding 60s before emergency detection). Recovery requires 15 consecutive one-second samples at or below that threshold. Episodes with no observed recovery before 1200s are right-censored and cannot be treated as zero or dropped silently. Excess queue area integrates queue above the threshold for up to 120s after clearance.')
    p('Reported passage means pool the completed emergency episodes within a controller. The p95 is the 95th percentile of pooled episode passages using linear interpolation, not the average of per-seed p95 values. For learned controllers, the reported seed standard deviation is the sample standard deviation of the three seed means. It is not a confidence interval, and 36 episode records are not 36 independently trained agents. No hypothesis test or statistically significant superiority is claimed.')
    p('Matching background inputs enables controlled within-simulation comparisons, including emergency-minus-normal delay. The stochastic simulator can still diverge after interventions affect traffic interactions. Results describe this fixed scenario family. Different demand patterns, pedestrian traffic, sensor noise or a larger road network may change the ranking.')

    h('8 Experimental results and analysis')
    table(['Controller','Runs','Mean (s)','Pooled p95 (s)','Scheduled to clear (s)'],[
        [NAMES[i],str(METRICS[c]['runs']),f"{METRICS[c]['passage_mean']:.1f}",f"{METRICS[c]['passage_p95_pooled']:.1f}",f"{METRICS[c]['scheduled_to_clear_mean']:.1f}"] for i,c in enumerate(ORDER)],
        [1.65*inch,.55*inch,1.0*inch,1.4*inch,2.3*inch], 'Table 4. Ambulance-present test episodes only. Means and p95 describe the same fixed held-out route family; deterministic baselines have no learner-seed replication.')
    figure(OUT/'passage.png', 'Figure 2. Mean ambulance passage. Error bars for learned/hybrid policies show sample SD of three learner-seed means; the deterministic baselines have no learner-seed error bar.')
    reduction = 100*(METRICS['standard']['passage_mean']-METRICS['emergency']['passage_mean'])/METRICS['standard']['passage_mean']
    p(f'Emergency-aware DQN achieves a mean passage of 41.4s versus 75.6s for standard DQN, a descriptive reduction of {reduction:.1f} percent on the same held-out traffic. Its pooled p95 is 105.2s versus 215.5s for standard DQN. This is consistent with the intended effect of adding emergency state and reward information, but the experiment changes both together and therefore does not isolate which component causes the difference.')
    p('Rule preemption reaches 38.2s mean passage, slightly lower than emergency-aware DQN. The learned policy should therefore not be presented as the best ambulance-priority controller on mean passage. Hybrid reaches 33.5s, but its priority logic is explicitly rule-based; the learned component operates outside the detection window. Tail values also matter: all three emergency-priority methods have pooled p95 near 105-106s, far above their means.')
    table(['Learned policy','Seed 0 mean (s)','Seed 1 mean (s)','Seed 2 mean (s)','Seed SD (s)'],[
        [NAMES[ORDER.index(c)],*[f'{x:.1f}' for x in METRICS[c]['seed_means']],f"{METRICS[c]['seed_sd']:.1f}"] for c in ('standard','emergency','hybrid')],
        [1.8*inch,1.3*inch,1.3*inch,1.3*inch,1.2*inch], 'Table 5. Seed-level passage means, each averaged over the same 12 emergency routes.')
    figure(OUT/'regular_wait.png', 'Figure 3. Regular-vehicle stopped wait in the emergency-present test cases. These are equal-weight averages of episode means over inserted regular vehicles; uninserted demand is accounted for separately.')
    table(['Controller','Stopped wait (s)','Congestion (s/request)','Unfinished','Uninserted'],[
        [NAMES[i],f"{METRICS[c]['regular_wait']:.1f}",f"{METRICS[c]['congestion']:.1f}",str(METRICS[c]['unfinished_regular']),str(METRICS[c]['uninserted_regular'])] for i,c in enumerate(ORDER)],
        [1.7*inch,1.3*inch,1.8*inch,1.05*inch,1.05*inch], 'Table 6. Ordinary-traffic outcomes in ambulance-present episodes. Learned/hybrid count totals pool three model seeds; deterministic totals cover one run per route and must not be compared as equal-exposure event rates.')
    p('Emergency-aware DQN averages 36.8s ordinary stopped wait, close to standard DQN at 35.9s and below rule preemption at 61.6s. This illustrates the intended tradeoff: the rule has a slightly better mean ambulance passage, while the learned traffic policies show lower average stopped delay for inserted ordinary vehicles. Means alone are insufficient because congested cases can leave vehicles pending or unfinished.')
    p('Across all 264 test episodes, including normal cases, there are 4859 unfinished regular trips and 1416 scheduled regular vehicles that never enter before the horizon. There are no incomplete ambulance passages among the 132 emergency episodes. Recorded ambulance red crossings, collisions and teleports are all zero in this evaluation. These findings do not mean all demand was served or that gridlock is impossible. One rule-preemption queue-recovery episode is censored; the other controllers have no recovery censoring in their emergency test records.')
    p('Mean post-clearance excess queue area over 120s is 1177.4 vehicle-seconds for fixed-time, 1089.9 for rule preemption, 697.4 for standard DQN, 533.3 for emergency-aware DQN and 707.4 for hybrid. This measure captures residual congestion after emergency clearance. Its ranking differs from mean passage, reinforcing the need to report more than one outcome.')

    h('9 Simulation application and user workflow')
    p('The interface centers on a large junction view with compact telemetry below it. Single mode uses one policy; Side-by-side uses two independent workers synchronized on the same traffic. Controller switches are exclusive because the supported policies are distinct. When all switches are off, the interface explicitly labels standard DQN. Fixed-time, rule preemption and emergency-aware RL are selectable without implying a nonexistent combined policy.')
    p('Traffic controls select low, medium or rush demand; any subset of heavy approaches; ambulance presence; and its origin. In interactive scenarios, each selected direction receives 45 percent more arrivals without reducing the other directions. None selected means balanced expected demand. The scheduled interactive ambulance requests departure at 300s. The traffic seed controls route generation and simulator randomness; the displayed learned policy uses checkpoint seed 0. Interactive traffic is not mixed with held-out benchmark statistics.')
    p('The app advances actual SUMO states one second at a time and interpolates positions and shortest-angle heading changes between consecutive states. Signal changes, vehicle creation/removal and telemetry retain their true state boundaries. Pause freezes the current visual interpolation; resume continues from it. Seeking displays recorded history and pauses both views. It does not rewind or branch a running simulator, and it cannot seek into an uncomputed future.')
    p('Live injection sends the same ambulance request to both workers at a shared boundary. The worker acknowledges a request before physical insertion; SUMO may delay insertion if a lane is occupied. The app reports active ambulances from the displayed actual state, avoiding an artificial emergency animation. Stop closes workers and waits asynchronously for cleanup, with termination as a bounded fallback. Restart waits for prior processes to retire, clears playback state and applies pending configuration changes.')
    figure(ROOT/'docs/images/simulation.png', 'Figure 4. Actual synchronized application capture. Vehicle coordinates, phase lamps and telemetry come from separate SUMO workers at the same time. This is an interactive demonstration rather than a held-out result plot.')

    h('10 Implementation difficulties and resolutions')
    table(['Problem encountered','Resolution and evidence'],[
        ['Short queues can hide long waiting age','Added maximum approach-wait features and an elapsed-phase feature, together with a starvation penalty. The observation/schema tests verify dimensions and bounded features.'],
        ['Emergency reward can arrive too late if tied only to stopping','Included proximity-conditioned red/yellow approach impedance before a complete stop, alongside stop and waiting penalties. Reward tests distinguish approaching-red and green cases.'],
        ['Ambulance presence could change background traffic draws','Used independent random streams for regular traffic and ambulance timing. Pairing tests check exact equality of ordinary vehicle schedules.'],
        ['Two comparisons could drift or share simulator state','Isolated libsumo workers and accepted only equal-time state batches with matching input hashes and traffic seeds. All ordered controller pairs were exercised.'],
        ['One-second snapshots produced visibly stepped movement','Interpolated only presentation coordinates/headings, retaining immutable raw traces and discrete state boundaries. Pause and seek tests verify stable positions and history preservation.'],
        ['An acknowledged injection was not always physically inserted immediately','Separated request status from active-vehicle counts and advanced actual simulator states until lane space permitted insertion. No forced insertion or bypass was introduced.'],
        ['Small windows and repeated runs could hide controls or leave work active','Reserved playback controls before the expandable viewport, added scrolling configuration, made headings responsive, tracked the UI timer and reaped child processes before restart. Desktop checks cover minimum size and startup cancellation.'],
    ], [2.1*inch,4.8*inch], 'Table 7. Engineering problems, implemented responses and observable checks. These are implementation decisions, not ablation results proving each individual performance contribution.')
    p('A worker regression test initially used a 45s cold-start deadline while the application allowed 90s. Concurrent imports caused one intermediate timeout. The test was aligned with the app deadline and cancellation was distinguished from normal graceful exit. The final targeted rerun passed all 22 selected worker checks. Cold import time remains dependent on machine load; cancellation keeps the UI responsive during startup.')

    h('11 Verification reproducibility and storage')
    table(['Verification','Recorded result'],[
        ['Complete simulator/data/state test suite','300 tests passed'],['Final actual-worker rerun','22 selected checks passed'],
        ['Full native desktop integration','24 checks passed, including two simulations completing 1200s'],
        ['Final layout and interaction run','23 checks passed; captures inspected at 1280x850 and 1040x720'],
        ['Repository evidence audit','168 included routes and 12 final checkpoints checked; primary result summaries independently recomputed from raw episodes'],
        ['Offline replay','Two archived policies reproduced their recorded metrics with external connections blocked'],
    ], [3.0*inch,3.9*inch], 'Table 8. Verification scopes overlap; counts are separate recorded runs and should not be added as independent experiments. Primary experiment performance comes from the 264 held-out episodes, not UI tests.')
    p('A file-based repository supports offline inspection. The primary dataset is under experiments/v2/main/data; trained models and run configurations are under its models directory; validation and test CSV/JSON records are under results/evaluation. Manifest and checkpoint hashes connect the evaluated runs to exact inputs and policies. No SQLite, SQL, MongoDB or other database is used. Generated interactive routes and recordings are kept separate from the versioned benchmark inputs.')
    p('The installed project was tested on Windows x64 with Python 3.12, SUMO/libsumo 1.28.0, Stable-Baselines3 2.9.2, PyTorch 2.14.1+cpu and the pinned dependency set. The application is native Tk, so browser testing does not apply. Internet is needed to install dependencies; installed simulation and inference run offline. A full dependency installation on every supported operating system is not claimed. A Windows CI workflow template is included for activation in GitHub Actions. Its presence does not establish a successful remote CI run; the README explains activation.')
    code('python -m pip install --extra-index-url\n  https://download.pytorch.org/whl/cpu -r requirements.txt\npython app.py\npython verify_repository.py --replay\npython -m pytest tests -q -p no:cacheprovider\npython validate_ui.py --screenshots')
    p('The first installation command is wrapped for printing; enter it as one line. Supplied checkpoints run without retraining. For new experiments, training supports a separate output group and evaluation supports a separate output root. The contribution guide documents these options so an evaluator need not overwrite supplied evidence. The report builder independently recomputes its numeric evidence before generating the PDF and editable Markdown source.')

    h('12 Limitations conclusions and further work')
    p('The experiment uses one synthetic four-way network, Poisson demand, limited directional patterns and three learner seeds. It omits pedestrians, coordinated junctions, realistic emergency dispatch, noisy sensing and changing road geometry. The compact balanced design does not cover every interaction among direction, intensity and imbalance. Hyperparameter/reward ablations and independent network replication are needed before assigning performance changes to an individual feature or generalizing the controller ranking.')
    p('The finite horizon creates unfinished and uninserted ordinary trips. Their counts and the pending-aware congestion metric should remain visible whenever mean delay is quoted. Zero observed simulation safety events do not establish real-world safety. The multiple-ambulance sandbox is a visual experiment: existing passage/recovery tracking follows the first observed ambulance, so these interactive events do not establish a validated multi-ambulance benchmark. Straight interpolation between one-second positions also approximates motion during turns; raw endpoints remain exact.')
    p('The project demonstrates a measurable emergency-aware learned policy with lower mean ambulance passage than standard DQN on controlled held-out traffic, alongside lower ordinary stopped delay than the direct preemption rule. The rule still has the lower mean ambulance passage of those two policies, and the hybrid achieves the best mean passage in this sample with its explicit rule component. This supports a practical comparison of priority and ordinary-traffic service rather than a claim that learning dominates every baseline.')
    p('Further work should test independent networks and non-Poisson demand, add explicit pedestrian and transit constraints, validate multiple ambulances with per-vehicle metrics, compare additional RL algorithms under equal budgets, perform feature/reward ablations, and expand seed replication. Those are proposed extensions, not completed capabilities of this submission.')

    h('References')
    refs = [
        '[1] V. Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 529-533, 2015. DOI 10.1038/nature14236. https://doi.org/10.1038/nature14236',
        '[2] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus and N. Dormann. Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 22(268), 1-8, 2021. https://www.jmlr.org/papers/v22/20-1364.html',
        '[3] L. N. Alegre and contributors. SUMO-RL source and documentation. Pinned revision 86f0fc41401eeca724b2e8878f5020dc6d70c9b4. MIT license. https://github.com/LucasAlegre/sumo-rl',
        '[4] Eclipse SUMO. Traffic lights, TraCI, libsumo and emergency-vehicle documentation. Runtime version 1.28.0. https://sumo.dlr.de/docs/',
        '[5] Project experiment records. experiments/v2/main/config.json; data/manifest.json; models/*/config.json and run.json; results/evaluation/test/all_episodes.csv; results/reported_metrics.json. Included in the repository.',
        '[6] Project verification records. reports/final/report_evidence.json; reports/final/software_verification.json; tests/; validate_ui.py; verify_repository.py. Included in the repository.',
    ]
    for ref in refs:
        p(escape(ref))
    p('Documentation references accessed 18 November 2026. Local MIT and third-party notices identify the terms applying to project files and installed dependencies. Dataset descriptions and conclusions concern the synthetic simulation only.')
    STORY.append(Spacer(1, 8))
    doc = Report(OUT/'Adaptive_Crossroads_Project_Report.pdf')
    doc.multiBuild(STORY)
    (OUT/'Adaptive_Crossroads_Project_Report.md').write_text('\n\n'.join(MARKDOWN), encoding='utf-8')
    # Preserve the separately recorded offline checks while updating recomputed statistics.
    old = json.loads((OUT/'report_evidence.json').read_text()) if (OUT/'report_evidence.json').exists() else {}
    FACTS.update({k:v for k,v in old.items() if k.startswith('offline_')})
    (OUT/'report_evidence.json').write_text(json.dumps(FACTS, indent=2), encoding='utf-8')
    print('Built', OUT/'Adaptive_Crossroads_Project_Report.pdf')


if __name__ == '__main__':
    main()
