"""Simulation-first native desktop demo. Research evidence stays in the repository."""
import argparse
from datetime import datetime
import json
import math
import time
import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as ET
from runtime import ROOT
from experiments import get_experiment
from ui_scenarios import ScenarioConfig, write_scenario
from ui_playback import CONTROLLERS, SimulationSession, controller_flags, select_controller, interpolate

BG, PAPER, INK, MUTED = "#F4F2ED", "#FEFDF9", "#263A38", "#687570"
ACCENT, LINE, RED, GOLD, BLUE = "#166B60", "#DCDDD4", "#D25049", "#BD8536", "#447F9C"
DIRECTIONS = (("n", "North"), ("s", "South"), ("e", "East"), ("w", "West"))


def label(parent, text="", **kw):
    return tk.Label(parent, text=text, bg=kw.pop("bg", PAPER), fg=kw.pop("fg", INK),
                    font=kw.pop("font", ("Segoe UI", 10)), **kw)


class Segments(tk.Frame):
    def __init__(self, parent, variable, choices, command=None, **kw):
        super().__init__(parent, bg=kw.pop("bg", PAPER), **kw)
        self.variable, self.command, self.buttons = variable, command, []
        for value, text in choices:
            button = tk.Button(self, text=text, relief="flat", bd=0, padx=10, pady=7,
                               cursor="hand2", font=("Segoe UI", 9), command=lambda v=value: self.choose(v))
            button.pack(side="left", fill="x", expand=True, padx=(0, 3))
            self.buttons.append((value, button))
        self.variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def choose(self, value):
        self.variable.set(value)
        if self.command:
            self.command()

    def refresh(self):
        for value, button in self.buttons:
            selected = value == self.variable.get()
            button.configure(bg=ACCENT if selected else BG, fg="white" if selected else INK,
                             activebackground=ACCENT, activeforeground="white")

    def enable(self, enabled):
        for _, button in self.buttons:
            button.configure(state="normal" if enabled else "disabled")


class ControllerPicker(tk.Frame):
    def __init__(self, parent, initial, changed):
        super().__init__(parent, bg=PAPER)
        self.selected, self.changed = initial, changed
        self.flags = {c: tk.BooleanVar(value=v) for c, v in controller_flags(initial).items()}
        self.buttons = {}
        for key, text in (("rule", "Rule-based preemption"), ("emergency", "Emergency-aware RL"), ("fixed", "Fixed-time cyclic")):
            button = tk.Checkbutton(self, text=text, variable=self.flags[key], bg=PAPER, fg=INK,
                                   selectcolor=PAPER, activebackground=PAPER, anchor="w", font=("Segoe UI", 10),
                                   bd=0, highlightthickness=0, command=lambda k=key: self.choose(k))
            button.pack(fill="x", pady=1)
            self.buttons[key] = button
        self.active = label(self, fg=ACCENT, font=("Segoe UI", 9, "bold"), anchor="w")
        self.active.pack(fill="x", pady=(5, 0))
        self.refresh()

    def choose(self, key):
        self.selected = select_controller(self.selected, key, self.flags[key].get())
        self.refresh()
        self.changed()

    def refresh(self):
        for key, enabled in controller_flags(self.selected).items():
            self.flags[key].set(enabled)
        self.active.configure(text=CONTROLLERS[self.selected])


class JunctionView(tk.Frame):
    def __init__(self, parent, index):
        super().__init__(parent, bg=PAPER, highlightthickness=1, highlightbackground=LINE)
        self.index = index
        self.heading = label(self, font=("Segoe UI", 12, "bold"), anchor="w")
        self.heading.pack(fill="x", padx=14, pady=(9, 2))
        label(self, "SUMO positions · lamps: right / through / left", fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=14, pady=(0, 6))
        bottom = tk.Frame(self, bg=PAPER)
        bottom.pack(side="bottom", fill="x")
        self.canvas = tk.Canvas(self, bg="#E9EDE4", highlightthickness=0, height=180)
        self.canvas.pack(fill="both", expand=True, padx=1)
        self.count = label(bottom, "Junction view · waiting for a run", fg=MUTED, font=("Segoe UI", 9), anchor="w")
        self.count.pack(fill="x", padx=14, pady=5)
        self.telemetry = label(bottom, "Stopped  —     Active  —     Ambulances  —", anchor="w", font=("Segoe UI", 11))
        self.telemetry.pack(fill="x", padx=14, pady=(0, 6))
        self.detail = label(bottom, fg=MUTED, font=("Segoe UI", 9), anchor="w", justify="left")
        self.frame, self.radius = None, 60
        self.vehicle_items, self.signal_items = {}, []
        self.net = ET.parse(get_experiment("v2").network_file).getroot()
        junction = self.net.find("junction[@id='t']")
        self.center = (float(junction.get("x")), float(junction.get("y")))
        self.canvas.bind("<Configure>", lambda _: self.redraw_static())
        self.bind("<Configure>", self.resize_heading)

    def resize_heading(self, event):
        if event.widget is self:
            self.heading.configure(font=("Segoe UI", 10 if event.width < 400 else 12, "bold"))

    def point(self, x, y):
        c = self.canvas
        scale = min(c.winfo_width(), c.winfo_height()) / (2 * self.radius)
        return c.winfo_width() / 2 + (x - self.center[0]) * scale, c.winfo_height() / 2 - (y - self.center[1]) * scale

    def redraw_static(self):
        c = self.canvas
        c.delete("all")
        self.vehicle_items.clear()
        self.signal_items.clear()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 40 or h < 40:
            return
        scale = min(w, h) / (2 * self.radius)
        # Geometry is read from the actual SUMO network, including turn paths.
        junction = self.net.find("junction[@id='t']")
        shape = [self.point(*map(float, p.split(","))) for p in junction.get("shape").split()]
        c.create_polygon(*[xy for p in shape for xy in p], fill="#697571", outline="")
        lanes = {}
        for edge in self.net.findall("edge"):
            if edge.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                points = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
                lanes[(edge.get("id"), lane.get("index"))] = points
                coords = [xy for p in points for xy in self.point(*p)]
                c.create_line(*coords, fill="#697571", width=3.2 * scale)
                c.create_line(*coords, fill="#C1C7BE", width=1, dash=(3, 12))
        for connection in self.net.findall("connection"):
            if connection.get("tl") != "t":
                continue
            p = lanes[(connection.get("from"), connection.get("fromLane"))][-1]
            x, y = self.point(*p)
            r = max(3., min(5., scale * 2.2))
            item = c.create_oval(x - r, y - r, x + r, y + r, fill=RED, outline=PAPER, width=1.2)
            self.signal_items.append((int(connection.get("linkIndex")), item))
        for x, y, txt, anchor in ((w / 2, 20, "NORTH ↓", "n"), (w / 2, h - 17, "SOUTH ↑", "s"),
                                  (18, h / 2 + 27, "WEST →", "w"), (w - 18, h / 2 - 27, "← EAST", "e")):
            c.create_text(x, y, text=txt, fill=INK, font=("Segoe UI", 9, "bold"), anchor=anchor)
        self.draw(self.frame)

    def draw(self, frame):
        self.frame = frame
        if not frame:
            return
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        scale = min(w, h) / (2 * self.radius)
        visible, current_ids = 0, set()
        for vid, emergency, wx, wy, angle, speed, wait, road in frame.get("vehicles", []):
            x, y = self.point(wx, wy)
            if not (-12 <= x <= w + 12 and -12 <= y <= h + 12):
                continue
            current_ids.add(vid)
            visible += 1
            direction = math.radians(angle)
            dx, dy = math.sin(direction), -math.cos(direction)
            length, width = max(4., 2.5 * scale), max(1.7, 1.0 * scale)
            if emergency:
                length, width = max(6., length), max(3., width)
            points = [coord for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
                      for coord in (x + a * dx * length - b * dy * width, y + a * dy * length + b * dx * width)]
            color = RED if emergency else GOLD if speed < 0.1 else BLUE
            if vid not in self.vehicle_items:
                body = c.create_polygon(points, fill=color, outline=PAPER if emergency else "", width=1.2)
                text = c.create_text(x + 12, y - 14, text="+ AMB", fill=RED,
                                     font=("Segoe UI", 9, "bold"), anchor="w") if emergency else None
                self.vehicle_items[vid] = (body, text)
            body, text = self.vehicle_items[vid]
            c.coords(body, *points)
            c.itemconfigure(body, fill=color)
            if text:
                c.coords(text, x + 12, y - 14)
        for vid in set(self.vehicle_items) - current_ids:
            body, text = self.vehicle_items.pop(vid)
            c.delete(body)
            if text:
                c.delete(text)
        signal = frame.get("sample", {}).get("signal", "r" * 12)
        for index, item in self.signal_items:
            ch = signal[index] if index < len(signal) else "r"
            c.itemconfigure(item, fill=ACCENT if ch in "Gg" else GOLD if ch in "Yy" else RED)
        self.count.configure(text=f"{visible} visible / {len(frame['vehicles'])} active · {2 * self.radius:.0f} m view height")

    def update_telemetry(self, frame, details=False):
        sample = frame.get("sample", {})
        ambulances = sum(bool(v[1]) for v in frame["vehicles"])
        self.telemetry.configure(text=f"Stopped  {sample.get('total_queue', 0)}    Active  {len(frame['vehicles'])}    AMB  {ambulances}",
                                 font=("Segoe UI", 10))
        if details:
            self.detail.configure(text="Queues  " + "   ".join(f"{a.upper()} {sample.get('queue_' + a, 0)}" for a, _ in DIRECTIONS)
                                  + f"\nPending insertion  {sample.get('pending', 0)} · Signal phase {sample.get('green_phase', 0) + 1}")
            self.detail.pack(fill="x", padx=14, pady=(0, 6))
        else:
            self.detail.pack_forget()


class App(tk.Tk):
    def __init__(self, initial_experiment="v2"):
        super().__init__()
        self.title("Crossroads — Adaptive Signals")
        width = min(1400, self.winfo_screenwidth() - 70)
        height = min(920, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(1040, 720)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.session, self.retiring = None, None
        self.restart_pending, self.closing, self.playing, self.single_step = False, False, False, False
        self.cursor, self.animation_started, self.animation_duration = 0, None, .2
        self.frozen_frames, self.frozen_fraction = None, 0.
        self.pending_injection, self.active_config, self.views = None, None, []
        self.config_dirty, self.last_error = False, None
        self.mode, self.level = tk.StringVar(value="single"), tk.StringVar(value="medium")
        self.emergency, self.origin, self.seed = tk.BooleanVar(value=True), tk.StringVar(value="n"), tk.StringVar(value="42")
        self.heavy = {a: tk.BooleanVar(value=False) for a, _ in DIRECTIONS}
        self.speed, self.zoom, self.details = tk.DoubleVar(value=5), tk.DoubleVar(value=60), tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready · configure your junction, then start")
        self.context = tk.StringVar(value="Interactive simulation · v2 trained policies · not a held-out benchmark")
        self.clock = tk.StringVar(value="00:00 / 20:00")
        self.inject_status = tk.StringVar(value="Scheduled ambulance at 05:00 · North")
        self._build()
        self.tick_id = self.after(20, self._tick)

    def _build(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale", background=BG, troughcolor=LINE)
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=28, pady=(14, 10))
        label(header, "CROSSROADS", bg=BG, font=("Segoe UI", 22, "bold")).pack(side="left")
        label(header, "  /  adaptive signals", bg=BG, fg=MUTED, font=("Segoe UI", 13)).pack(side="left", pady=(8, 0))
        self.mode_segments = Segments(header, self.mode, [("single", "Single"), ("compare", "Side-by-side")], self.mode_changed, bg=BG)
        self.mode_segments.pack(side="right")
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=28)
        sidebar = tk.Frame(body, bg=PAPER, width=278, highlightthickness=1, highlightbackground=LINE)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        scroll_canvas = tk.Canvas(sidebar, bg=PAPER, highlightthickness=0)
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=scroll_canvas.yview)
        scroll.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scroll_canvas.configure(yscrollcommand=scroll.set)
        self.config_frame = tk.Frame(scroll_canvas, bg=PAPER, padx=15, pady=14)
        child = scroll_canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        scroll_canvas.bind("<Configure>", lambda e: scroll_canvas.itemconfigure(child, width=e.width))
        self.config_frame.bind("<Configure>", lambda _: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        sidebar.bind("<Enter>", lambda _: self.bind_all("<MouseWheel>", lambda e: scroll_canvas.yview_scroll(-int(e.delta / 120), "units")))
        sidebar.bind("<Leave>", lambda _: self.unbind_all("<MouseWheel>"))
        self.section(self.config_frame, "01   CONTROLLER")
        self.picker_a = ControllerPicker(self.config_frame, "emergency", self.config_changed)
        self.picker_a.pack(fill="x")
        self.picker_b_frame = tk.Frame(self.config_frame, bg=PAPER)
        self.section(self.picker_b_frame, "COMPARE WITH")
        self.picker_b = ControllerPicker(self.picker_b_frame, "fixed", self.config_changed)
        self.picker_b.pack(fill="x")
        self.scenario_frame = tk.Frame(self.config_frame, bg=PAPER)
        self.scenario_frame.pack(fill="x")
        self.section(self.scenario_frame, "02   TRAFFIC")
        self.traffic_segments = Segments(self.scenario_frame, self.level, [("low", "Low"), ("medium", "Medium"), ("rush", "Rush hour")], self.config_changed)
        self.traffic_segments.pack(fill="x")
        label(self.scenario_frame, "Traffic-heavy approaches", anchor="w").pack(fill="x", pady=(13, 3))
        row = tk.Frame(self.scenario_frame, bg=PAPER)
        row.pack(fill="x")
        for a, name in DIRECTIONS:
            tk.Checkbutton(row, text=a.upper(), variable=self.heavy[a], bg=PAPER, fg=INK,
                           activebackground=PAPER, selectcolor=PAPER, font=("Segoe UI", 10), command=self.config_changed).pack(side="left", expand=True)
        label(self.scenario_frame, "None = balanced. Selected directions receive 45% more arrivals.", fg=MUTED,
              font=("Segoe UI", 9), anchor="w", justify="left", wraplength=215).pack(fill="x", pady=5)
        self.section(self.scenario_frame, "03   EMERGENCY")
        self.emergency_button = tk.Checkbutton(self.scenario_frame, text="Emergency vehicle", variable=self.emergency, bg=PAPER,
                       selectcolor=PAPER, activebackground=PAPER, font=("Segoe UI", 11), fg=INK, command=self.emergency_changed)
        self.emergency_button.pack(anchor="w")
        label(self.scenario_frame, "Ambulance origin · arrives at 05:00", fg=MUTED, font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(7, 5))
        self.origin_segments = Segments(self.scenario_frame, self.origin, [(a, a.upper()) for a, _ in DIRECTIONS], self.config_changed)
        self.origin_segments.pack(fill="x")
        self.advanced_button = tk.Button(self.scenario_frame, text="+ Reproducible traffic", bg=PAPER, fg=MUTED, relief="flat",
                                         anchor="w", font=("Segoe UI", 10), command=self.toggle_advanced)
        self.advanced_button.pack(fill="x", pady=(18, 0))
        self.advanced = tk.Frame(self.scenario_frame, bg=PAPER)
        label(self.advanced, "Traffic seed", anchor="w").pack(fill="x")
        tk.Spinbox(self.advanced, from_=0, to=999999, textvariable=self.seed, width=12, font=("Segoe UI", 10), bg=BG, fg=INK).pack(anchor="w", pady=5)
        label(self.advanced, "Same number = same scheduled traffic\nand simulator randomness. Model seed: 0.", fg=MUTED,
              font=("Segoe UI", 9), anchor="w", justify="left").pack(fill="x")
        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)
        label(main, "Simulation", bg=BG, font=("Segoe UI", 16, "bold"), anchor="w").pack(fill="x")
        label(main, textvariable=self.context, bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(4, 12))
        footer = tk.Frame(main, bg=BG)
        # Reserve controls before allocating the expandable viewport, including on 720p.
        footer.pack(side="bottom", fill="x")
        self.view_area = tk.Frame(main, bg=BG)
        self.view_area.pack(fill="both", expand=True)
        self.build_views()
        legend = tk.Frame(footer, bg=BG)
        legend.pack(fill="x", pady=(9, 5))
        for text, color in (("● Moving", BLUE), ("● Stopped", GOLD), ("+ Ambulance", RED)):
            label(legend, text, bg=BG, fg=color, font=("Segoe UI", 9)).pack(side="left", padx=(0, 16))
        tk.Checkbutton(legend, text="Details", variable=self.details, bg=BG, fg=MUTED, activebackground=BG,
                       font=("Segoe UI", 9), command=self.refresh_exact).pack(side="right")
        ttk.Scale(legend, from_=45, to=400, variable=self.zoom, command=self.zoom_changed, length=85).pack(side="right", padx=8)
        label(legend, "Zoom", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right")
        controls = tk.Frame(footer, bg=BG)
        controls.pack(fill="x", pady=(2, 4))
        self.start_button = self.button(controls, "Start simulation", self.start, primary=True)
        self.start_button.pack(side="left", padx=(0, 7))
        self.pause_button = self.button(controls, "Pause", self.pause)
        self.pause_button.pack(side="left", padx=(0, 7))
        self.stop_button = self.button(controls, "Stop", self.stop)
        self.stop_button.pack(side="left", padx=(0, 7))
        self.step_button = self.button(controls, "+1s", self.step_once)
        self.step_button.pack(side="left")
        self.speed_text = label(controls, "5×", bg=BG, fg=ACCENT, width=4)
        self.speed_text.pack(side="right")
        ttk.Scale(controls, from_=1, to=25, variable=self.speed, command=self.speed_changed, length=110).pack(side="right", padx=8)
        label(controls, "Speed", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right")
        timeline = tk.Frame(footer, bg=BG)
        timeline.pack(fill="x", pady=4)
        label(timeline, "Replay", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(0, 7))
        self.timeline = ttk.Scale(timeline, from_=0, to=1, command=self.seek)
        self.timeline.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.timeline.bind("<ButtonPress-1>", lambda _: self.pause(force=True))
        label(timeline, textvariable=self.clock, bg=BG, fg=MUTED, font=("Consolas", 10)).pack(side="right")
        actions = tk.Frame(timeline, bg=BG)
        actions.pack(side="right", padx=(0, 12))
        label(actions, "Inject", bg=BG, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        self.inject_buttons = []
        for a, name in DIRECTIONS:
            btn = self.button(actions, a.upper(), lambda d=a: self.inject(d))
            btn.pack(side="left", padx=(0, 4))
            self.inject_buttons.append(btn)
        label(footer, textvariable=self.inject_status, bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(0, 3))
        label(self, textvariable=self.status, bg=BG, fg=ACCENT, font=("Segoe UI", 10), anchor="w").pack(side="bottom", before=body, fill="x", padx=28, pady=(8, 10))
        self.update_buttons()

    @staticmethod
    def section(parent, text):
        label(parent, text, fg=MUTED, font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", pady=(15, 10))

    @staticmethod
    def button(parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else PAPER,
                         fg="white" if primary else INK, activebackground=ACCENT, activeforeground="white",
                         relief="flat", bd=0, padx=12, pady=8, cursor="hand2", font=("Segoe UI", 10))

    def build_views(self):
        for view in self.views:
            view.destroy()
        self.views = []
        controllers = [self.picker_a.selected]
        if self.mode.get() == "compare":
            controllers.append(self.picker_b.selected)
        for i, controller in enumerate(controllers):
            self.view_area.columnconfigure(i, weight=1, uniform="views")
            view = JunctionView(self.view_area, i)
            view.heading.configure(text=(f"{'AB'[i]}  ·  " if len(controllers) == 2 else "") + CONTROLLERS[controller])
            view.radius = self.zoom.get()
            view.grid(row=0, column=i, sticky="nsew", padx=(0, 8) if len(controllers) == 2 and i == 0 else 0)
            self.views.append(view)
        self.view_area.columnconfigure(1, weight=1 if len(controllers) == 2 else 0)
        self.view_area.columnconfigure(1, uniform="views" if len(controllers) == 2 else "")
        self.view_area.rowconfigure(0, weight=1)

    def mode_changed(self):
        self.stop(clear=True)
        if self.mode.get() == "compare":
            self.picker_b_frame.pack(fill="x", before=self.scenario_frame)
        else:
            self.picker_b_frame.pack_forget()
        self.build_views()
        self.context.set("Same generated route + simulator seed · two controllers, one synchronized clock" if self.mode.get() == "compare"
                         else "Interactive simulation · v2 trained policies · not a held-out benchmark")

    def config_changed(self):
        if self.session:
            self.config_dirty = True
            self.status.set("Configuration changed · Restart applies it to a fresh run")
        else:
            self.build_views()
            self.inject_status.set(f"Scheduled ambulance at 05:00 · {dict(DIRECTIONS)[self.origin.get()]}" if self.emergency.get()
                                   else "Emergency off · no scheduled ambulance or live injection")

    def emergency_changed(self):
        self.origin_segments.enable(self.emergency.get())
        self.config_changed()

    def toggle_advanced(self):
        if self.advanced.winfo_manager():
            self.advanced.pack_forget()
            self.advanced_button.configure(text="+ Reproducible traffic")
        else:
            self.advanced.pack(fill="x", pady=6)
            self.advanced_button.configure(text="− Reproducible traffic")

    def start(self):
        try:
            self.next_config = ScenarioConfig(self.level.get(), tuple(a for a, _ in DIRECTIONS if self.heavy[a].get()),
                                              self.emergency.get(), self.origin.get(), int(self.seed.get()))
            self.next_config.canonical()
        except ValueError as exc:
            self.status.set(f"Check configuration: {exc}")
            return
        self.restart_pending = True
        self.stop(preserve_restart=True)
        if not self.retiring:
            self._launch()

    def _launch(self):
        self.restart_pending = False
        try:
            path, self.spec = write_scenario(self.next_config)
            self.active_config = self.next_config
            self.config_dirty, self.last_error = False, None
            self.run_dir = ROOT / "results/ui_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            controllers = [self.picker_a.selected]
            if self.mode.get() == "compare":
                controllers.append(self.picker_b.selected)
            self.session = SimulationSession(controllers, path, self.run_dir)
            (self.run_dir / "run.json").write_text(json.dumps(dict(scenario=self.spec, controllers=controllers,
                 experiment="v2", model_seed=0, interactive=True), indent=2), encoding="utf-8")
            self.build_views()
            self.cursor, self.animation_started = 0, None
            self.frozen_frames = None
            self.playing, self.single_step, self.pending_injection = True, False, None
            self.timeline.configure(to=1)
            self.status.set("Starting SUMO · loading controller(s)…")
            self.context.set(f"{'Same inputs on both sides · ' if len(controllers) == 2 else ''}{self.active_config.level.title()} traffic · seed {self.active_config.seed} · input {self.spec['sha256'][:8]} · interactive")
            self.inject_status.set("Scheduled ambulance at 05:00 · " + dict(DIRECTIONS)[self.active_config.origin]
                                   if self.active_config.emergency else "Emergency off · no scheduled ambulance or live injection")
        except Exception as exc:
            self.last_error = f"Unable to start: {exc}"
            self.status.set(self.last_error)
            if self.session:
                self.session.close()
                self.retiring, self.session = self.session, None
        self.update_buttons()

    def stop(self, preserve_restart=False, clear=False):
        if not preserve_restart:
            self.restart_pending = False
        self.freeze_animation()
        self.playing, self.animation_started, self.pending_injection = False, None, None
        if self.session:
            self.refresh_exact()
            self.session.close()
            self.retiring, self.session = self.session, None
            self.status.set("Stopping · releasing simulator processes…")
        if clear:
            self.cursor = 0
            self.clock.set("00:00 / 20:00")
            self.timeline.set(0)
        self.update_buttons()

    def pause(self, force=False):
        if not self.session or not self.session.is_ready:
            return
        was_playing = self.playing
        if was_playing:
            self.freeze_animation()
        self.playing = False if force else not self.playing
        self.single_step = False
        if self.playing and self.frozen_frames:
            self.animation_duration = 1 / self.speed.get()
            self.animation_started = time.perf_counter() - self.frozen_fraction * self.animation_duration
            self.frozen_frames = None
        elif not self.playing:
            self.animation_started = None
        self.refresh_exact()
        self.status.set("Playing · synchronized simulation time" if self.playing else "Paused · exact recorded state · resume or scrub the timeline")
        self.update_buttons()

    def step_once(self):
        if not self.session or not self.session.is_ready:
            return
        self.playing, self.animation_started = False, None
        self.frozen_frames = None
        if self.cursor < len(self.session.history) - 1:
            self.cursor += 1
            self.refresh_exact()
        else:
            self.single_step = True
            self.session.step()

    def seek(self, value):
        if getattr(self, "setting_timeline", False) or not self.session or not self.session.history:
            return
        self.playing, self.single_step, self.animation_started = False, False, None
        self.frozen_frames = None
        self.cursor = max(0, min(len(self.session.history) - 1, int(float(value))))
        self.refresh_exact()
        self.status.set("Paused · reviewing recorded states" if self.cursor < len(self.session.history) - 1 else "Paused · at live edge")
        self.update_buttons()

    def refresh_exact(self):
        if self.session and self.session.history:
            self.cursor = min(self.cursor, len(self.session.history) - 1)
            self.show(self.frozen_frames or self.session.history[self.cursor])

    def freeze_animation(self):
        if self.session and self.animation_started is not None and self.cursor + 1 < len(self.session.history):
            self.frozen_fraction = min(1., (time.perf_counter() - self.animation_started) / self.animation_duration)
            self.frozen_frames = [interpolate(a, b, self.frozen_fraction) for a, b in
                                  zip(self.session.history[self.cursor], self.session.history[self.cursor + 1])]

    def show(self, frames):
        for view, frame in zip(self.views, frames):
            view.draw(frame)
            view.update_telemetry(frame, self.details.get())
        if any(frame.get("injected_ambulances", 0) for frame in frames):
            counts = [sum(v[0].startswith("ambulance_inj_") for v in frame["vehicles"]) for frame in frames]
            self.inject_status.set(f"Injected requests: {frames[0]['injected_ambulances']} · active in displayed state: " +
                                   " / ".join(f"{'AB'[i]} {n}" for i, n in enumerate(counts)))
        t = frames[0].get("display_time", frames[0]["time"])
        self.clock.set(f"{int(t) // 60:02}:{int(t) % 60:02} / 20:00")
        self.setting_timeline = True
        self.timeline.set(t)
        self.setting_timeline = False

    def speed_changed(self, _=None):
        self.speed_text.configure(text=f"{self.speed.get():.0f}×")

    def zoom_changed(self, _=None):
        for view in self.views:
            view.radius = self.zoom.get()
            view.redraw_static()

    def inject(self, origin):
        if not self.session or not self.active_config.emergency or not self.session.is_ready:
            return
        live_edge = len(self.session.history) - (2 if self.playing else 1)
        if self.cursor < live_edge or self.pending_injection:
            self.inject_status.set("Return to the live edge before injecting another ambulance")
            return
        self.pending_injection = origin
        self.inject_status.set(f"Queued from {dict(DIRECTIONS)[origin]} · applying at the next shared simulator boundary")

    def update_buttons(self):
        session = self.session
        ready = bool(session and session.is_ready and not session.closed)
        for button in (self.pause_button, self.step_button):
            button.configure(state="normal" if ready else "disabled")
        self.stop_button.configure(state="normal" if session or self.retiring else "disabled")
        self.start_button.configure(text="Restart" if session else "Start simulation", state="disabled" if self.retiring else "normal")
        self.pause_button.configure(text="Pause" if self.playing else "Resume")
        self.timeline.configure(state="normal" if ready else "disabled")
        injection_ok = ready and self.active_config.emergency and not session.finished and self.cursor >= len(session.history) - (2 if self.playing else 1)
        for button in self.inject_buttons:
            button.configure(state="normal" if injection_ok and not self.pending_injection else "disabled")

    def _tick(self):
        self.tick_id = None
        if self.retiring and self.retiring.stopped.is_set():
            self.retiring = None
            self.status.set(self.last_error or "Stopped · simulator processes released")
            if self.closing:
                self.destroy()
                return
            if self.restart_pending:
                self._launch()
        session = self.session
        if session:
            events = session.poll()
            for index, msg in events:
                if msg.get("event") == "injection_requested":
                    self.inject_status.set(f"{'AB'[index]}: requested {msg['vehicle']} at {msg['time']}s · awaiting insertion")
            if session.error:
                error = session.error
                self.stop()
                self.last_error = error
                self.status.set(error)
            elif session.is_ready:
                self.timeline.configure(to=max(1, len(session.history) - 1))
                if self.single_step and not session.in_flight:
                    self.single_step = False
                    self.cursor = len(session.history) - 1
                    self.refresh_exact()
                if self.pending_injection and not session.in_flight:
                    if session.inject(self.pending_injection):
                        self.pending_injection = None
                        self.single_step = not self.playing
                        session.step()
                if self.playing:
                    self.status.set("Configuration changed · Restart applies it to a fresh run" if self.config_dirty else
                                    "Playing · same input, synchronized clock" if len(self.views) == 2 else "Playing · live SUMO simulation")
                    if self.cursor < len(session.history) - 1:
                        now = time.perf_counter()
                        if self.animation_started is None:
                            self.animation_started, self.animation_duration = now, 1 / self.speed.get()
                        fraction = min(1., (now - self.animation_started) / self.animation_duration)
                        self.show([interpolate(a, b, fraction) for a, b in zip(session.history[self.cursor], session.history[self.cursor + 1])])
                        if fraction >= 1:
                            self.cursor += 1
                            self.animation_started = None
                    elif not session.in_flight:
                        self.animation_started = None
                        self.refresh_exact()
                        if not session.finished:
                            session.step()
                        elif len(session.finished) == len(session.controllers):
                            self.playing = False
                            self.status.set("Complete · replay with the timeline, or Restart a fresh simulation")
                else:
                    self.refresh_exact()
            self.update_buttons()
        elif not self.retiring:
            self.update_buttons()
        self.tick_id = self.after(16, self._tick)

    def destroy(self):
        if getattr(self, "tick_id", None):
            self.after_cancel(self.tick_id)
            self.tick_id = None
        super().destroy()

    def shutdown(self):
        self.closing = True
        self.stop()
        if not self.retiring:
            self.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["v2"], default="v2", help="UI uses v2; research CLIs retain v1/v2")
    args = parser.parse_args()
    App(args.experiment).mainloop()


if __name__ == "__main__":
    main()
