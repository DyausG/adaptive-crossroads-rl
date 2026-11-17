"""Shared controller semantics, state interpolation, and isolated worker lifecycle."""
from copy import deepcopy
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from runtime import ROOT

CONTROLLERS = {"fixed": "Fixed-time cyclic", "rule": "Rule-based preemption",
               "standard": "Standard RL · DQN", "emergency": "Emergency-aware RL · DQN"}


def controller_flags(selected):
    if selected not in CONTROLLERS:
        raise ValueError(selected)
    return {key: key == selected for key in ("fixed", "rule", "emergency")}


def select_controller(current, clicked, enabled):
    controller_flags(current)
    if clicked not in ("fixed", "rule", "emergency"):
        raise ValueError(clicked)
    return clicked if enabled else ("standard" if current == clicked else current)


def interpolate(previous, following, fraction):
    """Only coordinates/heading interpolate; discrete events change at the true boundary.

    New IDs appear and departed IDs disappear at following.time, never early. A seek,
    pause or restart renders an exact frame rather than blending unrelated snapshots.
    """
    if not previous or fraction >= 1 or following["time"] != previous["time"] + 1:
        return deepcopy(following)
    a = max(0., min(1., fraction))
    result = deepcopy(previous)
    result["display_time"] = previous["time"] + a
    nxt = {v[0]: v for v in following["vehicles"]}
    for vehicle in result["vehicles"]:
        other = nxt.get(vehicle[0])
        if other is None:
            continue
        vehicle[2] += (other[2] - vehicle[2]) * a
        vehicle[3] += (other[3] - vehicle[3]) * a
        vehicle[4] = (vehicle[4] + ((other[4] - vehicle[4] + 180) % 360 - 180) * a) % 360
    return result


class SimulationSession:
    """One subprocess per libsumo instance. A barrier accepts only equal-time frames."""
    def __init__(self, controllers, scenario_file, output_dir):
        if not controllers or len(controllers) > 2 or any(c not in CONTROLLERS for c in controllers):
            raise ValueError("Choose one or two supported controllers")
        self.controllers = tuple(controllers)
        self.scenario_file = Path(scenario_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.messages = queue.Queue()
        self.processes, self.ready, self.pending, self.latest = [], {}, {}, {}
        self.readers = []
        self.history = []
        self.in_flight = False
        self.closed = False
        self.stopped = threading.Event()
        self.error = None
        self.started_at = time.monotonic()
        self.requested_at = None
        self.finished = set()
        try:
            for i, controller in enumerate(controllers):
                cmd = [sys.executable, "-u", str(ROOT / "sim_worker.py"), "--experiment", "v2",
                       "--controller", controller, "--seed", "0", "--scenario-file", str(scenario_file),
                       "--mode", "sandbox", "--speed", "0", "--start-paused"]
                proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.processes.append(proc)
                for target in (self._read, self._stderr):
                    thread = threading.Thread(target=target, args=(i, proc), daemon=True)
                    self.readers.append(thread)
                    thread.start()
        except Exception:
            self.close()
            raise

    def _stderr(self, index, proc):
        with (self.output_dir / f"worker_{index}.log").open("w", encoding="utf-8") as out:
            for line in proc.stderr:
                out.write(line)
        proc.stderr.close()

    def _read(self, index, proc):
        with (self.output_dir / f"trace_{index}.jsonl").open("w", encoding="utf-8") as out:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue  # SUMO may write its own banner to stdout.
                if not isinstance(msg, dict):
                    continue
                out.write(json.dumps(msg) + "\n")
                self.messages.put((index, msg))
        proc.stdout.close()
        self.messages.put((index, {"type": "exit", "code": proc.wait()}))

    @property
    def is_ready(self):
        return len(self.ready) == len(self.controllers) and bool(self.history)

    def send(self, message):
        if self.closed:
            return
        for proc in self.processes:
            try:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.error = f"Worker connection closed: {exc}"

    def step(self):
        if not self.is_ready or self.in_flight or self.closed or self.finished:
            return False
        self.in_flight = True
        self.requested_at = time.monotonic()
        self.send({"cmd": "step", "seconds": 1})
        return True

    def inject(self, origin):
        if origin not in ("n", "s", "e", "w") or not self.is_ready or self.in_flight or self.finished:
            return False
        self.send({"cmd": "inject_ambulance", "approach": origin})
        return True

    def poll(self):
        events = []
        while True:
            try:
                index, msg = self.messages.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            events.append((index, msg))
            if kind == "ready":
                self.ready[index] = msg
                if len(self.ready) == len(self.controllers):
                    signatures = {(m["scenario"], m["route_sha256"], m["traffic_seed"]) for m in self.ready.values()}
                    if len(signatures) != 1:
                        self.error = "Comparison inputs do not match"
            elif kind == "frame":
                self.pending[index] = msg
                if len(self.pending) == len(self.controllers):
                    times = {m["time"] for m in self.pending.values()}
                    expected = self.history[-1][0]["time"] + 1 if self.history else 0
                    if times != {expected}:
                        self.error = f"Playback barrier mismatch: {times}, expected {expected}"
                    else:
                        self.latest = dict(self.pending)
                        self.history.append([self.latest[i] for i in range(len(self.controllers))])
                    self.pending.clear()
                    self.in_flight = False
            elif kind == "done":
                self.finished.add(index)
            elif kind == "exit" and not self.closed and index not in self.finished:
                self.error = f"Worker {index + 1} exited unexpectedly ({msg['code']}). See worker log."
        now = time.monotonic()
        if not self.closed and not self.is_ready and now - self.started_at > 90:
            self.error = "Simulator startup timed out. See worker log."
        if not self.closed and self.in_flight and now - self.requested_at > 30:
            self.error = "Simulator step timed out. See worker log."
        return events

    def close(self):
        if self.closed:
            return
        self.send({"cmd": "stop"})
        self.closed = True
        threading.Thread(target=self._reap, daemon=True).start()

    def _reap(self):
        for proc in self.processes:
            try:
                proc.stdin.close()
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for thread in self.readers:
            thread.join(timeout=2)
        self.stopped.set()
