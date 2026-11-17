"""Exercise actual Tk controls and real SUMO workers; save separate UI evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from PIL import ImageGrab
from app import App
from runtime import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshots', action='store_true', help='Capture only the test application window (Windows)')
    parser.add_argument('--quick', action='store_true', help='Repeat UI/layout checks through 420s; omit the already-covered full horizon')
    args = parser.parse_args()
    folder = ROOT / 'results/ui_runs/verification'
    folder.mkdir(parents=True, exist_ok=True)
    app = App()
    errors, checks = [], []
    app.report_callback_exception = lambda *error: errors.append(str(error))

    def check(name, condition):
        assert condition, name
        checks.append(name)
        print('PASS', name, flush=True)

    def wait(condition, seconds=60):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.update()
            assert not errors, errors
            if condition():
                return
            time.sleep(.005)
        raise AssertionError(f'Timeout; status: {app.status.get()}')

    def screenshot(name):
        if args.screenshots:
            app.update()
            ImageGrab.grab(window=app.winfo_id()).save(folder / name)

    try:
        app.geometry('1280x850')
        app.update()
        # Exercise the real mutually exclusive widgets, including the all-off state.
        for key in ('fixed', 'rule', 'emergency', 'emergency'):
            app.picker_a.buttons[key].invoke()
        check('all switches off is standard DQN', app.picker_a.selected == 'standard')
        app.emergency_button.invoke()
        check('emergency off disables ambulance origin', all(b['state'] == 'disabled' for _, b in app.origin_segments.buttons))
        app.level.set('low')
        app.heavy['s'].set(True)
        app.start_button.invoke()
        wait(lambda: app.session and app.session.is_ready)
        session = app.session
        wait(lambda: app.cursor >= 12)
        app.pause_button.invoke()
        frozen = deepcopy([v.frame for v in app.views])
        wait(lambda: not session.in_flight)
        for _ in range(6):
            app.update()
            time.sleep(.02)
        check('pause freezes rendered positions and time', [v.frame for v in app.views] == frozen)
        check('emergency off disables injection', all(b['state'] == 'disabled' for b in app.inject_buttons))
        app.seek('3')
        check('seek displays exact historical state', app.views[0].frame == session.history[3][0])
        original = deepcopy(session.history)
        app.step_button.invoke()
        check('step advances one recorded second', app.cursor == 4)
        check('seek and rendering never mutate evidence', session.history == original)
        screenshot('single.png')
        app.stop_button.invoke()
        wait(lambda: app.session is None and app.retiring is None)
        check('Stop reaps single worker', all(p.poll() == 0 for p in session.processes))
        app.mode_segments.choose('compare')
        app.emergency_button.invoke()
        app.origin_segments.choose('w')
        app.picker_a.buttons['emergency'].invoke()
        app.picker_b.buttons['rule'].invoke()
        app.level.set('rush')
        app.heavy['n'].set(True)
        app.seed.set('99')
        app.speed.set(25)
        app.start_button.invoke()
        wait(lambda: app.session and app.session.is_ready)
        session = app.session
        wait(lambda: app.cursor >= 20)
        app.pause(force=True)
        wait(lambda: not session.in_flight)
        app.seek(str(len(session.history) - 1))
        check('paired startup shares route hash and seed', len({(r['route_sha256'], r['traffic_seed']) for r in session.ready.values()}) == 1)
        check('comparison uses selected controllers', session.controllers == ('emergency', 'rule'))
        check('composable inputs reach SUMO', {r['imbalance'] for r in session.ready.values()} == {'n+s'} and app.spec['ambulance_approach'] == 'w')
        for origin, button in zip(('n', 's', 'e', 'w'), app.inject_buttons):
            before = len(session.history)
            vehicle_id = f"ambulance_inj_{session.history[-1][0]['injected_ambulances'] + 1}"
            button.invoke()
            wait(lambda: len(session.history) > before and app.pending_injection is None and not app.single_step)
            def inserted():
                return all(any(v[0] == vehicle_id and v[7] == origin + '_t' for v in frame['vehicles']) for frame in session.history[-1])
            # SUMO can queue a departure if another vehicle occupies the insertion lane.
            # Advance real states instead of forcing or fabricating an insertion.
            for _ in range(20):
                if inserted():
                    break
                count = len(session.history)
                app.step_button.invoke()
                wait(lambda: len(session.history) > count and not app.single_step)
            check(f'paired {origin.upper()} injection enters actual network', inserted())
        # Generate real subsequent states without waiting for a presentation clock.
        while session.history[-1][0]['time'] < 420:
            session.step()
            wait(lambda: not session.in_flight, 15)
        app.seek('365')
        app.details.set(True)
        app.refresh_exact()
        screenshot('comparison.png')
        check('all comparison frames synchronized', all(len({f['time'] for f in frames}) == 1 for frames in session.history))
        check('scheduled ambulance truly departs from West', all(any(v[0] == 'ambulance_0' and v[7] == 'w_t' for v in frame[i]['vehicles']) for i in range(2) for frame in [session.history[301]]))
        check('paused seek presents equal timestamps', len({v.frame['time'] for v in app.views}) == 1)
        app.geometry('1040x720')
        app.update()
        check('minimum window exposes essential controls', all(w.winfo_rooty() + w.winfo_height() <= app.winfo_rooty() + app.winfo_height() for w in [app.start_button, app.stop_button, *app.inject_buttons]))
        screenshot('minimum_window.png')
        # Verify the natural episode end and process exits as well as early Stop.
        if not args.quick:
            while session.history[-1][0]['time'] < 1200:
                session.step()
                wait(lambda: not session.in_flight, 15)
            wait(lambda: len(session.finished) == 2)
            check('both real runs complete at 1200 seconds', {f['time'] for f in session.history[-1]} == {1200})
        # Restart while active must finish retiring old processes before replacements.
        old_processes = list(session.processes)
        app.start_button.invoke()
        wait(lambda: app.session and app.session is not session and app.session.is_ready)
        check('Restart replaces workers without overlap', all(p.poll() == 0 for p in old_processes) and app.session.history[0][0]['time'] == 0)
        replacement = app.session
        app.stop_button.invoke()
        wait(lambda: app.retiring is None)
        check('repeat Stop cleans both workers', all(p.poll() == 0 for p in replacement.processes))
        # Early cancellation is also an actual UI action.
        app.start_button.invoke()
        startup = app.session
        app.stop_button.invoke()
        wait(lambda: app.retiring is None)
        check('Stop during startup has no orphan workers', all(p.poll() is not None for p in startup.processes))
        check('no Tk callback errors', not errors)
        output = 'desktop_layout_checks.json' if args.quick else 'desktop_checks.json'
        (folder / output).write_text(json.dumps(dict(passed=True, checks=checks, comparison_run=str(session.output_dir),
             full_horizon=not args.quick, screenshots=args.screenshots,
             note='Programmatic real Tk widget invocation; real SUMO processes. No benchmark metrics changed.'), indent=2), encoding='utf-8')
    finally:
        app.shutdown()
        try:
            wait(lambda: not app.winfo_exists(), 15)
        except Exception:
            pass


if __name__ == '__main__':
    main()
