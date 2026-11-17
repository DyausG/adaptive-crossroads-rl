"""UI inputs, real SUMO barriers, interpolation, and subprocess cleanup."""
from copy import deepcopy
import itertools
import json
import time
import xml.etree.ElementTree as ET
import pytest
from ui_scenarios import ScenarioConfig, scenario_xml, write_scenario
from ui_playback import CONTROLLERS, controller_flags, select_controller, interpolate, SimulationSession


APPROACHES = ('n', 's', 'e', 'w')
HEAVY_SETS = [tuple(a for a, enabled in zip(APPROACHES, flags) if enabled)
              for flags in itertools.product((False, True), repeat=4)]


@pytest.mark.parametrize('level,heavy,origin', list(itertools.product(('low', 'medium', 'rush'), HEAVY_SETS, (None, *APPROACHES))))
def test_every_visible_scenario(level, heavy, origin):
    cfg = ScenarioConfig(level, heavy, origin is not None, origin or 'n', 42)
    content, count = scenario_xml(cfg)
    root = ET.fromstring(content)
    vehicles = root.findall('vehicle')
    assert count == len(vehicles) and count > 0
    assert scenario_xml(cfg)[0] == content
    assert [float(v.get('depart')) for v in vehicles] == sorted(float(v.get('depart')) for v in vehicles)
    routes = {r.get('id') for r in root.findall('route')}
    assert all(v.get('route') in routes for v in vehicles)
    ambulance = [v for v in vehicles if v.get('type') == 'ambulance']
    assert len(ambulance) == int(origin is not None)
    if origin:
        assert ambulance[0].get('route').startswith(origin + '_')
        assert ambulance[0].get('depart') == '300.000'


def test_reproducibility_and_independent_background(tmp_path):
    configs = [ScenarioConfig(emergency=False), ScenarioConfig(origin='w'), ScenarioConfig(origin='s')]
    backgrounds = []
    for cfg in configs:
        content, _ = scenario_xml(cfg)
        backgrounds.append([ET.tostring(v) for v in ET.fromstring(content).findall('vehicle') if v.get('type') == 'regular'])
    assert backgrounds[0] == backgrounds[1] == backgrounds[2]
    normal, _ = scenario_xml(ScenarioConfig())
    other, _ = scenario_xml(ScenarioConfig(seed=43))
    assert normal != other
    p, spec = write_scenario(configs[0], tmp_path)
    assert write_scenario(configs[0], tmp_path) == (p, spec)
    route = __import__('pathlib').Path(spec['route_file'])
    route.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='differs'):
        write_scenario(configs[0], tmp_path)


@pytest.mark.parametrize('current,clicked,enabled', list(itertools.product(CONTROLLERS, ('fixed', 'rule', 'emergency'), (False, True))))
def test_controller_exclusivity(current, clicked, enabled):
    chosen = select_controller(current, clicked, enabled)
    flags = controller_flags(chosen)
    assert sum(flags.values()) <= 1
    if enabled:
        assert chosen == clicked
    if not any(flags.values()):
        assert chosen == 'standard'


def test_interpolation_events_identity_seek_and_immutability():
    a = dict(time=10, sample={'signal': 'r'}, vehicles=[['car', 0, 0., 0., 350., 3., 0, 'n_t'],
                                                     ['gone', 0, 4., 4., 0., 1., 0, 't_s']])
    b = dict(time=11, sample={'signal': 'G'}, vehicles=[['car', 0, 10., 6., 10., 9., 0, ':t_0'],
                                                     ['new', 1, 8., 8., 0., 4., 0, 'n_t']])
    originals = deepcopy((a, b))
    mid = interpolate(a, b, .5)
    assert mid['vehicles'][0][2:5] == [5., 3., 0.]
    assert mid['sample']['signal'] == 'r' and [v[0] for v in mid['vehicles']] == ['car', 'gone']
    assert interpolate(a, b, 1) == b
    distant = dict(b, time=300)
    assert interpolate(a, distant, .5) == distant
    assert (a, b) == originals


def wait_for(session, condition, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session.poll()
        assert not session.error, session.error
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for simulator')


RUNS = [(c,) for c in CONTROLLERS] + list(itertools.product(CONTROLLERS, repeat=2))


@pytest.mark.parametrize('controllers', RUNS)
def test_actual_controllers_and_comparison_lifecycle(controllers, tmp_path):
    path, spec = write_scenario(ScenarioConfig('rush', ('n', 'e'), True, 'w', 73), tmp_path / 'scenarios')
    session = SimulationSession(controllers, path, tmp_path / 'run')
    normal_run = False
    try:
        wait_for(session, lambda: session.is_ready)
        normal_run = True
        assert {m['route_sha256'] for m in session.ready.values()} == {spec['sha256']}
        assert {m['traffic_seed'] for m in session.ready.values()} == {73}
        assert [m['controller'] for _, m in sorted(session.ready.items())] == list(controllers)
        # All four manual origins are real SUMO insertions at shared state boundaries.
        for step, origin in enumerate(APPROACHES, 1):
            assert session.inject(origin)
            assert session.step()
            assert not session.step()  # Cannot advance one worker ahead of another.
            wait_for(session, lambda: not session.in_flight, 10)
            assert {f['time'] for f in session.history[-1]} == {step}
            for frame in session.history[-1]:
                injected = [v for v in frame['vehicles'] if v[0] == f'ambulance_inj_{step}']
                assert injected and injected[0][7] == origin + '_t'
        assert len(session.history) == 5
        if len(controllers) == 2 and controllers[0] == controllers[1]:
            assert all(a == b for a, b in session.history)
        # Paused worker doesn't move without a grant.
        time.sleep(.1)
        session.poll()
        assert len(session.history) == 5
    finally:
        session.close()
        assert session.stopped.wait(10)
        assert all(p.poll() is not None for p in session.processes)
        if normal_run:
            assert all(p.poll() == 0 for p in session.processes)


def test_cancellation_during_startup_and_failure(tmp_path):
    path, _ = write_scenario(ScenarioConfig(), tmp_path / 'scenarios')
    session = SimulationSession(['emergency', 'standard'], path, tmp_path / 'cancel')
    session.close()
    assert session.stopped.wait(12)
    assert all(p.poll() is not None for p in session.processes)
    broken = SimulationSession(['fixed', 'rule'], tmp_path / 'missing.json', tmp_path / 'broken')
    try:
        deadline = time.monotonic() + 45
        while not broken.error and time.monotonic() < deadline:
            broken.poll()
            time.sleep(.02)
        assert broken.error and 'exited' in broken.error
    finally:
        broken.close()
        assert broken.stopped.wait(10)


def test_barrier_rejects_mismatched_frames():
    # A protocol regression must fail closed rather than compare unequal clocks.
    import queue
    fake = SimulationSession.__new__(SimulationSession)
    fake.controllers = ('fixed', 'rule')
    fake.messages, fake.ready, fake.pending, fake.latest = queue.Queue(), {}, {}, {}
    fake.history, fake.finished, fake.error, fake.closed = [], set(), None, True
    fake.in_flight = True
    fake.messages.put((0, dict(type='frame', time=0, vehicles=[])))
    fake.messages.put((1, dict(type='frame', time=1, vehicles=[])))
    fake.poll()
    assert 'mismatch' in fake.error and not fake.history
