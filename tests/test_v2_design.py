"""Tests for v2 unconfounded paired data, MDP v2 schemas, versioning guards, and extended metrics."""
from collections import Counter
import hashlib
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from runtime import ROOT
from experiments import get_experiment, validate_checkpoint_schema
from generate_routes import route_xml_v2, APPROACHES, IMBALANCES, LEVELS
from environment import (
    CrossroadsEnv,
    emergency_penalty_v2,
    episode_metrics,
    load_scenarios,
)
from controllers import hybrid_action, preemption_action


def test_v2_dataset_determinism_pairing_and_orthogonality():
    import sumolib
    v1_manifest = json.loads(get_experiment("v1").manifest_file.read_text(encoding="utf-8"))
    exp = get_experiment("v2")
    manifest = json.loads(exp.manifest_file.read_text(encoding="utf-8"))
    assert len(manifest) == 96
    assert len({s["seed"] for s in manifest}) == len(manifest)
    v1_seeds = {s["seed"] for s in v1_manifest}
    v2_bg_seeds = {s["bg_seed"] for s in manifest}
    assert v1_seeds.isdisjoint(v2_bg_seeds)

    net = sumolib.net.readNet(str(exp.network_file))
    for a in APPROACHES:
        assert net.getEdge(a + "_t").getLength() >= 250

    pairs = {}
    for spec in manifest:
        content, _ = route_xml_v2(
            bg_seed=spec["bg_seed"],
            level=spec["level"],
            imbalance=spec["imbalance"],
            emergency=spec["emergency"],
            approach=spec["ambulance_approach"],
            ambulance_depart=spec["ambulance_depart"],
            timing_window=spec["timing_window"],
        )
        assert hashlib.sha256(content).hexdigest() == spec["sha256"]
        assert content == (ROOT / spec["route_file"]).read_bytes()
        root = ET.fromstring(content)
        assert not root.findall(".//param")
        for route in root.findall("route"):
            first, second = route.get("edges").split()
            assert net.getEdge(second) in net.getEdge(first).getOutgoing()
        pairs.setdefault(spec["bg_seed"], {})[spec["emergency"]] = (spec, root)

    assert len(pairs) == 48
    for bg_seed, pair in pairs.items():
        assert set(pair.keys()) == {False, True}
        norm_spec, norm_root = pair[False]
        emerg_spec, emerg_root = pair[True]
        assert norm_spec["split"] == emerg_spec["split"]
        assert norm_spec["level"] == emerg_spec["level"]
        assert norm_spec["imbalance"] == emerg_spec["imbalance"]
        assert emerg_spec["vehicle_count"] == norm_spec["vehicle_count"] + 1
        norm_cars = [v.attrib for v in norm_root.findall("vehicle") if v.get("type") == "regular"]
        emerg_cars = [v.attrib for v in emerg_root.findall("vehicle") if v.get("type") == "regular"]
        assert norm_cars == emerg_cars, f"Regular arrivals differ in pair {bg_seed}"

    for split, expected_pairs in [("train", 24), ("validation", 12), ("test", 12)]:
        emerg = [s for s in manifest if s["split"] == split and s["emergency"]]
        assert len(emerg) == expected_pairs
        assert Counter(s["ambulance_approach"] for s in emerg) == {a: expected_pairs // 4 for a in APPROACHES}
        assert Counter(s["level"] for s in emerg) == {l: expected_pairs // 3 for l in LEVELS}
        assert Counter(s["imbalance"] for s in emerg) == {im: expected_pairs // 3 for im in IMBALANCES}
        # Every approach sees every imbalance mode equally within each split.
        app_imb = Counter((s["ambulance_approach"], s["imbalance"]) for s in emerg)
        for a in APPROACHES:
            for im in IMBALANCES:
                assert app_imb[(a, im)] == expected_pairs // 12


def test_v2_mdp_shapes_rewards_stratification_and_schema_guard(tmp_path):
    exp = get_experiment("v2")
    scenarios = load_scenarios("train", experiment="v2")
    env_std = CrossroadsEnv(
        scenarios, variant="standard", schema_version="v2",
        net_file=exp.network_file, stratified_sampling=True, seed=0,
    )
    try:
        seen = []
        for _ in range(len(scenarios)):
            obs, _ = env_std.reset()
            assert obs.shape == (26,) and env_std.observation_space.contains(obs)
            seen.append(env_std.current_scenario["id"])
        # Stratified epoch covers all scenarios without replacement before repeating.
        assert len(set(seen)) == len(scenarios)
        obs, reward, term, trunc, _ = env_std.step(0)
        assert env_std.observation_space.contains(obs) and np.isfinite(reward)
        assert not term and not trunc
    finally:
        env_std.close()

    env_emg = CrossroadsEnv(
        scenarios, variant="emergency", schema_version="v2",
        net_file=exp.network_file, telemetry=True, seed=0,
    )
    try:
        emg_s = next(s for s in scenarios if s["emergency"] and s["level"] == "low")
        obs, _ = env_emg.reset(options={"scenario": emg_s})
        assert obs.shape == (34,) and env_emg.observation_space.contains(obs)
        # Verify red penalty triggers immediately on approach even when not stopped.
        pen_moving_red = emergency_penalty_v2([{"proximity": 0.5, "wait": 0.0, "stopped": False, "light": "r"}])
        pen_moving_green = emergency_penalty_v2([{"proximity": 0.5, "wait": 0.0, "stopped": False, "light": "G"}])
        assert pen_moving_red > pen_moving_green == 0.0

        class DummyModel:
            def predict(self, ob, deterministic=True):
                return 0, None

        for _ in range(10):
            act = hybrid_action(env_emg, DummyModel(), obs)
            assert env_emg.action_space.contains(act)
            obs, reward, _, _, _ = env_emg.step(act)
            assert env_emg.observation_space.contains(obs) and np.isfinite(reward)
        assert "queue_n" in env_emg.samples[-1] and "wait_n" in env_emg.samples[-1]
    finally:
        env_emg.close()

    # Schema guard rejects v1 checkpoint loaded against v2 schema (and vice versa).
    v1_model_dir = get_experiment("v1").models_dir / "emergency_seed0"
    with pytest.raises(ValueError, match="Observation schema mismatch"):
        validate_checkpoint_schema(v1_model_dir, expected_obs_schema="v2", expected_variant="emergency")
    bad_cfg = tmp_path / "bad_model"
    bad_cfg.mkdir()
    (bad_cfg / "config.json").write_text(json.dumps({
        "variant": "standard", "obs_schema_version": "v2", "obs_dim": 21,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="obs_dim mismatch"):
        validate_checkpoint_schema(bad_cfg, expected_obs_schema="v2", expected_variant="standard")


def test_extended_episode_metrics_computation():
    samples = [{"time": t, "total_queue": 2 if t < 300 else (8 if 350 <= t < 370 else 2)} for t in range(1, 1201)]
    ep = SimpleNamespace(
        current_scenario={
            "vehicle_count": 10, "emergency": True, "id": "pair_test",
            "seed": 30000, "bg_seed": 30000, "level": "medium", "imbalance": "symmetric",
            "asymmetric": False, "ambulance_approach": "n", "ambulance_depart": 280,
        },
        reg_departures={"car1": 10}, reg_arrivals={"car1": 50},
        reg_wait={"car1": 15}, reg_loss={"car1": 20},
        sim_step=1200, ambulance_entry=300, ambulance_clear=350, ambulance_depart_actual=285,
        collisions=0, red_crossings=[], num_teleported_vehicles=0,
        max_approach_wait=15.0, samples=samples,
    )
    m = episode_metrics(ep)
    assert m["ambulance_requested_depart"] == 280
    assert m["ambulance_insertion_delay_seconds"] == 5.0
    assert m["ambulance_scheduled_to_clear_seconds"] == 70.0
    assert m["ambulance_passage_seconds"] == 50.0
    assert m["max_approach_wait_seconds"] == 15.0
    # Baseline pre_arrival_queue=2.0 -> recovery_threshold=2.2; excess for t in [350..369] = 20 * (8 - 2.2) = 116.0
    assert m["post_clearance_excess_queue_area_120s"] == pytest.approx(116.0)


def test_sim_worker_sandbox_injection_and_step():
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable, "-u", "sim_worker.py",
            "--experiment", "v2",
            "--split", "test",
            "--controller", "hybrid",
            "--seed", "0",
            "--scenario", "scenario_30000_normal",
            "--mode", "sandbox",
            "--demand-scale", "1.25",
            "--start-paused",
        ],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = None
        for line in proc.stdout:
            if line.strip().startswith("{"):
                msg = json.loads(line)
                if msg.get("type") == "ready":
                    ready = msg
                    break
        assert ready is not None and ready["mode"] == "sandbox"
        proc.stdin.write(json.dumps({"cmd": "inject_ambulance", "approach": "e"}) + "\n")
        proc.stdin.write(json.dumps({"cmd": "step", "seconds": 15}) + "\n")
        proc.stdin.flush()

        seen_amb = False
        last_t = 0
        for line in proc.stdout:
            if not line.strip().startswith("{"):
                continue
            msg = json.loads(line)
            if msg.get("type") == "frame":
                last_t = msg["time"]
                if any(v[0].startswith("ambulance_inj_") for v in msg.get("vehicles", [])):
                    seen_amb = True
                if last_t >= 15:
                    break
        assert seen_amb and last_t == 15
        proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
        proc.stdin.flush()
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
