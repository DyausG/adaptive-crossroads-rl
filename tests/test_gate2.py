import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from runtime import ROOT
from generate_routes import route_xml, APPROACHES
from environment import (CrossroadsEnv, ambulance_features, emergency_penalty,
                         load_scenarios, recovery_metrics, episode_metrics)
from controllers import fixed_action, preemption_action


def test_generated_dataset_determinism_connectivity_partitions():
    import sumolib
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    net = sumolib.net.readNet(str(ROOT / "data/network/crossroads.net.xml"))
    assert len(set(s["seed"] for s in manifest)) == len(manifest)
    assert {s["split"] for s in manifest} == {"train", "validation", "test"}
    for spec in manifest:
        content, _ = route_xml(spec["seed"], spec["level"], spec["asymmetric"], spec["emergency"],
                               spec["ambulance_approach"], spec["ambulance_depart"])
        assert hashlib.sha256(content).hexdigest() == spec["sha256"]
        assert content == (ROOT / spec["route_file"]).read_bytes()
        root = ET.fromstring(content)
        assert not root.findall(".//param")  # No bluelight or safety override parameters.
        for route in root.findall("route"):
            first, second = route.get("edges").split()
            assert net.getEdge(second) in net.getEdge(first).getOutgoing()
        for a in APPROACHES:
            assert net.getEdge(a + "_t").getLength() >= 250


def test_features_mapping_and_reward():
    assert np.array_equal(ambulance_features([]), np.zeros(8))
    records = [{"approach": "e", "proximity": 0.6, "wait": 10, "stopped": True}]
    assert np.allclose(ambulance_features(records), [0, 0, 0, 0, 1, .6, 0, 0])
    assert emergency_penalty(records) == pytest.approx(.7)
    assert emergency_penalty([]) == 0


def test_recovery_edge_cases():
    assert recovery_metrics([], None, None)["recovery_status"] == "no_detection"
    samples = [{"time": t, "total_queue": 0} for t in range(1, 101)]
    result = recovery_metrics(samples, 61, 70, 100)
    assert result["recovery_threshold"] == 1
    assert result["queue_recovery_seconds"] == 0
    samples = [{"time": t, "total_queue": 0 if t < 61 else 10} for t in range(1, 101)]
    result = recovery_metrics(samples, 61, 70, 100)
    assert result["recovery_status"] == "right_censored"
    assert result["queue_recovery_seconds"] is None
    assert result["recovery_observed_seconds"] == 30
    assert recovery_metrics(samples, 61, None)["recovery_status"] == "ambulance_not_cleared"


def phase_segments(samples):
    segments = []
    for row in samples:
        if segments and row["signal"] == segments[-1]["signal"]:
            segments[-1]["duration"] += 1
        else:
            segments.append({"signal": row["signal"], "duration": 1})
    return segments


def test_integration_all_approaches_signal_safety_and_boundaries(tmp_path):
    scenarios = load_scenarios("train")
    selected = [next(s for s in scenarios if s["emergency"] and s["ambulance_approach"] == a) for a in APPROACHES]
    results = []
    env = CrossroadsEnv(selected, variant="emergency", telemetry=True)
    try:
        for scenario in selected:
            obs, _ = env.reset(options={"scenario": scenario})
            assert obs.shape == (29,) and env.observation_space.contains(obs)
            assert env.ambulance_entry is None and not env.samples
            done = False
            detected = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(fixed_action(env))
                assert env.observation_space.contains(obs) and np.isfinite(reward)
                assert not terminated
                assert info["step"] <= 1200
                if obs[-8:].sum() > 0:
                    detected = True
                    expected = APPROACHES.index(scenario["ambulance_approach"])
                    assert obs[-8 + 2 * expected] == 1
                    assert sum(obs[-8::2]) == 1
                    for vid in env.sumo.vehicle.getIDList():
                        if vid.startswith("ambulance"):
                            assert env.sumo.vehicle.getSpeedMode(vid) == 31
                            assert env.sumo.vehicle.getParameter(vid, "has.bluelight.device").lower() == "false"
                done = truncated
            assert detected
            assert env.sim_step == 1200 and len(env.samples) == 1200
            assert env.signal_crossings and not env.red_crossings
            segments = phase_segments(env.samples)
            for segment in segments[1:-1]:
                if "y" in segment["signal"]:
                    assert segment["duration"] == 3, segment
                else:
                    assert segment["duration"] >= 10, segment
            metrics = episode_metrics(env)
            assert metrics["ambulance_passage_status"] == "completed"
            results.append({"scenario": scenario["id"], "approach": scenario["ambulance_approach"],
                            "signal_crossings": env.signal_crossings,
                            "stopped_on_red_seconds": sum(x["ambulance_stopped_on_red"] for x in env.samples),
                            "metrics": metrics})
        # Verify the ordinary learner has neither additional features nor reward function.
    finally:
        env.close()
    env = CrossroadsEnv(selected[:1], variant="standard")
    try:
        obs, _ = env.reset()
        assert obs.shape == (21,) and env.observation_space.contains(obs)
        assert env.reward_fn["t"] == "diff-waiting-time"
    finally:
        env.close()
    # Test runs must not overwrite the archived submission's reproducibility evidence.
    (tmp_path / "integration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def test_rule_preemption_uses_legal_transition_path():
    scenarios = [s for s in load_scenarios("train") if s["emergency"]][:1]
    env = CrossroadsEnv(scenarios, telemetry=True)
    try:
        env.reset()
        while env.sim_step < 1200:
            action = preemption_action(env)
            assert env.action_space.contains(action)
            env.step(action)
        assert env.ambulance_clear is not None
        assert not env.red_crossings
        for segment in phase_segments(env.samples)[1:-1]:
            if "y" in segment["signal"]:
                assert segment["duration"] == 3
            else:
                assert segment["duration"] >= 10
    finally:
        env.close()
