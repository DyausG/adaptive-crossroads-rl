"""Seeded synthetic traffic inputs (v1 legacy and v2 unconfounded paired design)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from runtime import ROOT
from experiments import get_experiment, OBS_SCHEMA_V2, REWARD_SCHEMA_V2, SCHEMA_OBS_DIMS
import sumo

APPROACHES = ("n", "s", "e", "w")
OPPOSITE = {"n": "s", "s": "n", "e": "w", "w": "e"}
LEVELS = ("low", "medium", "rush")
IMBALANCES = ("symmetric", "ns_heavy", "ew_heavy")
HORIZON = 1200
DEMAND_END = 900
BASE_RATES = {"low": 0.055, "medium": 0.12, "rush": 0.21}
SIGNAL_CFG = dict(delta_time=5, yellow_time=3, min_green=10, max_green=60, enforce_max_green=True)


def build_network(folder=None):
    folder = Path(folder) if folder else (ROOT / "data/network")
    folder.mkdir(parents=True, exist_ok=True)
    upstream = ROOT / "vendor/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
    binary = str(Path(sumo.SUMO_HOME) / "bin/netconvert.exe")
    prefix = folder / "base"
    commands = [[binary, "-s", str(upstream), "--plain-output-prefix", str(prefix)]]
    subprocess_run_checked(commands[0])
    nodes = ET.parse(folder / "base.nod.xml")
    coords = {"n": (150, 550), "s": (150, -250), "e": (550, 150), "w": (-250, 150), "t": (150, 150)}
    for node in nodes.getroot():
        if node.get("id") in coords:
            x, y = coords[node.get("id")]
            node.set("x", str(x)); node.set("y", str(y))
            node.attrib.pop("shape", None)
    nodes.write(folder / "extended.nod.xml", encoding="utf-8", xml_declaration=True)
    edges = ET.parse(folder / "base.edg.xml")
    for edge in edges.getroot():
        edge.attrib.pop("shape", None)
        edge.attrib.pop("length", None)
        for lane in edge:
            lane.attrib.pop("shape", None); lane.attrib.pop("length", None)
    edges.write(folder / "extended.edg.xml", encoding="utf-8", xml_declaration=True)
    commands.append([binary, "--node-files", str(folder / "extended.nod.xml"),
                     "--edge-files", str(folder / "extended.edg.xml"),
                     "--connection-files", str(folder / "base.con.xml"),
                     "--tllogic-files", str(folder / "base.tll.xml"),
                     "--no-turnarounds", "true", "-o", str(folder / "crossroads.net.xml")])
    subprocess_run_checked(commands[-1])
    net = ET.parse(folder / "crossroads.net.xml").getroot()
    lengths = {edge.get("id"): [float(lane.get("length")) for lane in edge.findall("lane")]
               for edge in net.findall("edge") if edge.get("id") in [a + "_t" for a in APPROACHES]}
    assert len(lengths) == 4 and min(min(v) for v in lengths.values()) >= 250, lengths
    (folder / "adaptation.json").write_text(json.dumps({
        "commands": commands,
        "approach_lengths_m": lengths,
        "source": str(upstream.relative_to(ROOT)),
        "reason": "Upstream approaches are about 142m; extend outer nodes to 400m for a true 250m detection window.",
    }, indent=2), encoding="utf-8")


def subprocess_run_checked(cmd):
    import subprocess
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def ensure_v2_network(exp):
    dest = exp.data_dir / "network"
    src = ROOT / "data/network"
    dest.mkdir(parents=True, exist_ok=True)
    if (src / "crossroads.net.xml").exists():
        for name in ["crossroads.net.xml", "adaptation.json", "base.con.xml", "base.edg.xml",
                     "base.netccfg", "base.nod.xml", "base.tll.xml", "extended.edg.xml", "extended.nod.xml"]:
            if (src / name).exists():
                shutil.copy2(src / name, dest / name)
    else:
        build_network(dest)


def _approach_rate(base_rate, imbalance, origin):
    if imbalance == "symmetric":
        return base_rate
    if imbalance == "ns_heavy":
        return base_rate * (1.45 if origin in ("n", "s") else 0.55)
    if imbalance == "ew_heavy":
        return base_rate * (1.45 if origin in ("e", "w") else 0.55)
    raise ValueError(f"Unknown imbalance: {imbalance!r}")


def route_xml(seed, level, asymmetric, emergency, approach=None, ambulance_depart=None):
    """v1 deterministic route generator (preserved untouched for baseline hash compatibility)."""
    rng = np.random.default_rng(seed)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="regular", vClass="passenger", accel="2.6", decel="4.5", sigma="0.5", length="5", minGap="2.5", maxSpeed="13.9")
    ET.SubElement(root, "vType", id="ambulance", vClass="emergency", guiShape="emergency", color="1,0,0", accel="2.6", decel="4.5", sigma="0", length="5", minGap="2.5", maxSpeed="13.9", speedFactor="1", speedDev="0")
    for origin in APPROACHES:
        for destination in APPROACHES:
            if origin != destination:
                ET.SubElement(root, "route", id=f"{origin}_{destination}", edges=f"{origin}_t t_{destination}")
    base_rate = BASE_RATES[level]
    arrivals = []
    for origin in APPROACHES:
        rate = base_rate * (1.45 if asymmetric and origin in ("n", "s") else 0.55 if asymmetric else 1)
        t = float(rng.exponential(1 / rate))
        index = 0
        destinations = [OPPOSITE[origin]] + [x for x in APPROACHES if x not in (origin, OPPOSITE[origin])]
        while t < DEMAND_END:
            destination = rng.choice(destinations, p=[0.8, 0.1, 0.1])
            arrivals.append((t, f"car_{origin}_{index:04d}", "regular", f"{origin}_{destination}"))
            index += 1
            t += float(rng.exponential(1 / rate))
    if emergency:
        approach = approach or APPROACHES[seed % 4]
        ambulance_depart = int(ambulance_depart if ambulance_depart is not None else rng.integers(240, 421))
        arrivals.append((float(ambulance_depart), "ambulance_0", "ambulance", f"{approach}_{OPPOSITE[approach]}"))
    for depart, vid, kind, route in sorted(arrivals):
        ET.SubElement(root, "vehicle", id=vid, type=kind, route=route, depart=f"{depart:.3f}", departLane="best", departSpeed="max")
    ET.indent(root)
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return content, {"seed": seed, "level": level, "asymmetric": asymmetric, "emergency": emergency,
                     "ambulance_approach": approach if emergency else None,
                     "ambulance_depart": ambulance_depart if emergency else None,
                     "vehicle_count": len(arrivals), "horizon": HORIZON, "demand_end": DEMAND_END}


def route_xml_v2(bg_seed, level, imbalance, emergency, approach=None, ambulance_depart=None, timing_window=None):
    """v2 unconfounded route generator with independent background and ambulance RNG streams."""
    bg_rng = np.random.default_rng(int(bg_seed))
    amb_rng = np.random.default_rng(int(bg_seed) + 1_000_000)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="regular", vClass="passenger", accel="2.6", decel="4.5", sigma="0.5", length="5", minGap="2.5", maxSpeed="13.9")
    ET.SubElement(root, "vType", id="ambulance", vClass="emergency", guiShape="emergency", color="1,0,0", accel="2.6", decel="4.5", sigma="0", length="5", minGap="2.5", maxSpeed="13.9", speedFactor="1", speedDev="0")
    for origin in APPROACHES:
        for destination in APPROACHES:
            if origin != destination:
                ET.SubElement(root, "route", id=f"{origin}_{destination}", edges=f"{origin}_t t_{destination}")
    base_rate = BASE_RATES[level]
    arrivals = []
    for origin in APPROACHES:
        rate = _approach_rate(base_rate, imbalance, origin)
        t = float(bg_rng.exponential(1 / rate))
        index = 0
        destinations = [OPPOSITE[origin]] + [x for x in APPROACHES if x not in (origin, OPPOSITE[origin])]
        while t < DEMAND_END:
            destination = bg_rng.choice(destinations, p=[0.8, 0.1, 0.1])
            arrivals.append((t, f"car_{origin}_{index:04d}", "regular", f"{origin}_{destination}"))
            index += 1
            t += float(bg_rng.exponential(1 / rate))
    if emergency:
        approach = approach or APPROACHES[int(bg_seed) % 4]
        if timing_window == "early":
            lo, hi = 240, 330
        elif timing_window == "late":
            lo, hi = 331, 420
        else:
            lo, hi = 240, 420
        ambulance_depart = int(ambulance_depart if ambulance_depart is not None else amb_rng.integers(lo, hi + 1))
        arrivals.append((float(ambulance_depart), "ambulance_0", "ambulance", f"{approach}_{OPPOSITE[approach]}"))
    for depart, vid, kind, route in sorted(arrivals):
        ET.SubElement(root, "vehicle", id=vid, type=kind, route=route, depart=f"{depart:.3f}", departLane="best", departSpeed="max")
    ET.indent(root)
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    route_seed = int(bg_seed) + (100_000 if emergency else 0)
    return content, {
        "seed": route_seed,
        "bg_seed": int(bg_seed),
        "sumo_seed": int(bg_seed),
        "pair_id": f"pair_{int(bg_seed)}",
        "level": level,
        "imbalance": imbalance,
        "asymmetric": imbalance != "symmetric",
        "emergency": bool(emergency),
        "ambulance_approach": approach if emergency else None,
        "ambulance_depart": ambulance_depart if emergency else None,
        "timing_window": timing_window if emergency else None,
        "vehicle_count": len(arrivals),
        "horizon": HORIZON,
        "demand_end": DEMAND_END,
    }


def generate_v1():
    build_network()
    scenarios = []
    for split, start, count in [("train", 1000, 36), ("validation", 2000, 12), ("test", 3000, 24)]:
        folder = ROOT / "data/routes" / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            seed = start + i
            level = LEVELS[i % 3]
            emergency = i % 2 == 0
            content, spec = route_xml(seed, level, bool((i // 2) % 2), emergency,
                                      approach=APPROACHES[(i // 2) % 4])
            filename = folder / f"scenario_{seed}.rou.xml"
            filename.write_bytes(content)
            spec.update({"split": split, "id": f"scenario_{seed}", "route_file": filename.relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256(content).hexdigest()})
            scenarios.append(spec)
    (ROOT / "data/manifest.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    print(json.dumps({"experiment": "v1", "scenario_count": len(scenarios), "split_counts": {s: sum(x["split"] == s for x in scenarios) for s in ["train", "validation", "test"]}}, indent=2))


def generate_v2():
    exp = get_experiment("v2")
    exp.root.mkdir(parents=True, exist_ok=True)
    ensure_v2_network(exp)
    scenarios = []
    # 24 train background pairs (2 Latin-square blocks), 12 validation pairs, 12 test pairs.
    split_specs = [("train", 10000, 24), ("validation", 20000, 12), ("test", 30000, 12)]
    for split, start, pair_count in split_specs:
        folder = exp.routes_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(pair_count):
            bg_seed = start + i
            block_idx = i // 12
            cell_idx = i % 12
            a_idx = cell_idx // 3
            l_idx = cell_idx % 3
            approach = APPROACHES[a_idx]
            level = LEVELS[l_idx]
            imbalance = IMBALANCES[(a_idx + l_idx + block_idx) % 3]
            timing_window = ("early", "late")[(a_idx + l_idx + block_idx) % 2]
            for emergency in (False, True):
                suffix = "emergency" if emergency else "normal"
                content, spec = route_xml_v2(
                    bg_seed=bg_seed,
                    level=level,
                    imbalance=imbalance,
                    emergency=emergency,
                    approach=approach,
                    timing_window=timing_window,
                )
                filename = folder / f"scenario_{bg_seed}_{suffix}.rou.xml"
                filename.write_bytes(content)
                spec.update({
                    "split": split,
                    "id": f"scenario_{bg_seed}_{suffix}",
                    "route_file": filename.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
                scenarios.append(spec)
    manifest_bytes = json.dumps(scenarios, indent=2).encode("utf-8")
    exp.manifest_file.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    (exp.data_dir / "README.md").write_text(
        "# Crossroads v2 Dataset — Unconfounded Paired Scenarios\n\n"
        "All 96 route files in `experiments/v2/main/data/routes/` are synthetic Poisson simulation inputs "
        "using the extended 389.6m four-way single-intersection network.\n\n"
        "## Design improvements over v1\n\n"
        "- **Matched Normal/Emergency Pairs**: Each background stream (`bg_seed`) generates both a `_normal` "
        "and an `_emergency` scenario using the exact same regular-vehicle arrival RNG stream (`np.random.default_rng(bg_seed)`) "
        "and shared `sumo_seed`, enabling exact causal measurement of ambulance insertion.\n"
        "- **Independent Ambulance RNG Stream**: Ambulance departure time (`240-330s` early or `331-420s` late) is drawn from "
        "`np.random.default_rng(bg_seed + 1_000_000)` so adding an ambulance never perturbs regular vehicle arrivals.\n"
        "- **Balanced Latin-Square Coverage**: Each 12-pair block covers all 4 approaches (`n`, `s`, `e`, `w`) × 3 demand levels "
        "(`low`, `medium`, `rush`) paired across `symmetric`, `ns_heavy` (N/S×1.45, E/W×0.55), and `ew_heavy` (E/W×1.45, N/S×0.55) flows.\n"
        "- **Reserved New Splits**: Train `10000-10023` (24 pairs / 48 routes), Validation `20000-20011` (12 pairs / 24 routes), "
        "Held-out Test `30000-30011` (12 pairs / 24 routes). Disjoint from all v1 seeds.\n",
        encoding="utf-8",
    )

    exp_config = {
        "experiment_id": exp.experiment_id,
        "run_id": exp.run_id,
        "obs_schema_version": OBS_SCHEMA_V2,
        "reward_schema_version": REWARD_SCHEMA_V2,
        "obs_dims": {"standard": SCHEMA_OBS_DIMS[("v2", "standard")], "emergency": SCHEMA_OBS_DIMS[("v2", "emergency")]},
        "controllers": list(exp.controllers),
        "signal": SIGNAL_CFG,
        "detection_m": 250.0,
        "seed_allocation": {
            "train_bg_seeds": "10000-10023 (24 paired = 48 routes)",
            "validation_bg_seeds": "20000-20011 (12 paired = 24 routes)",
            "test_bg_seeds": "30000-30011 (12 paired = 24 routes)",
            "learner_seeds": [0, 1, 2],
        },
        "training_budget": {
            "steps_per_model": exp.default_steps,
            "exploration_fraction": exp.default_exploration_fraction,
            "checkpoint_selection": "Final checkpoint at predeclared 40,000-step budget after validation verification; no test-based selection",
        },
        "predeclared_demo_scenario": exp.default_demo_scenario,
        "manifest_sha256": manifest_sha,
    }
    exp.config_file.write_text(json.dumps(exp_config, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": "v2",
        "scenario_count": len(scenarios),
        "split_counts": {s: sum(x["split"] == s for x in scenarios) for s in ["train", "validation", "test"]},
        "manifest_sha256": manifest_sha,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    if args.experiment == "v1":
        generate_v1()
    else:
        generate_v2()
