"""Reproducible interactive inputs. Never writes into the benchmark datasets."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from runtime import ROOT
from generate_routes import APPROACHES, OPPOSITE, BASE_RATES, HORIZON, DEMAND_END


@dataclass(frozen=True)
class ScenarioConfig:
    level: str = "medium"
    heavy: tuple[str, ...] = ()
    emergency: bool = True
    origin: str = "n"
    seed: int = 42

    def canonical(self):
        if self.level not in BASE_RATES or self.origin not in APPROACHES:
            raise ValueError("Invalid traffic level or ambulance origin")
        if not set(self.heavy) <= set(APPROACHES) or not 0 <= self.seed <= 999999:
            raise ValueError("Invalid approach or traffic seed (0–999999)")
        return dict(level=self.level, heavy=sorted(set(self.heavy)), emergency=bool(self.emergency),
                    origin=self.origin if self.emergency else None, seed=int(self.seed), generator=1)


def scenario_xml(config):
    cfg = config.canonical()
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="regular", vClass="passenger", accel="2.6", decel="4.5",
                  sigma="0.5", length="5", minGap="2.5", maxSpeed="13.9")
    ET.SubElement(root, "vType", id="ambulance", vClass="emergency", guiShape="emergency",
                  color="1,0,0", accel="2.6", decel="4.5", sigma="0", length="5", minGap="2.5",
                  maxSpeed="13.9", speedFactor="1", speedDev="0")
    for a in APPROACHES:
        for b in APPROACHES:
            if a != b:
                ET.SubElement(root, "route", id=f"{a}_{b}", edges=f"{a}_t t_{b}")
    arrivals = []
    # Independent streams keep the other approaches unchanged when one is made heavy.
    for i, a in enumerate(APPROACHES):
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], i]))
        rate = BASE_RATES[cfg["level"]] * (1.45 if a in cfg["heavy"] else 1.0)
        t, index = float(rng.exponential(1 / rate)), 0
        destinations = [OPPOSITE[a]] + [b for b in APPROACHES if b not in (a, OPPOSITE[a])]
        while t < DEMAND_END:
            dest = rng.choice(destinations, p=[0.8, 0.1, 0.1])
            arrivals.append((t, f"car_{a}_{index:04d}", "regular", f"{a}_{dest}"))
            index += 1
            t += float(rng.exponential(1 / rate))
    if cfg["emergency"]:
        a = cfg["origin"]
        arrivals.append((300., "ambulance_0", "ambulance", f"{a}_{OPPOSITE[a]}"))
    for depart, vid, kind, route in sorted(arrivals):
        ET.SubElement(root, "vehicle", id=vid, type=kind, route=route, depart=f"{depart:.3f}",
                      departLane="best", departSpeed="max")
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), len(arrivals)


def write_scenario(config, folder=None):
    cfg = config.canonical()
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(folder) if folder else ROOT / "results/ui_runs/scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    xml, count = scenario_xml(config)
    route = folder / f"{key}.rou.xml"
    spec = dict(id=f"interactive_{key}", seed=cfg["seed"], bg_seed=cfg["seed"], sumo_seed=cfg["seed"],
                level=cfg["level"], asymmetric=bool(cfg["heavy"]), imbalance="+".join(cfg["heavy"]) or "symmetric",
                emergency=cfg["emergency"], ambulance_approach=cfg["origin"],
                ambulance_depart=300 if cfg["emergency"] else None, vehicle_count=count,
                horizon=HORIZON, demand_end=DEMAND_END, route_file=str(route.resolve()),
                sha256=hashlib.sha256(xml).hexdigest(), interactive_config=cfg, split="interactive")
    spec_path = folder / f"{key}.json"
    # Immutable content addressed inputs; refuse silent corruption on a later restart.
    for path, content in [(route, xml), (spec_path, json.dumps(spec, indent=2).encode())]:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Interactive input differs from its recorded content: {path}")
        if not path.exists():
            path.write_bytes(content)
    return spec_path, spec
