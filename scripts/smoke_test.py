"""Gate 1: two reproducible 500-simulator-step upstream episodes."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gate1"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["LIBSUMO_AS_TRACI"] = "1"
import sumo
os.environ["SUMO_HOME"] = sumo.SUMO_HOME
os.environ["PATH"] = str(Path(sumo.SUMO_HOME) / "bin") + os.pathsep + os.environ["PATH"]
import numpy as np
import pandas as pd
import matplotlib
import stable_baselines3
import traci  # Import adapter first; importing libsumo first creates a circular alias.
import libsumo
from sumo_rl import SumoEnvironment


class CountedEnvironment(SumoEnvironment):
    def _sumo_step(self):
        super()._sumo_step()
        self.actual_steps += 1
        self.trace.append({
            "t": self.sim_step,
            "signal": self.sumo.trafficlight.getRedYellowGreenState(self.ts_ids[0]),
            "vehicles": [(v, self.sumo.vehicle.getLaneID(v),
                          round(self.sumo.vehicle.getLanePosition(v), 5),
                          round(self.sumo.vehicle.getSpeed(v), 5))
                         for v in sorted(self.sumo.vehicle.getIDList())],
        })


def main():
    started = time.perf_counter()
    versions = {name: metadata.version(name) for name in
                ["eclipse-sumo", "libsumo", "sumo-rl", "stable-baselines3", "pandas", "matplotlib", "torch", "gymnasium"]}
    net = ROOT / "vendor/sumo-rl/sumo_rl/nets/2way-single-intersection"
    env = CountedEnvironment(
        net_file=str(net / "single-intersection.net.xml"),
        route_file=str(net / "single-intersection-vhvh.rou.xml"),
        out_csv_name=str(OUT / "upstream"), single_agent=True,
        use_gui=False, num_seconds=500, sumo_seed=123,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log true --duration-log.disable true",
    )
    episodes = []
    try:
        for episode in range(2):
            env.actual_steps = 0
            env.trace = []
            obs, info = env.reset(seed=123)
            initial_obs = obs.tolist()
            action_count = 0
            episode_start = time.perf_counter()
            done = False
            while not done:
                action = (action_count // 3) % env.action_space.n
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                done = terminated or truncated
                action_count += 1
            elapsed = time.perf_counter() - episode_start
            env.save_csv(env.out_csv_name, env.episode)
            assert env.actual_steps == 500, env.actual_steps
            assert env.sim_step == 500, env.sim_step
            trace_bytes = json.dumps(env.trace, sort_keys=True).encode()
            (OUT / f"trace_{episode}.json").write_bytes(trace_bytes)
            episodes.append({"episode": episode, "simulator_steps": env.actual_steps,
                             "agent_steps": action_count, "initial_observation": initial_obs,
                             "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
                             "elapsed_seconds": elapsed, "simulator_steps_per_second": 500 / elapsed})
    finally:
        env.close()
    csv_files = sorted(OUT.glob("upstream*.csv"))
    csv_rows = {p.name: len(pd.read_csv(p)) for p in csv_files}
    assert len(csv_files) >= 2 and all(x > 0 for x in csv_rows.values()), csv_rows
    assert episodes[0]["trace_sha256"] == episodes[1]["trace_sha256"], "Reset not reproducible"
    assert not libsumo.isLoaded(), "Simulation left open"
    result = {"pass": True, "python": sys.version, "executable": sys.executable,
              "SUMO_HOME": os.environ["SUMO_HOME"], "LIBSUMO_AS_TRACI": os.environ["LIBSUMO_AS_TRACI"],
              "versions": versions, "sumo_version": subprocess.check_output(
                  [str(Path(sumo.SUMO_HOME) / "bin/sumo.exe"), "--version"], text=True),
              "episodes": episodes, "csv_rows": csv_rows,
              "elapsed_seconds": time.perf_counter() - started}
    (OUT / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
