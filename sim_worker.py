"""Isolated single-simulation child worker streaming 1s SUMO state and vehicle coordinates to app.py."""
import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path
from runtime import ROOT  # noqa: F401 - establishes libsumo before sumo_rl
from experiments import (
    get_experiment,
    checkpoint_variant_for_controller,
    validate_checkpoint_schema,
)
import torch
from stable_baselines3 import DQN
from generate_routes import APPROACHES, OPPOSITE, HORIZON
from environment import CrossroadsEnv, load_scenarios, episode_metrics, recovery_metrics
from controllers import fixed_action, preemption_action, hybrid_action


def _stdin_reader(cmd_queue: queue.Queue):
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            cmd_queue.put(json.loads(text))
        except json.JSONDecodeError:
            parts = text.split()
            obj = {"cmd": parts[0]}
            if len(parts) > 1:
                obj["arg"] = parts[1]
            cmd_queue.put(obj)
    cmd_queue.put({"cmd": "stop"})


def _choose_action(controller, env, model, obs):
    if controller == "fixed":
        return fixed_action(env)
    if controller == "rule":
        return preemption_action(env)
    if controller == "hybrid":
        return hybrid_action(env, model, obs)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def _emit(payload: dict):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _frame_payload(env, injected_count=0):
    sumo = env.sumo
    vehs = []
    for vid in sumo.vehicle.getIDList():
        x, y = sumo.vehicle.getPosition(vid)
        angle = sumo.vehicle.getAngle(vid)
        speed = sumo.vehicle.getSpeed(vid)
        wait = sumo.vehicle.getWaitingTime(vid)
        road = sumo.vehicle.getRoadID(vid)
        vehs.append([
            vid,
            1 if vid.startswith("ambulance") else 0,
            round(float(x), 1),
            round(float(y), 1),
            round(float(angle), 1),
            round(float(speed), 2),
            round(float(wait), 1),
            road,
        ])
    sample = env.samples[-1] if env.samples else {}
    rec = (
        recovery_metrics(env.samples, env.ambulance_entry, env.ambulance_clear, int(env.sim_step))
        if env.ambulance_entry is not None
        else None
    )
    return {
        "type": "frame",
        "time": int(env.sim_step),
        "sample": sample,
        "ambulance_entry": env.ambulance_entry,
        "ambulance_clear": env.ambulance_clear,
        "ambulance_depart_actual": env.ambulance_depart_actual,
        "ambulance_red_crossings": len(env.red_crossings),
        "injected_ambulances": injected_count,
        "recovery": rec,
        "vehicles": vehs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--controller", choices=["fixed", "rule", "standard", "emergency", "hybrid"], default="emergency")
    parser.add_argument("--seed", type=int, default=0)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scenario")
    inputs.add_argument("--scenario-file", type=Path, help="Recorded interactive scenario JSON; benchmark inputs stay unchanged")
    parser.add_argument("--mode", choices=["benchmark", "sandbox"], default="benchmark")
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=5.0, help="Sim seconds per wall second; 0 means max speed.")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()

    exp = get_experiment(args.experiment)
    if args.controller not in exp.controllers:
        raise ValueError(f"Controller {args.controller!r} not valid for {exp.experiment_id!r}")
    if args.scenario_file:
        if args.mode != "sandbox":
            raise ValueError("Interactive scenarios must use sandbox mode")
        scenarios = [json.loads(args.scenario_file.read_text(encoding="utf-8"))]
        route = ROOT / scenarios[0]["route_file"]
        if hashlib.sha256(route.read_bytes()).hexdigest() != scenarios[0]["sha256"]:
            raise ValueError("Interactive route hash does not match its manifest")
    else:
        scenarios = [s for s in load_scenarios(args.split, experiment=exp.experiment_id) if s["id"] == args.scenario]
    if len(scenarios) != 1:
        raise ValueError(f"Scenario {args.scenario!r} not found in {exp.experiment_id}/{args.split}")
    spec = scenarios[0]

    torch.set_num_threads(1)
    model = None
    ckpt_variant = checkpoint_variant_for_controller(args.controller)
    if ckpt_variant is not None:
        model_folder = exp.models_dir / f"{ckpt_variant}_seed{args.seed}"
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant)
        model = DQN.load(model_folder / "final.zip", device="cpu")
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant, loaded_model=model)

    env_variant = "emergency" if args.controller == "emergency" else "standard"
    env = CrossroadsEnv(
        scenarios,
        variant=env_variant,
        telemetry=True,
        gui=False,
        schema_version=exp.obs_schema_version,
        net_file=exp.network_file,
    )
    if args.mode == "sandbox" and abs(args.demand_scale - 1.0) > 1e-6:
        env.additional_sumo_cmd += f" --scale {max(0.25, min(2.0, args.demand_scale)):.2f}"

    cmd_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_stdin_reader, args=(cmd_queue,), daemon=True)
    reader.start()

    paused = bool(args.start_paused)
    speed_hz = float(args.speed)
    step_budget = 0
    fast_forward_until = None
    injected_count = 0

    try:
        obs, _ = env.reset(options={"scenario": spec})
        jx, jy = env.sumo.junction.getPosition("t")
        _emit({
            "type": "ready",
            "experiment": exp.experiment_id,
            "mode": args.mode,
            "controller": args.controller,
            "seed": args.seed if ckpt_variant else None,
            "scenario": spec["id"],
            "level": spec["level"],
            "imbalance": spec.get("imbalance", "ns_heavy" if spec["asymmetric"] else "symmetric"),
            "emergency": bool(spec["emergency"]),
            "ambulance_approach": spec.get("ambulance_approach"),
            "ambulance_depart": spec.get("ambulance_depart"),
            "demand_scale": args.demand_scale if args.mode == "sandbox" else 1.0,
            "junction": [float(jx), float(jy)],
            "horizon": HORIZON,
            "route_sha256": hashlib.sha256((ROOT / spec["route_file"]).read_bytes()).hexdigest(),
            "traffic_seed": spec.get("sumo_seed", spec["seed"]),
        })
        # An exact t=0 state lets the UI synchronize startup and interpolation.
        _emit(_frame_payload(env))

        ts = env.traffic_signals[env.ts_ids[0]]
        while env.sim_step < HORIZON:
            # Drain pending UI commands.
            while True:
                try:
                    msg = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                cmd = msg.get("cmd")
                if cmd == "stop":
                    return
                if cmd == "pause":
                    paused = True
                    fast_forward_until = None
                elif cmd == "resume":
                    paused = False
                    fast_forward_until = None
                elif cmd == "step":
                    step_budget += int(msg.get("seconds", msg.get("arg", 1)))
                    fast_forward_until = None
                elif cmd == "speed":
                    speed_hz = float(msg.get("hz", msg.get("arg", 5.0)))
                elif cmd == "fast_forward":
                    fast_forward_until = min(HORIZON, int(msg.get("target_time", msg.get("arg", 0))))
                elif cmd == "inject_ambulance":
                    if args.mode == "sandbox":
                        approach = str(msg.get("approach", msg.get("arg", "n"))).lower()
                        if approach in APPROACHES:
                            injected_count += 1
                            vid = f"ambulance_inj_{injected_count}"
                            route_id = f"{approach}_{OPPOSITE[approach]}"
                            env.sumo.vehicle.add(
                                vehID=vid,
                                routeID=route_id,
                                typeID="ambulance",
                                depart="now",
                                departLane="best",
                                departSpeed="max",
                            )
                            _emit({"type": "event", "event": "injection_requested", "vehicle": vid,
                                   "approach": approach, "time": int(env.sim_step),
                                   "message": f"Requested {vid} from {approach.upper()} at t={int(env.sim_step)}s; awaiting SUMO insertion"})

            in_ff = fast_forward_until is not None and env.sim_step < fast_forward_until
            if paused and step_budget <= 0 and not in_ff:
                time.sleep(0.03)
                continue

            tick_start = time.perf_counter()
            if ts.time_to_act:
                action = _choose_action(args.controller, env, model, obs)
                env._apply_actions(action)
            env._sumo_step()
            for ts_id in env.ts_ids:
                env.traffic_signals[ts_id].update()
            if ts.time_to_act:
                obs = env._compute_observations()[env.ts_ids[0]]
                env._compute_rewards()

            if step_budget > 0:
                step_budget -= 1
            if fast_forward_until is not None and env.sim_step >= fast_forward_until:
                fast_forward_until = None
                in_ff = False

            # Emit every frame during normal playback/step, or every 5s during fast-forward.
            if not in_ff or int(env.sim_step) % 5 == 0:
                _emit(_frame_payload(env, injected_count=injected_count))

            if not paused and not in_ff and step_budget == 0 and speed_hz > 0:
                delay = max(0.0, (1.0 / speed_hz) - (time.perf_counter() - tick_start))
                if delay > 0:
                    time.sleep(delay)

        metrics = episode_metrics(env)
        metrics["mode"] = args.mode
        metrics["injected_ambulances"] = injected_count
        metrics["evaluation_eligible"] = args.mode == "benchmark" and not args.scenario_file
        if injected_count:
            metrics["ambulance_metric_scope"] = "First observed ambulance across scheduled and injected vehicles; not a benchmark result"
        _emit({"type": "done", "metrics": metrics})
    finally:
        env.close()


if __name__ == "__main__":
    main()
