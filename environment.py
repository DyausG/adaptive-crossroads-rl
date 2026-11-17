"""Lean SUMO-RL adaptation with v1/v2 ambulance features, fairness signals, and auditable telemetry."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from gymnasium import spaces
from runtime import ROOT
from experiments import get_experiment
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction, DefaultObservationFunction
from generate_routes import APPROACHES, HORIZON

DETECTION_M = 250.0
SIGNAL = dict(delta_time=5, yellow_time=3, min_green=10, max_green=60, enforce_max_green=True)
APPROACH_EDGES = {a + "_t" for a in APPROACHES}


def emergency_snapshot(sumo, vehicles=None):
    """Incoming ambulances only, using actual distance to their next stop line."""
    records = []
    for vid in (vehicles if vehicles is not None else sumo.vehicle.getIDList()):
        if not vid.startswith("ambulance"):
            continue
        road = sumo.vehicle.getRoadID(vid)
        if road not in APPROACH_EDGES:
            continue
        next_tls = [tls for tls in sumo.vehicle.getNextTLS(vid) if tls[0] == "t"]
        if not next_tls:
            continue
        _, link, distance, light = next_tls[0]
        if 0 <= distance <= DETECTION_M:
            records.append({"id": vid, "approach": road[0], "distance": float(distance),
                            "proximity": float(np.clip(1 - distance / DETECTION_M, 0, 1)),
                            "wait": float(sumo.vehicle.getWaitingTime(vid)),
                            "stopped": sumo.vehicle.getSpeed(vid) < 0.1,
                            "link": link, "light": light})
    return records


def ambulance_features(records):
    values = []
    for approach in APPROACHES:
        group = [r for r in records if r["approach"] == approach]
        values.extend([float(bool(group)), max((r["proximity"] for r in group), default=0.0)])
    return np.asarray(values, dtype=np.float32)


def approach_wait_features(sumo, cap_seconds=120.0):
    """Normalized maximum regular-vehicle waiting time on each incoming approach."""
    waits = []
    for approach in APPROACHES:
        edge_id = f"{approach}_t"
        max_wait = 0.0
        for vid in sumo.edge.getLastStepVehicleIDs(edge_id):
            if vid.startswith("ambulance"):
                continue
            w = float(sumo.vehicle.getWaitingTime(vid))
            if w > max_wait:
                max_wait = w
        waits.append(float(np.clip(max_wait / cap_seconds, 0.0, 1.0)))
    return np.asarray(waits, dtype=np.float32)


def phase_elapsed_feature(ts):
    """Normalized elapsed duration in the current phase relative to max_green."""
    denom = float(max(1, getattr(ts, "max_green", 60)))
    return float(np.clip(getattr(ts, "time_since_last_phase_change", 0) / denom, 0.0, 1.0))


class EmergencyObservation(ObservationFunction):
    """v1 emergency observation (29 dims): default 21 + 8 ambulance features."""
    def __init__(self, ts):
        super().__init__(ts)
        self.base = DefaultObservationFunction(ts)

    def __call__(self):
        return np.concatenate([self.base(), ambulance_features(emergency_snapshot(self.ts.sumo))]).astype(np.float32)

    def observation_space(self):
        n = self.base.observation_space().shape[0] + 8
        return spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)


class StandardObservationV2(ObservationFunction):
    """v2 standard observation (26 dims): default 21 + 4 approach max-wait + 1 phase elapsed."""
    def __init__(self, ts):
        super().__init__(ts)
        self.base = DefaultObservationFunction(ts)

    def __call__(self):
        return np.concatenate([
            self.base(),
            approach_wait_features(self.ts.sumo),
            np.asarray([phase_elapsed_feature(self.ts)], dtype=np.float32),
        ]).astype(np.float32)

    def observation_space(self):
        n = self.base.observation_space().shape[0] + 5
        return spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)


class EmergencyObservationV2(ObservationFunction):
    """v2 emergency observation (34 dims): StandardObservationV2 (26) + 8 ambulance features."""
    def __init__(self, ts):
        super().__init__(ts)
        self.base_v2 = StandardObservationV2(ts)

    def __call__(self):
        return np.concatenate([self.base_v2(), ambulance_features(emergency_snapshot(self.ts.sumo))]).astype(np.float32)

    def observation_space(self):
        n = self.base_v2.observation_space().shape[0] + 8
        return spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)


def emergency_penalty(records):
    # v1 per 5-second decision: continuous stopped age plus a separate stop penalty.
    return sum(0.02 * min(r["wait"], 120.0) + 0.5 * int(r["stopped"]) for r in records)


def emergency_reward(ts):
    return float(ts._diff_waiting_time_reward() - emergency_penalty(emergency_snapshot(ts.sumo)))


def standard_reward_v2(ts):
    """v2 standard reward: diff-waiting-time minus normalized per-approach starvation penalty."""
    waits = approach_wait_features(ts.sumo)
    return float(ts._diff_waiting_time_reward() - 0.15 * float(np.sum(waits)))


def emergency_penalty_v2(records):
    """v2 emergency penalty: immediate red/yellow approach impedance + stronger stop/wait penalty."""
    total = 0.0
    for r in records:
        if r["light"] not in ("G", "g"):
            total += 1.5 * (1.0 + float(r["proximity"]))
        total += 2.0 * int(r["stopped"]) + 0.04 * min(float(r["wait"]), 120.0)
    return float(total)


def emergency_reward_v2(ts):
    return float(standard_reward_v2(ts) - emergency_penalty_v2(emergency_snapshot(ts.sumo)))


def load_scenarios(split, experiment="v1", manifest_path=None):
    path = Path(manifest_path) if manifest_path else get_experiment(experiment).manifest_file
    return [x for x in json.loads(path.read_text(encoding="utf-8")) if x["split"] == split]


class CrossroadsEnv(SumoEnvironment):
    def __init__(self, scenarios, variant="standard", seed=0, telemetry=False, gui=False,
                 schema_version="v1", net_file=None, stratified_sampling=False):
        self.scenarios = list(scenarios)
        self.variant = variant
        self.telemetry_enabled = telemetry
        self.schema_version = schema_version
        self.stratified_sampling = stratified_sampling
        self.scenario_rng = np.random.default_rng(seed)
        self._epoch_indices = []
        self.current_scenario = self.scenarios[0]
        self.clear_telemetry()

        if schema_version == "v2":
            obs_cls = EmergencyObservationV2 if variant == "emergency" else StandardObservationV2
            reward_fn = emergency_reward_v2 if variant == "emergency" else standard_reward_v2
        elif schema_version == "v1":
            obs_cls = EmergencyObservation if variant == "emergency" else DefaultObservationFunction
            reward_fn = emergency_reward if variant == "emergency" else "diff-waiting-time"
        else:
            raise ValueError(f"Unknown schema_version: {schema_version!r}")

        resolved_net = str(net_file) if net_file else str(ROOT / "data/network/crossroads.net.xml")
        super().__init__(
            net_file=resolved_net,
            route_file=str(ROOT / self.current_scenario["route_file"]),
            single_agent=True, use_gui=gui, num_seconds=HORIZON,
            sumo_seed=seed, reward_fn=reward_fn,
            observation_class=obs_cls,
            sumo_warnings=False,
            additional_sumo_cmd="--no-step-log true --duration-log.disable true --collision.action warn",
            **SIGNAL,
        )

    def clear_telemetry(self):
        self.samples = []
        self.reg_wait = defaultdict(float)
        self.reg_loss = defaultdict(float)
        self.reg_departures = {}
        self.reg_arrivals = {}
        self.ambulance_entry = None
        self.ambulance_clear = None
        self.ambulance_depart_actual = None
        self.red_crossings = []
        self.signal_crossings = []
        self.collisions = 0
        self.max_approach_wait = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.scenario_rng = np.random.default_rng(seed)
            self._epoch_indices = []
        if options and "scenario" in options:
            self.current_scenario = options["scenario"]
        elif self.stratified_sampling:
            if not self._epoch_indices:
                self._epoch_indices = self.scenario_rng.permutation(len(self.scenarios)).tolist()
            self.current_scenario = self.scenarios[self._epoch_indices.pop(0)]
        else:
            self.current_scenario = self.scenarios[int(self.scenario_rng.integers(len(self.scenarios)))]
        self._route = str(ROOT / self.current_scenario["route_file"])
        self.clear_telemetry()
        # Separate fixed simulator seed per route (or shared bg_seed across matched pairs) from learner seed.
        sumo_seed = int(self.current_scenario.get("sumo_seed", self.current_scenario["seed"]))
        return super().reset(seed=sumo_seed)

    def _compute_info(self):
        info = super()._compute_info()
        info["scenario_id"] = self.current_scenario["id"]
        return info

    def _sumo_step(self):
        if not self.telemetry_enabled:
            return super()._sumo_step()
        before = {}
        for vid in self.sumo.vehicle.getIDList():
            if vid.startswith("ambulance"):
                tls = self.sumo.vehicle.getNextTLS(vid)
                before[vid] = (self.sumo.vehicle.getRoadID(vid), tls[0][3] if tls else None)
        super()._sumo_step()
        t = self.sim_step
        ids = self.sumo.vehicle.getIDList()
        self.collisions += self.sumo.simulation.getCollidingVehiclesNumber()
        for vid in self.sumo.simulation.getDepartedIDList():
            if vid.startswith("ambulance"):
                if self.ambulance_depart_actual is None:
                    self.ambulance_depart_actual = t
            else:
                self.reg_departures[vid] = t
        for vid in self.sumo.simulation.getArrivedIDList():
            if not vid.startswith("ambulance"):
                self.reg_arrivals[vid] = t
        records = emergency_snapshot(self.sumo, ids)
        if records and self.ambulance_entry is None:
            self.ambulance_entry = t
        total_queue = 0
        regular_stopped = 0
        approach_queues = {a: 0 for a in APPROACHES}
        approach_waits = {a: 0.0 for a in APPROACHES}
        for vid in ids:
            speed = self.sumo.vehicle.getSpeed(vid)
            stopped = int(speed < 0.1)
            total_queue += stopped
            road = self.sumo.vehicle.getRoadID(vid)
            if vid.startswith("ambulance"):
                if road.startswith("t_") and self.ambulance_entry is not None and self.ambulance_clear is None:
                    self.ambulance_clear = t
                if vid in before and before[vid][0] in APPROACH_EDGES and (road.startswith(":t") or road.startswith("t_")):
                    crossing = {"time": t, "vehicle": vid, "pre_step_signal": before[vid][1]}
                    self.signal_crossings.append(crossing)
                    if before[vid][1] in ("r", "R"):
                        self.red_crossings.append(crossing)
            else:
                self.reg_wait[vid] += stopped
                regular_stopped += stopped
                self.reg_loss[vid] = self.sumo.vehicle.getTimeLoss(vid)
                if road in APPROACH_EDGES:
                    app = road[0]
                    approach_queues[app] += stopped
                    w = float(self.sumo.vehicle.getWaitingTime(vid))
                    if w > approach_waits[app]:
                        approach_waits[app] = w
        self.max_approach_wait = max(self.max_approach_wait, max(approach_waits.values()))
        features = ambulance_features(records)
        ts = self.traffic_signals.get(self.ts_ids[0]) if self.ts_ids else None
        row = {
            "time": t,
            "total_queue": total_queue,
            "regular_stopped": regular_stopped,
            "regular_wait_total": sum(self.reg_wait.values()),
            "ambulance_detected": len(records),
            "ambulance_distance": min((r["distance"] for r in records), default=None),
            "ambulance_wait": max((r["wait"] for r in records), default=0.0),
            "ambulance_stopped_on_red": int(any(r["stopped"] and r["light"] in ("r", "R") for r in records)),
            "signal": self.sumo.trafficlight.getRedYellowGreenState("t"),
            "green_phase": int(ts.green_phase) if ts is not None else 0,
            "is_yellow": int(ts.is_yellow) if ts is not None else 0,
            "time_since_phase_change": int(ts.time_since_last_phase_change) if ts is not None else 0,
            "pending": len(self.sumo.simulation.getPendingVehicles()),
        }
        for a in APPROACHES:
            row[f"queue_{a}"] = approach_queues[a]
            row[f"wait_{a}"] = approach_waits[a]
        for i, approach in enumerate(APPROACHES):
            row[f"ambulance_{approach}_presence"] = float(features[2 * i])
            row[f"ambulance_{approach}_proximity"] = float(features[2 * i + 1])
        self.samples.append(row)


def recovery_metrics(samples, entry, clear, horizon=HORIZON):
    result = {"pre_arrival_queue": None, "recovery_threshold": None,
              "queue_recovery_seconds": None, "recovery_status": "not_applicable",
              "recovery_observed_seconds": None}
    if entry is None:
        result["recovery_status"] = "no_detection"
        return result
    baseline = [r["total_queue"] for r in samples if entry - 60 <= r["time"] < entry]
    mean = float(np.mean(baseline)) if baseline else 0.0
    threshold = max(1.0, 1.1 * mean)
    result.update(pre_arrival_queue=mean, recovery_threshold=threshold)
    if clear is None:
        result["recovery_status"] = "ambulance_not_cleared"
        return result
    candidate = None
    for row in samples:
        if row["time"] < clear:
            continue
        if row["total_queue"] <= threshold:
            if candidate is None:
                candidate = row["time"]
            if row["time"] - candidate >= 14:
                result.update(queue_recovery_seconds=float(candidate - clear), recovery_status="recovered",
                              recovery_observed_seconds=float(candidate - clear))
                return result
        else:
            candidate = None
    result.update(recovery_status="right_censored", recovery_observed_seconds=float(horizon - clear))
    return result


def episode_metrics(env):
    s = env.current_scenario
    regular_scheduled = s["vehicle_count"] - int(s["emergency"])
    departed = len(env.reg_departures)
    completed_waits = [env.reg_wait[v] for v in env.reg_arrivals]
    end = env.sim_step
    if not s["emergency"]:
        passage_status = "not_applicable"
    elif env.ambulance_entry is None:
        passage_status = "not_detected"
    elif env.ambulance_clear is None:
        passage_status = "right_censored"
    else:
        passage_status = "completed"

    req_depart = s.get("ambulance_depart") if s["emergency"] else None
    insertion_delay = (
        float(env.ambulance_depart_actual - req_depart)
        if (s["emergency"] and req_depart is not None and env.ambulance_depart_actual is not None)
        else None
    )
    scheduled_to_clear = (
        float(env.ambulance_clear - req_depart)
        if (passage_status == "completed" and req_depart is not None)
        else None
    )

    result = {
        "scenario": s["id"],
        "route_seed": s["seed"],
        "bg_seed": s.get("bg_seed", s["seed"]),
        "level": s["level"],
        "asymmetric": s["asymmetric"],
        "imbalance": s.get("imbalance", "ns_heavy" if s["asymmetric"] else "symmetric"),
        "emergency": s["emergency"],
        "ambulance_approach": s["ambulance_approach"],
        "horizon": end,
        "ambulance_requested_depart": req_depart,
        "ambulance_entry": env.ambulance_entry,
        "ambulance_clear": env.ambulance_clear,
        "ambulance_depart_actual": env.ambulance_depart_actual,
        "ambulance_insertion_delay_seconds": insertion_delay,
        "ambulance_scheduled_to_clear_seconds": scheduled_to_clear,
        "ambulance_passage_seconds": env.ambulance_clear - env.ambulance_entry if passage_status == "completed" else None,
        "ambulance_passage_status": passage_status,
        "ambulance_observed_seconds": (env.ambulance_clear or end) - env.ambulance_entry if env.ambulance_entry is not None else None,
        "regular_wait_mean_completed": float(np.mean(completed_waits)) if completed_waits else None,
        "regular_wait_p95_completed": float(np.percentile(completed_waits, 95)) if completed_waits else None,
        "regular_wait_mean_inserted": sum(env.reg_wait.values()) / departed if departed else None,
        "regular_time_loss_mean_inserted": sum(env.reg_loss.values()) / departed if departed else None,
        "max_approach_wait_seconds": getattr(env, "max_approach_wait", None),
        "regular_scheduled": regular_scheduled,
        "regular_departed": departed,
        "regular_completed": len(env.reg_arrivals),
        "regular_unfinished": departed - len(env.reg_arrivals),
        "regular_not_inserted": regular_scheduled - departed,
        "colliding_vehicle_events": env.collisions,
        "ambulance_red_crossings": len(env.red_crossings),
        "teleported_vehicles": env.num_teleported_vehicles,
        "final_queue": env.samples[-1]["total_queue"] if env.samples else None,
    }
    result["pending_vehicle_seconds"] = sum(r.get("pending", 0) for r in env.samples)
    result["stopped_plus_pending_seconds_per_requested_vehicle"] = (
        sum(r["total_queue"] + r.get("pending", 0) for r in env.samples) / s["vehicle_count"]
    )
    rec = recovery_metrics(env.samples, env.ambulance_entry, env.ambulance_clear, end) if s["emergency"] else recovery_metrics([], None, None, end)
    result.update(rec)
    if not s["emergency"]:
        result["recovery_status"] = "not_applicable"
    if s["emergency"] and env.ambulance_clear is not None and result["recovery_threshold"] is not None:
        thresh = float(result["recovery_threshold"])
        window_end = min(end, env.ambulance_clear + 120)
        result["post_clearance_excess_queue_area_120s"] = float(
            sum(max(0.0, r["total_queue"] - thresh) for r in env.samples if env.ambulance_clear <= r["time"] < window_end)
        )
    else:
        result["post_clearance_excess_queue_area_120s"] = None
    return result
