"""All policies request phases through the same SUMO-RL timing constraints."""
import numpy as np
from environment import emergency_snapshot


def fixed_action(env):
    # Four movement groups: through/right and protected turn groups.
    t = env.sim_step % 80
    return 0 if t < 25 else 1 if t < 40 else 2 if t < 65 else 3


def preemption_action(env):
    ts = env.traffic_signals[env.ts_ids[0]]
    records = emergency_snapshot(env.sumo)
    if not records:
        return fixed_action(env)
    nearest = min(records, key=lambda r: r["distance"])
    link = nearest["link"]
    candidates = [i for i, phase in enumerate(ts.green_phases) if phase.state[link] in ("G", "g")]
    if not candidates:
        return fixed_action(env)
    return ts.green_phase if ts.green_phase in candidates else candidates[0]


def hybrid_action(env, model, obs):
    """Rule-based legal preemption inside 250 m; learned RL policy otherwise."""
    if emergency_snapshot(env.sumo):
        return preemption_action(env)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

