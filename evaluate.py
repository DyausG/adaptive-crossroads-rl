"""Matched held-out evaluation with schema validation, extended metrics, and paired-delta aggregation."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import pandas as pd
from runtime import ROOT
from experiments import (
    get_experiment,
    checkpoint_variant_for_controller,
    validate_checkpoint_schema,
)
import torch
from stable_baselines3 import DQN
from controllers import fixed_action, preemption_action, hybrid_action
from environment import CrossroadsEnv, load_scenarios, episode_metrics


def evaluate(controller, split, model_seed=-1, scenario_id=None, output_root=None, experiment="v2"):
    exp = get_experiment(experiment)
    torch.set_num_threads(1)
    folder = Path(output_root) if output_root else exp.eval_dir / split / controller / f"seed_{model_seed}"
    folder.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(split, experiment=exp.experiment_id)
    if scenario_id:
        scenarios = [s for s in scenarios if s["id"] == scenario_id]
    assert scenarios, f"No scenarios found for split={split!r}, experiment={exp.experiment_id!r}, id={scenario_id!r}"

    model = None
    checkpoint = None
    ckpt_variant = checkpoint_variant_for_controller(controller)
    if ckpt_variant is not None:
        model_folder = exp.models_dir / f"{ckpt_variant}_seed{model_seed}"
        checkpoint = model_folder / "final.zip"
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant)
        model = DQN.load(checkpoint, device="cpu")
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant, loaded_model=model)

    env_variant = "emergency" if controller == "emergency" else "standard"
    env = CrossroadsEnv(
        scenarios,
        variant=env_variant,
        telemetry=True,
        schema_version=exp.obs_schema_version,
        net_file=exp.network_file,
    )
    rows = []
    started = time.perf_counter()
    try:
        for spec in scenarios:
            episode_start = time.perf_counter()
            obs, info = env.reset(options={"scenario": spec})
            done = False
            total_reward = 0.0
            while not done:
                if controller == "fixed":
                    action = fixed_action(env)
                elif controller == "rule":
                    action = preemption_action(env)
                elif controller == "hybrid":
                    action = hybrid_action(env, model, obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                done = terminated or truncated
            metrics = episode_metrics(env)
            metrics.update(
                experiment_id=exp.experiment_id,
                controller=controller,
                model_seed=model_seed,
                route_sha256=spec["sha256"],
                total_reward=total_reward,
                wall_seconds=time.perf_counter() - episode_start,
            )
            assert metrics["ambulance_red_crossings"] == 0, env.red_crossings
            rows.append(metrics)
            pd.DataFrame(env.samples).to_csv(folder / f"{spec['id']}_trace.csv", index=False)
            (folder / f"{spec['id']}_crossings.json").write_text(json.dumps(env.signal_crossings, indent=2), encoding="utf-8")
            pd.DataFrame(rows).to_csv(folder / "episodes.csv", index=False)
            print(
                f"{exp.experiment_id}/{split}/{controller}/seed{model_seed}: {spec['id']} "
                f"passage={metrics['ambulance_passage_seconds']} regular_wait={metrics['regular_wait_mean_inserted']:.1f}",
                flush=True,
            )
    except Exception:
        (folder / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        env.close()

    result = {
        "pass": True,
        "experiment_id": exp.experiment_id,
        "obs_schema_version": exp.obs_schema_version,
        "controller": controller,
        "model_seed": model_seed,
        "split": split,
        "episodes": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "sumo_seed_policy": "Fixed manifest sumo_seed/bg_seed per route, independent of model seed",
    }
    (folder / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def collect(split, experiment="v2"):
    exp = get_experiment(experiment)
    folder = exp.eval_dir / split
    files = sorted(folder.glob("*/seed_*/episodes.csv"))
    assert files, f"No evaluation episode files found under {folder}"
    frames = []
    # Compute horizon-restricted congestion measures from saved raw traces,
    # including insertion backlog, rather than omitting vehicles not admitted.
    for f in files:
        data = pd.read_csv(f)
        pending, congestion = [], []
        for row in data.itertuples():
            trace = pd.read_csv(f.parent / f"{row.scenario}_trace.csv")
            pending.append(int(trace.pending.sum()))
            congestion.append(float((trace.total_queue + trace.pending).sum() /
                                    (row.regular_scheduled + int(row.emergency))))
        data["pending_vehicle_seconds"] = pending
        data["stopped_plus_pending_seconds_per_requested_vehicle"] = congestion
        data.to_csv(f, index=False)
        frames.append(data)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(folder / "all_episodes.csv", index=False)

    rows = []
    for (controller, seed, emergency), group in frame.groupby(["controller", "model_seed", "emergency"]):
        completed = group[group.ambulance_passage_status == "completed"]
        recovered = group[group.recovery_status == "recovered"]
        rows.append({
            "controller": controller,
            "model_seed": seed,
            "emergency": emergency,
            "episodes": len(group),
            "ambulance_completed": len(completed),
            "ambulance_incomplete": int(emergency) * (len(group) - len(completed)),
            "ambulance_passage_mean_completed": completed.ambulance_passage_seconds.mean(),
            "ambulance_passage_p95_completed": completed.ambulance_passage_seconds.quantile(0.95),
            "ambulance_insertion_delay_mean": completed.ambulance_insertion_delay_seconds.mean() if "ambulance_insertion_delay_seconds" in completed else None,
            "ambulance_scheduled_to_clear_mean": completed.ambulance_scheduled_to_clear_seconds.mean() if "ambulance_scheduled_to_clear_seconds" in completed else None,
            "regular_wait_mean_inserted": group.regular_wait_mean_inserted.mean(),
            "regular_time_loss_mean_inserted": group.regular_time_loss_mean_inserted.mean(),
            "max_approach_wait_mean": group.max_approach_wait_seconds.mean() if "max_approach_wait_seconds" in group else None,
            "stopped_plus_pending_seconds_per_requested_vehicle": group.stopped_plus_pending_seconds_per_requested_vehicle.mean(),
            "regular_unfinished": group.regular_unfinished.sum(),
            "regular_not_inserted": group.regular_not_inserted.sum(),
            "recovery_completed": len(recovered),
            "recovery_censored": int((group.recovery_status == "right_censored").sum()),
            "recovery_mean_completed": recovered.queue_recovery_seconds.mean(),
            "post_clearance_excess_queue_area_120s_mean": completed.post_clearance_excess_queue_area_120s.mean() if "post_clearance_excess_queue_area_120s" in completed else None,
            "colliding_vehicle_events": group.colliding_vehicle_events.sum(),
        })
    seeds = pd.DataFrame(rows)
    seeds.to_csv(folder / "seed_summary.csv", index=False)

    # Compute matched-pair causal delta on bg_seed where both normal and emergency exist.
    pair_deltas = {}
    if "bg_seed" in frame.columns:
        for (controller, seed), grp in frame.groupby(["controller", "model_seed"]):
            norm = grp[~grp.emergency].set_index("bg_seed")
            emerg = grp[grp.emergency].set_index("bg_seed")
            common = norm.index.intersection(emerg.index)
            if len(common) > 0:
                pair_deltas[(controller, seed)] = (
                    float((emerg.loc[common, "regular_wait_mean_inserted"] - norm.loc[common, "regular_wait_mean_inserted"]).mean()),
                    float((emerg.loc[common, "stopped_plus_pending_seconds_per_requested_vehicle"] - norm.loc[common, "stopped_plus_pending_seconds_per_requested_vehicle"]).mean()),
                )

    summary = []
    for (controller, emergency), group in seeds.groupby(["controller", "emergency"]):
        raw_subset = frame[(frame.controller == controller) & (frame.emergency == emergency)]
        raw_completed = raw_subset[raw_subset.ambulance_passage_status == "completed"]
        pooled_p95 = float(raw_completed.ambulance_passage_seconds.quantile(0.95)) if len(raw_completed) else None

        row = {
            "controller": controller,
            "emergency": emergency,
            "training_seeds": 0 if controller in ("fixed", "rule") else len(group),
            "episodes": int(group.episodes.sum()),
            "ambulance_incomplete": int(group.ambulance_incomplete.sum()),
            "recovery_censored": int(group.recovery_censored.sum()),
            "regular_unfinished": int(group.regular_unfinished.sum()),
            "regular_not_inserted": int(group.regular_not_inserted.sum()),
            "ambulance_passage_p95_pooled": pooled_p95,
        }
        cols = [
            "ambulance_passage_mean_completed",
            "ambulance_passage_p95_completed",
            "ambulance_insertion_delay_mean",
            "ambulance_scheduled_to_clear_mean",
            "regular_wait_mean_inserted",
            "regular_time_loss_mean_inserted",
            "max_approach_wait_mean",
            "recovery_mean_completed",
            "post_clearance_excess_queue_area_120s_mean",
            "stopped_plus_pending_seconds_per_requested_vehicle",
        ]
        for col in cols:
            if col not in group or group[col].dropna().empty:
                row[col] = None
                row[col + "_seed_std"] = None
            else:
                row[col] = float(group[col].mean())
                row[col + "_seed_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else None
        # Preserve backward-compatible column alias while distinguishing seed-mean vs pooled p95.
        row["ambulance_passage_p95_seed_mean"] = row["ambulance_passage_p95_completed"]
        if emergency:
            deltas = [pair_deltas[(controller, s)] for s in group.model_seed if (controller, s) in pair_deltas]
            if deltas:
                row["paired_regular_wait_delta_mean"] = float(pd.Series([d[0] for d in deltas]).mean())
                row["paired_congestion_delta_mean"] = float(pd.Series([d[1] for d in deltas]).mean())
            else:
                row["paired_regular_wait_delta_mean"] = None
                row["paired_congestion_delta_mean"] = None
        else:
            row["paired_regular_wait_delta_mean"] = None
            row["paired_congestion_delta_mean"] = None
        summary.append(row)
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-collect", action="store_true", help="Use for independent concurrent controller jobs; collect once after all finish.")
    args = parser.parse_args()
    exp = get_experiment(args.experiment)
    controllers = args.controllers if args.controllers is not None else list(exp.controllers)
    for controller in controllers:
        if controller not in exp.controllers:
            raise ValueError(f"Controller {controller!r} not enabled for experiment {exp.experiment_id!r}")
        for seed in args.seeds if checkpoint_variant_for_controller(controller) else [-1]:
            evaluate(controller, args.split, seed, experiment=exp.experiment_id)
    if not args.no_collect:
        collect(args.split, experiment=exp.experiment_id)
