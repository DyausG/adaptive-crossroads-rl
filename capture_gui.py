"""Genuine SUMO-GUI replay/capture using TraCI; never renders synthetic frames."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
os.environ["CROSSROADS_BACKEND"] = "traci"
os.environ.pop("LIBSUMO_AS_TRACI", None)
from runtime import ROOT
from experiments import (
    get_experiment,
    checkpoint_variant_for_controller,
    validate_checkpoint_schema,
)
import torch
from stable_baselines3 import DQN
from environment import CrossroadsEnv, load_scenarios, episode_metrics
from controllers import fixed_action, preemption_action, hybrid_action


def capture(
    controller,
    model_seed=0,
    scenario_id=None,
    capture_frames=True,
    delay=0.0,
    experiment="v2",
    split="test",
    verify_benchmark=True,
):
    exp = get_experiment(experiment)
    if controller not in exp.controllers:
        raise ValueError(f"Controller {controller!r} is not supported in experiment {exp.experiment_id!r}")
    scenario_id = scenario_id or exp.default_demo_scenario
    torch.set_num_threads(1)
    scenarios = [s for s in load_scenarios(split, experiment=exp.experiment_id) if s["id"] == scenario_id]
    assert len(scenarios) == 1, f"Scenario {scenario_id!r} not found in {exp.experiment_id}/{split}"
    spec = scenarios[0]

    folder = (
        exp.demo_dir / "raw" / f"{controller}_seed{model_seed}"
        if capture_frames
        else ROOT / "results/live" / f"{exp.experiment_id}_{scenario_id}_{controller}_seed{model_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    model = None
    ckpt_variant = checkpoint_variant_for_controller(controller)
    if ckpt_variant is not None:
        model_folder = exp.models_dir / f"{ckpt_variant}_seed{model_seed}"
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant)
        model = DQN.load(model_folder / "final.zip", device="cpu")
        validate_checkpoint_schema(model_folder, exp.obs_schema_version, ckpt_variant, loaded_model=model)

    env_variant = "emergency" if controller == "emergency" else "standard"
    env = CrossroadsEnv(
        scenarios,
        variant=env_variant,
        telemetry=True,
        gui=True,
        schema_version=exp.obs_schema_version,
        net_file=exp.network_file,
    )
    env.additional_sumo_cmd += " --gui-settings-file demo/gui.xml"
    frames = []
    headless_match = False
    try:
        obs, _ = env.reset(options={"scenario": spec})
        x, y = env.sumo.junction.getPosition("t")
        env.sumo.gui.setBoundary("View #0", x - 200, y - 200, x + 200, y + 200)
        env.sumo.gui.setSchema("View #0", "crossroads")
        env.sumo.vehicletype.setColor("regular", (70, 145, 220, 255))
        while env.sim_step < 1200:
            requested_at = int(env.sim_step)
            if capture_frames and 200 <= requested_at <= 900 and requested_at % 20 == 0:
                path = folder / f"frame_{requested_at:04d}.png"
                env.sumo.gui.screenshot("View #0", str(path), 700, 640)
                frames.append({
                    "requested_at_sim_second": requested_at,
                    "rendered": "next simulation step per SUMO API",
                    "path": path.relative_to(ROOT).as_posix(),
                })
            if controller == "fixed":
                action = fixed_action(env)
            elif controller == "rule":
                action = preemption_action(env)
            elif controller == "hybrid":
                action = hybrid_action(env, model, obs)
            else:
                action = int(model.predict(obs, deterministic=True)[0])
            obs, _, _, _, _ = env.step(action)
            if delay:
                time.sleep(delay)
        metrics = episode_metrics(env)
        if verify_benchmark:
            import pandas as pd
            eval_csv = exp.eval_dir / split / controller / f"seed_{model_seed if ckpt_variant else -1}" / "episodes.csv"
            if eval_csv.exists():
                expected = pd.read_csv(eval_csv)
                expected = expected[expected.scenario == scenario_id].iloc[0]
                for key in ["ambulance_passage_seconds", "regular_wait_mean_inserted", "regular_unfinished", "regular_not_inserted"]:
                    assert abs(float(metrics[key]) - float(expected[key])) < 1e-6, (key, metrics[key], expected[key])
                headless_match = True
    finally:
        env.close()

    for frame in frames:
        path = ROOT / frame["path"]
        assert path.exists() and path.stat().st_size > 1000, path
        frame["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "experiment_id": exp.experiment_id,
        "controller": controller,
        "model_seed": model_seed if ckpt_variant else None,
        "scenario": scenario_id,
        "split": split,
        "route_sha256": spec["sha256"],
        "backend": "TraCI + sumo-gui",
        "LIBSUMO_AS_TRACI": None,
        "headless_metrics_match": headless_match,
        "benchmark_verified": bool(verify_benchmark),
        "metrics": metrics,
        "frames": frames,
        "selection": f"First held-out asymmetric rush-hour ambulance scenario ({scenario_id}); model seed {model_seed if ckpt_variant else 'N/A'}, predeclared before evaluation",
    }
    (folder / "capture.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": exp.experiment_id,
        "controller": controller,
        "frames": len(frames),
        "headless_match": headless_match,
        "passage_seconds": metrics["ambulance_passage_seconds"],
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--controllers", nargs="+", default=["rule", "emergency"], choices=["fixed", "rule", "standard", "emergency", "hybrid"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--skip-benchmark-assert", action="store_true", help="Disable saved-CSV equality assertion for interactive/non-benchmark runs.")
    parser.add_argument("--write", action="store_true", help="Required if intentionally capturing canonical v1 demo frames.")
    args = parser.parse_args()
    if args.experiment == "v1" and not args.no_capture and not args.write:
        raise RuntimeError("Refusing to overwrite frozen v1 demo frames without --write")
    for c in args.controllers:
        capture(
            c,
            args.seed,
            args.scenario,
            not args.no_capture,
            args.delay,
            experiment=args.experiment,
            split=args.split,
            verify_benchmark=not args.skip_benchmark_assert,
        )
