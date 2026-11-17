"""Manifest-driven acceptance checks for independently trained models and matched evaluation."""
import argparse
import json
import pandas as pd
from experiments import get_experiment


def validate(split="test", experiment="v2", write=None):
    exp = get_experiment(experiment)
    if write is None:
        write = (exp.experiment_id != "v1")
    specs = json.loads(exp.manifest_file.read_text(encoding="utf-8"))
    expected = {x["id"]: x for x in specs if x["split"] == split}
    models = []
    for variant in ("standard", "emergency"):
        for seed in (0, 1, 2):
            path = exp.models_dir / f"{variant}_seed{seed}"
            run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            assert run["pass"] and run["completed_steps"] == run["requested_steps"] == exp.default_steps
            assert run["checkpoint_reload_predict"]
            assert run.get("obs_schema_version", "v1") == exp.obs_schema_version
            assert (path / "final.zip").exists()
            models.append(run)

    frame = pd.read_csv(exp.eval_dir / split / "all_episodes.csv")
    expected_runs_per_scenario = sum(3 if c in ("standard", "emergency", "hybrid") else 1 for c in exp.controllers)
    assert len(frame) == expected_runs_per_scenario * len(expected), (len(frame), expected_runs_per_scenario * len(expected))
    assert set(frame.controller) == set(exp.controllers)
    for (controller, seed), group in frame.groupby(["controller", "model_seed"]):
        assert set(group.scenario) == set(expected)
        assert len(group) == len(expected)
        assert all(row.route_sha256 == expected[row.scenario]["sha256"] for row in group.itertuples())
    assert (frame.horizon == 1200).all()
    assert (frame.ambulance_red_crossings == 0).all()
    assert (frame.teleported_vehicles == 0).all()
    assert (frame.regular_scheduled == frame.regular_departed + frame.regular_not_inserted).all()
    assert (frame.regular_departed == frame.regular_completed + frame.regular_unfinished).all()
    assert frame.ambulance_passage_status.isin(["completed", "right_censored", "not_detected", "not_applicable"]).all()

    normal_baselines = frame[(~frame.emergency) & frame.controller.isin(["fixed", "rule"])].pivot(
        index="scenario", columns="controller", values="regular_wait_mean_inserted"
    )
    assert (normal_baselines["fixed"] == normal_baselines["rule"]).all(), "Priority rule must equal cyclic policy when no ambulance exists"

    if "hybrid" in exp.controllers:
        for seed in (0, 1, 2):
            normal_rl = frame[(~frame.emergency) & (frame.model_seed == seed) & frame.controller.isin(["standard", "hybrid"])].pivot(
                index="scenario", columns="controller", values="regular_wait_mean_inserted"
            )
            assert (normal_rl["standard"] == normal_rl["hybrid"]).all(), f"Hybrid seed {seed} must equal standard DQN on normal scenarios"

    emergency = frame[frame.emergency]
    assert (emergency.ambulance_passage_status == "completed").sum() + (emergency.ambulance_passage_status != "completed").sum() == len(emergency)
    result = {
        "pass": True,
        "experiment_id": exp.experiment_id,
        "obs_schema_version": exp.obs_schema_version,
        "split": split,
        "trained_models": len(models),
        "matched_evaluation_episodes": len(frame),
        "scenarios": len(expected),
        "incomplete_ambulance_passages": int((emergency.ambulance_passage_status != "completed").sum()),
        "censored_queue_recovery": int((emergency.recovery_status == "right_censored").sum()),
        "unfinished_regular_trips": int(frame.regular_unfinished.sum()),
        "uninserted_regular_trips": int(frame.regular_not_inserted.sum()),
        "training_seconds_sum": sum(x["elapsed_seconds"] for x in models),
    }
    if write:
        (exp.results_dir / f"gate3_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", nargs="?", default="test", choices=["validation", "test"])
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--write", action="store_true", help="Force writing gate3 JSON even for frozen v1.")
    args = parser.parse_args()
    validate(args.split, args.experiment, write=True if args.write else None)
