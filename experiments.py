"""Experiment registry, path resolution, and MDP schema version guards."""
from dataclasses import dataclass
import json
from pathlib import Path
from runtime import ROOT

OBS_SCHEMA_V1 = "v1"
OBS_SCHEMA_V2 = "v2"
REWARD_SCHEMA_V1 = "v1"
REWARD_SCHEMA_V2 = "v2"

SCHEMA_OBS_DIMS = {
    ("v1", "standard"): 21,
    ("v1", "emergency"): 29,
    ("v2", "standard"): 26,
    ("v2", "emergency"): 34,
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    run_id: str
    label: str
    root: Path
    config_file: Path | None
    data_dir: Path
    network_file: Path
    manifest_file: Path
    routes_dir: Path
    models_dir: Path
    results_dir: Path
    eval_dir: Path
    figures_dir: Path
    reported_metrics_json: Path
    reported_metrics_csv: Path
    demo_dir: Path
    report_md: Path
    report_pdf: Path
    obs_schema_version: str
    reward_schema_version: str
    controllers: tuple[str, ...]
    default_steps: int
    default_demo_scenario: str
    default_exploration_fraction: float


def normalize_experiment_id(experiment: str | None) -> str:
    if not experiment or experiment in ("v2", "v2/main", "experiments/v2/main"):
        return "v2"
    if experiment in ("v1", "baseline"):
        return "v1"
    raise ValueError(f"Unsupported experiment id: {experiment!r} (expected 'v1' or 'v2')")


def get_experiment(experiment: str | None = "v2") -> ExperimentSpec:
    exp_id = normalize_experiment_id(experiment)
    if exp_id == "v1":
        return ExperimentSpec(
            experiment_id="v1",
            run_id="baseline",
            label="v1 — Original Baseline (72 routes, 20k DQN)",
            root=ROOT,
            config_file=None,
            data_dir=ROOT / "data",
            network_file=ROOT / "data/network/crossroads.net.xml",
            manifest_file=ROOT / "data/manifest.json",
            routes_dir=ROOT / "data/routes",
            models_dir=ROOT / "models",
            results_dir=ROOT / "results",
            eval_dir=ROOT / "results/evaluation",
            figures_dir=ROOT / "results/figures",
            reported_metrics_json=ROOT / "results/reported_metrics.json",
            reported_metrics_csv=ROOT / "results/reported_metrics.csv",
            demo_dir=ROOT / "demo",
            report_md=ROOT / "report/report.md",
            report_pdf=ROOT / "output/pdf/crossroads_report.pdf",
            obs_schema_version=OBS_SCHEMA_V1,
            reward_schema_version=REWARD_SCHEMA_V1,
            controllers=("fixed", "rule", "standard", "emergency"),
            default_steps=20000,
            default_demo_scenario="scenario_3002",
            default_exploration_fraction=0.5,
        )
    base = ROOT / "experiments/v2/main"
    return ExperimentSpec(
        experiment_id="v2",
        run_id="main",
        label="v2 — Balanced Paired Benchmark (96 routes, 40k DQN + Hybrid)",
        root=base,
        config_file=base / "config.json",
        data_dir=base / "data",
        network_file=base / "data/network/crossroads.net.xml",
        manifest_file=base / "data/manifest.json",
        routes_dir=base / "data/routes",
        models_dir=base / "models",
        results_dir=base / "results",
        eval_dir=base / "results/evaluation",
        figures_dir=base / "results/figures",
        reported_metrics_json=base / "results/reported_metrics.json",
        reported_metrics_csv=base / "results/reported_metrics.csv",
        demo_dir=base / "demo",
        report_md=base / "report/report.md",
        report_pdf=base / "report/crossroads_report.pdf",
        obs_schema_version=OBS_SCHEMA_V2,
        reward_schema_version=REWARD_SCHEMA_V2,
        controllers=("fixed", "rule", "standard", "emergency", "hybrid"),
        default_steps=40000,
        default_demo_scenario="scenario_30002_emergency",
        default_exploration_fraction=0.35,
    )


def checkpoint_variant_for_controller(controller: str) -> str | None:
    if controller in ("standard", "hybrid"):
        return "standard"
    if controller == "emergency":
        return "emergency"
    return None


def validate_checkpoint_schema(model_folder: Path, expected_obs_schema: str, expected_variant: str, loaded_model=None) -> dict:
    """Ensure checkpoint config and observation dimension match the target MDP schema."""
    config_path = model_folder / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    actual_schema = cfg.get("obs_schema_version", OBS_SCHEMA_V1)
    if actual_schema != expected_obs_schema:
        raise ValueError(
            f"Observation schema mismatch for {model_folder}: checkpoint uses {actual_schema!r}, "
            f"environment expects {expected_obs_schema!r}"
        )
    actual_variant = cfg.get("variant")
    if actual_variant != expected_variant:
        raise ValueError(
            f"Model variant mismatch for {model_folder}: checkpoint is {actual_variant!r}, "
            f"expected {expected_variant!r}"
        )
    expected_dim = SCHEMA_OBS_DIMS[(expected_obs_schema, expected_variant)]
    if "obs_dim" in cfg and int(cfg["obs_dim"]) != expected_dim:
        raise ValueError(
            f"Recorded obs_dim mismatch in {config_path}: {cfg['obs_dim']} != {expected_dim}"
        )
    if loaded_model is not None:
        shape = getattr(loaded_model.observation_space, "shape", None)
        if not shape or int(shape[0]) != expected_dim:
            raise ValueError(
                f"SB3 checkpoint observation_space {shape} does not match expected ({expected_dim},) "
                f"for schema={expected_obs_schema!r}, variant={expected_variant!r}"
            )
    return cfg
