"""Fixed-budget, three-seed DQN training with experiment isolation and schema versioning."""
import argparse
import hashlib
import importlib.metadata
import json
import time
import traceback
import numpy as np
from runtime import ROOT
from experiments import get_experiment, SCHEMA_OBS_DIMS
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from environment import CrossroadsEnv, load_scenarios


class Progress(BaseCallback):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.rows = []
        self.start = time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % 2500 == 0:
            elapsed = time.perf_counter() - self.start
            row = {"timesteps": self.num_timesteps, "elapsed_seconds": elapsed,
                   "steps_per_second": self.num_timesteps / elapsed}
            self.rows.append(row)
            (self.folder / "timing.json").write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
            self.model.save(self.folder / "latest.zip")
            print(f"{self.folder.parent.name}/{self.folder.name}: {self.num_timesteps} steps, {elapsed:.1f}s", flush=True)
        return True


def train_one(variant, seed, steps=None, group=None, experiment="v2"):
    exp = get_experiment(experiment)
    steps = int(steps if steps is not None else exp.default_steps)
    base_dir = (ROOT / group) if group else exp.models_dir
    folder = base_dir / f"{variant}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "run.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed run: {folder}")
    torch.set_num_threads(1)
    if not getattr(train_one, "threads_set", False):
        torch.set_num_interop_threads(1)
        train_one.threads_set = True
    np.random.seed(seed)

    stratified = exp.experiment_id == "v2"
    obs_dim = SCHEMA_OBS_DIMS[(exp.obs_schema_version, variant)]
    config = dict(
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=exp.default_exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        device="cpu",
    )
    spec = {
        "experiment_id": exp.experiment_id,
        "run_id": exp.run_id,
        "obs_schema_version": exp.obs_schema_version,
        "reward_schema_version": exp.reward_schema_version,
        "obs_dim": obs_dim,
        "stratified_sampling": stratified,
        "variant": variant,
        "seed": seed,
        "total_timesteps": steps,
        "config": config,
        "manifest_sha256": hashlib.sha256(exp.manifest_file.read_bytes()).hexdigest(),
        "checkpoint_selection": "Final checkpoint at predeclared budget; no test-based selection",
        "training_split": "train",
        "evaluation_split": "none during training",
        "torch_threads": 1,
        "sb3_version": importlib.metadata.version("stable-baselines3"),
    }
    (folder / "config.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    env = Monitor(
        CrossroadsEnv(
            load_scenarios("train", experiment=exp.experiment_id),
            variant=variant,
            seed=seed,
            schema_version=exp.obs_schema_version,
            net_file=exp.network_file,
            stratified_sampling=stratified,
        ),
        str(folder / "episodes.monitor.csv"),
        info_keywords=("scenario_id",),
    )
    start = time.perf_counter()
    try:
        model = DQN("MlpPolicy", env, verbose=0, **config)
        model.set_logger(configure(str(folder / "logger"), ["csv"]))
        model.learn(total_timesteps=steps, callback=Progress(folder), log_interval=1)
        model.save(folder / "final.zip")
        # A reload/predict check is distinct from evaluating a trained policy.
        restored = DQN.load(folder / "final.zip", device="cpu")
        obs, _ = env.reset()
        action, _ = restored.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        assert model.num_timesteps == steps
        result = {
            "pass": True,
            "experiment_id": exp.experiment_id,
            "obs_schema_version": exp.obs_schema_version,
            "reward_schema_version": exp.reward_schema_version,
            "obs_dim": obs_dim,
            "variant": variant,
            "seed": seed,
            "requested_steps": steps,
            "completed_steps": model.num_timesteps,
            "elapsed_seconds": time.perf_counter() - start,
            "checkpoint_sha256": hashlib.sha256((folder / "final.zip").read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in model.policy.parameters()),
            "checkpoint_reload_predict": True,
        }
        (folder / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    except Exception:
        (folder / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=["standard", "emergency"], default=["standard", "emergency"])
    parser.add_argument("--group", default=None)
    args = parser.parse_args()
    for variant in args.variants:
        for seed in args.seeds:
            train_one(variant, seed, args.steps, args.group, args.experiment)
