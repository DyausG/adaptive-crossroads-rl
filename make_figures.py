"""Rebuild scientific plots and machine-readable report values from saved evaluation records."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import get_experiment

LABELS = {
    "fixed": "Fixed-time",
    "rule": "Rule preemption",
    "standard": "Standard DQN",
    "emergency": "Emergency-aware DQN",
    "hybrid": "Hybrid (Rule+DQN)",
}
SHORT_LABELS = {
    "fixed": "Fixed-time",
    "rule": "Rule\npreemption",
    "standard": "Standard\nDQN",
    "emergency": "Emergency\nDQN",
    "hybrid": "Hybrid\n(Rule+DQN)",
}
COLORS = {
    "fixed": "#8794a6",
    "rule": "#208a80",
    "standard": "#416ac4",
    "emergency": "#cf7451",
    "hybrid": "#6e56cf",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#28374d",
    "text.color": "#18283c",
    "xtick.color": "#46556a",
    "ytick.color": "#46556a",
    "savefig.facecolor": "white",
})


def save(fig, out_dir, name):
    fig.savefig(out_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(experiment="v2", write=None):
    exp = get_experiment(experiment)
    if write is None:
        write = (exp.experiment_id != "v1")
    if not write:
        print("v1 artifacts are frozen; pass --write to regenerate v1 figures explicitly.")
        return

    out_dir = exp.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(exp.eval_dir / "test/all_episodes.csv")
    seeds = pd.read_csv(exp.eval_dir / "test/seed_summary.csv")
    summary_df = pd.read_csv(exp.eval_dir / "test/summary.csv")
    order = list(exp.controllers)

    summaries = []
    for controller in order:
        group = raw[(raw.controller == controller) & raw.emergency]
        seed_group = seeds[(seeds.controller == controller) & seeds.emergency]
        sum_row = summary_df[(summary_df.controller == controller) & summary_df.emergency].iloc[0]
        completed = group[group.ambulance_passage_status == "completed"]
        recovered = group[group.recovery_status == "recovered"]

        def opt_mean(col):
            if col not in group or group[col].dropna().empty:
                return None
            return float(group[col].dropna().mean())

        def opt_sum_row(col):
            if col not in sum_row or pd.isna(sum_row[col]):
                return None
            return float(sum_row[col])

        summaries.append({
            "controller": controller,
            "label": LABELS[controller],
            "runs": int(len(group)),
            "ambulance_completed": int(len(completed)),
            "ambulance_incomplete": int(len(group) - len(completed)),
            "passage_mean": float(completed.ambulance_passage_seconds.mean()) if len(completed) else None,
            "passage_p95": float(completed.ambulance_passage_seconds.quantile(0.95)) if len(completed) else None,
            "passage_p95_pooled": float(completed.ambulance_passage_seconds.quantile(0.95)) if len(completed) else None,
            "passage_p95_seed_mean": opt_sum_row("ambulance_passage_p95_seed_mean"),
            "passage_seed_std": float(seed_group.ambulance_passage_mean_completed.std(ddof=1)) if len(seed_group) > 1 else None,
            "insertion_delay_mean": opt_mean("ambulance_insertion_delay_seconds"),
            "scheduled_to_clear_mean": opt_mean("ambulance_scheduled_to_clear_seconds"),
            "regular_wait": float(group.regular_wait_mean_inserted.mean()),
            "regular_wait_seed_std": float(seed_group.regular_wait_mean_inserted.std(ddof=1)) if len(seed_group) > 1 else None,
            "max_approach_wait_mean": opt_mean("max_approach_wait_seconds"),
            "congestion": float(group.stopped_plus_pending_seconds_per_requested_vehicle.mean()),
            "recovery_completed": int(len(recovered)),
            "recovery_censored": int((group.recovery_status == "right_censored").sum()),
            "recovery_mean_completed": float(recovered.queue_recovery_seconds.mean()) if len(recovered) else None,
            "excess_queue_area_120s": opt_mean("post_clearance_excess_queue_area_120s"),
            "paired_regular_wait_delta": opt_sum_row("paired_regular_wait_delta_mean"),
            "paired_congestion_delta": opt_sum_row("paired_congestion_delta_mean"),
            "unfinished_regular": int(group.regular_unfinished.sum()),
            "uninserted_regular": int(group.regular_not_inserted.sum()),
        })

    exp.reported_metrics_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(exp.reported_metrics_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), layout="constrained")
    for ax, key, sd, title in [
        (axes[0], "passage_mean", "passage_seed_std", "Ambulance passage (250m to clear)"),
        (axes[1], "regular_wait", "regular_wait_seed_std", "Regular-vehicle stopped time"),
    ]:
        means = [r[key] or 0.0 for r in summaries]
        errors = [r[sd] or 0.0 for r in summaries]
        ax.bar(range(len(order)), means, yerr=errors, capsize=4, color=[COLORS[c] for c in order], width=0.6)
        ax.set_xticks(range(len(order)), [SHORT_LABELS[c] for c in order])
        ax.set_ylabel("Seconds")
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        top = max((m + e for m, e in zip(means, errors)), default=10.0)
        ax.set_ylim(0, max(10.0, top * 1.28))
        ax.yaxis.grid(True, alpha=0.15)
        ax.set_axisbelow(True)
        for i, mean in enumerate(means):
            ax.text(i, mean + errors[i] + top * 0.02, f"{mean:.1f}", ha="center", fontsize=9.5)
    fig.suptitle(f"Held-out ambulance scenarios ({exp.experiment_id}): mean +/- seed SD", fontsize=13, fontweight="bold")
    save(fig, out_dir, "controller_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), layout="constrained")
    for ax, variant in zip(axes, ["standard", "emergency"]):
        curves = []
        for seed in [0, 1, 2]:
            monitor = pd.read_csv(exp.models_dir / f"{variant}_seed{seed}/episodes.monitor.csv", comment="#")
            curves.append(monitor.r.rolling(5, min_periods=1).mean().to_numpy())
        n = min(map(len, curves))
        values = np.stack([x[:n] for x in curves])
        x = np.arange(1, n + 1) * 240
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.plot(x, mean, color=COLORS[variant], lw=2)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[variant], alpha=0.18)
        ax.set_title(LABELS[variant], loc="left", fontweight="bold")
        ax.set_xlabel("Agent interactions")
        ax.set_ylabel("Own training reward / episode")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Learning curves ({exp.experiment_id}): mean +/- sample SD across three seeds", fontsize=13, fontweight="bold")
    save(fig, out_dir, "learning_curves.png")

    case_id = exp.default_demo_scenario
    case_controllers = ["rule", "emergency", "hybrid"] if "hybrid" in order else ["rule", "emergency"]
    fig, axes = plt.subplots(1, len(case_controllers), figsize=(5.6 * len(case_controllers), 4.2), layout="constrained", sharey=True)
    if len(case_controllers) == 1:
        axes = [axes]
    max_q = 10
    for ax, controller in zip(axes, case_controllers):
        seed = -1 if controller == "rule" else 0
        trace = pd.read_csv(exp.eval_dir / f"test/{controller}/seed_{seed}/{case_id}_trace.csv")
        sub = trace[(trace.time >= 150) & (trace.time <= 1050)]
        max_q = max(max_q, int(sub.total_queue.max()))
        metric = raw[(raw.controller == controller) & (raw.model_seed == seed) & (raw.scenario == case_id)].iloc[0]
        ax.plot(trace.time, trace.total_queue, color=COLORS[controller], lw=1.3, label="Stopped vehicles")
        ax.axhline(metric.recovery_threshold, color="#7c8796", ls="--", lw=1, label="Recovery threshold")
        ax.axvline(metric.ambulance_entry, color="#4c637e", ls=":", lw=1, label="250 m detection")
        ax.axvline(metric.ambulance_clear, color="#a75631", ls="-.", lw=1, label="Junction cleared")
        title_suffix = " (seed 0)" if controller != "rule" else ""
        ax.set_title(f"{LABELS[controller]}{title_suffix}", loc="left", fontweight="bold")
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Total queue (vehicles)")
        ax.set_xlim(150, 1050)
        ax.grid(alpha=0.12)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylim(0, max_q * 1.18)
    fig.suptitle(f"Matched rush-hour case: {case_id}", fontsize=13, fontweight="bold")
    save(fig, out_dir, "queue_recovery.png")

    fig, ax = plt.subplots(figsize=(10.0, 3.8), layout="constrained")
    x = np.arange(len(order))
    width = 0.34
    max_cong = 0.0
    for offset, emergency, label in [(-width / 2, False, "Normal scenarios"), (width / 2, True, "Ambulance scenarios")]:
        values = [float(raw[(raw.controller == c) & (raw.emergency == emergency)].stopped_plus_pending_seconds_per_requested_vehicle.mean()) for c in order]
        max_cong = max(max_cong, max(values))
        bars = ax.bar(x + offset, values, width, label=label, color="#4672a8" if not emergency else "#d58a61")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5)
    ax.set_xticks(x, [LABELS[c] for c in order])
    ax.set_ylabel("Observed seconds / requested vehicle")
    ax.set_title("Horizon-restricted congestion includes vehicles waiting to enter", loc="left", fontweight="bold", pad=12)
    ax.set_ylim(0, max(110.0, max_cong * 1.22))
    ax.legend(frameon=False)
    ax.yaxis.grid(alpha=0.12)
    ax.set_axisbelow(True)
    save(fig, out_dir, "congestion.png")
    print(f"Rebuilt figures and reported metrics for {exp.experiment_id} into {out_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--write", action="store_true", help="Required if intentionally rebuilding frozen v1 figures.")
    args = parser.parse_args()
    main(args.experiment, write=True if args.write else None)
