"""Verify a fresh clone using only repository-contained evidence; no parent ZIPs needed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def offline_replay(experiment, scenario_id):
    """Reproduce a supplied policy without needing any external archive or report."""
    from runtime import ROOT as runtime_root
    from experiments import get_experiment, validate_checkpoint_schema
    import torch
    import pandas as pd
    from stable_baselines3 import DQN
    from environment import CrossroadsEnv, load_scenarios, episode_metrics
    torch.set_num_threads(1)
    exp = get_experiment(experiment)
    folder = exp.models_dir / 'emergency_seed0'
    model = DQN.load(folder / 'final.zip', device='cpu')
    validate_checkpoint_schema(folder, exp.obs_schema_version, 'emergency', loaded_model=model)
    spec = next(s for s in load_scenarios('test', experiment=experiment) if s['id'] == scenario_id)
    env = CrossroadsEnv([spec], variant='emergency', telemetry=True,
                        schema_version=exp.obs_schema_version, net_file=exp.network_file)
    try:
        obs, _ = env.reset(options={'scenario': spec})
        while env.sim_step < 1200:
            action, _ = model.predict(obs, deterministic=True)
            obs, *_ = env.step(int(action))
        metrics = episode_metrics(env)
    finally:
        env.close()
    reference = pd.read_csv(exp.eval_dir / 'test/emergency/seed_0/episodes.csv').set_index('scenario').loc[scenario_id]
    for field in ('ambulance_passage_seconds', 'regular_wait_mean_inserted', 'regular_unfinished', 'regular_not_inserted'):
        assert abs(float(metrics[field]) - float(reference[field])) < 1e-6, (experiment, field)
    return float(metrics['ambulance_passage_seconds'])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def evidence():
    models, route_counts = [], {}
    for version, base in [('v1', ROOT), ('v2', ROOT / 'experiments/v2/main')]:
        manifest = json.loads((base / 'data/manifest.json').read_text(encoding='utf-8'))
        assert len(manifest) == (72 if version == 'v1' else 96)
        route_counts[version] = len(manifest)
        for spec in manifest:
            assert sha256(ROOT / spec['route_file']) == spec['sha256'], spec['id']
        for variant in ('standard', 'emergency'):
            for seed in range(3):
                folder = base / 'models' / f'{variant}_seed{seed}'
                run = json.loads((folder / 'run.json').read_text())
                assert run['pass'] and run['checkpoint_reload_predict']
                assert sha256(folder / 'final.zip') == run['checkpoint_sha256']
                models.append(dict(experiment=version, variant=variant, seed=seed,
                                   steps=run['completed_steps'], seconds=run['elapsed_seconds']))
    base = ROOT / 'experiments/v2/main'
    manifest = {s['id']: s for s in json.loads((base / 'data/manifest.json').read_text())}
    raw = list(csv.DictReader((base / 'results/evaluation/test/all_episodes.csv').open(newline='')))
    validation = list(csv.DictReader((base / 'results/evaluation/validation/all_episodes.csv').open(newline='')))
    assert len(raw) == len(validation) == 264
    for split, records in [('test', raw), ('validation', validation)]:
        for row in records:
            assert manifest[row['scenario']]['split'] == split
            assert row['route_sha256'] == manifest[row['scenario']]['sha256']
            assert float(row['horizon']) == 1200
    summaries = json.loads((base / 'results/reported_metrics.json').read_text())
    checks = {}
    for reference in summaries:
        controller = reference['controller']
        records = [r for r in raw if r['controller'] == controller and r['emergency'] == 'True']
        vals = lambda key: [float(r[key]) for r in records if r[key] != '']
        seeds = sorted({r['model_seed'] for r in records})
        seed_means = [statistics.mean(float(r['ambulance_passage_seconds']) for r in records if r['model_seed'] == seed) for seed in seeds]
        calculated = dict(
            runs=len(records), passage_mean=statistics.mean(vals('ambulance_passage_seconds')),
            passage_p95_pooled=percentile(vals('ambulance_passage_seconds'), .95),
            scheduled_to_clear_mean=statistics.mean(vals('ambulance_scheduled_to_clear_seconds')),
            regular_wait=statistics.mean(vals('regular_wait_mean_inserted')),
            congestion=statistics.mean(vals('stopped_plus_pending_seconds_per_requested_vehicle')),
            excess_queue_area_120s=statistics.mean(vals('post_clearance_excess_queue_area_120s')),
            unfinished_regular=sum(int(r['regular_unfinished']) for r in records),
            uninserted_regular=sum(int(r['regular_not_inserted']) for r in records),
            recovery_censored=sum(r['recovery_status'] == 'right_censored' for r in records))
        for key, value in calculated.items():
            assert abs(value - reference[key]) < 1e-8, (controller, key, value, reference[key])
        calculated['seed_means'] = seed_means
        calculated['seed_sd'] = statistics.stdev(seed_means) if len(seed_means) > 1 else None
        if calculated['seed_sd'] is not None:
            assert abs(calculated['seed_sd'] - reference['passage_seed_std']) < 1e-8
        checks[controller] = calculated
    totals = {key: sum(int(r[key]) for r in raw) for key in
              ('ambulance_red_crossings', 'colliding_vehicle_events', 'teleported_vehicles', 'regular_unfinished', 'regular_not_inserted')}
    assert all(totals[k] == 0 for k in ('ambulance_red_crossings', 'colliding_vehicle_events', 'teleported_vehicles'))
    gate = json.loads((base / 'results/gate3_test.json').read_text())
    assert totals['regular_unfinished'] == gate['unfinished_regular_trips']
    assert totals['regular_not_inserted'] == gate['uninserted_regular_trips']
    return dict(passed=True, route_counts=route_counts, final_checkpoints_checked=len(models),
                test_episodes=len(raw), validation_episodes=len(validation), summaries=checks,
                test_event_totals=totals, models=models,
                source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
                  (base / 'results/evaluation/test/all_episodes.csv', base / 'results/reported_metrics.json', base / 'config.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ci/repository_check.json')
    parser.add_argument('--replay', action='store_true', help='Also replay archived v1/v2 policies offline with networking blocked')
    args = parser.parse_args()
    result = evidence()
    if args.replay:
        import socket
        def deny(*_args, **_kwargs):
            raise AssertionError('Unexpected external connection during replay')
        socket.socket.connect = socket.socket.connect_ex = deny
        result['offline_replay_passages'] = dict(v1=offline_replay('v1', 'scenario_3002'),
                                                v2=offline_replay('v2', 'scenario_30002_emergency'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('summaries', 'models', 'source_hashes')}, indent=2))


if __name__ == '__main__':
    main()
