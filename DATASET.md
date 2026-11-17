# Dataset and storage

This project uses **no database server or database file**. It stores inputs and results as versioned files. All benchmark datasets, network geometry, model checkpoints, and evaluation records needed to inspect the submission are included in this repository. No database credentials or external data download is required to run the supplied models.

## Included inputs

| Dataset | Location | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| v2 paired study used for current results | `experiments/v2/main/data/` | 48 | 24 | 24 | 96 routes |
| supplementary experiment set | `data/` | 36 | 12 | 24 | 72 routes |

Routes are `.rou.xml` files containing vehicle types, permitted routes, and scheduled departures. Each manifest records configuration, seeds, split, vehicle count, input path and SHA-256. Adapted `.net.xml` files and the inputs/provenance for network conversion are also included. These are synthetic Poisson arrivals, not measured city traffic or human data.

The v2 background seeds are 10000-10023 for training, 20000-20011 for validation and 30000-30011 for test. Each seed produces a matched normal/emergency pair with identical regular arrivals and simulator seed. Ambulance timing uses a separate random stream. The 12-pair blocks balance direction, traffic level and imbalance; they do **not** exhaust all 36 direction × level × imbalance combinations. There is no overlap with v1 seeds.

Low / medium / rush rates are 0.055 / 0.12 / 0.21 vehicles per second per approach. Benchmark directional imbalance uses multipliers 1.45 and 0.55 on opposing axes, preserving total requested intensity. Turns are 80% straight and 10% for each other destination. Regular arrivals end at 900s; simulations end at 1200s. Emergency departure is seeded between 240s and 420s. Physical insertion can be later if an entrance is occupied.

## Results and checkpoints

- `experiments/v2/main/results/evaluation/{validation,test}/all_episodes.csv`: 264 episodes in each split, including normal and emergency cases.
- The controller/seed subdirectories contain `episodes.csv`, raw 1-second `_trace.csv` files, actual signal-crossing records, and checkpoint/input provenance.
- `experiments/v2/main/results/reported_metrics.json`: summaries of the **ambulance-present** held-out test cases. There are 12 cases for each deterministic controller, and 36 for each learned/hybrid controller (12 routes × 3 model seeds).
- `experiments/v2/main/models/{standard,emergency}_seed{0,1,2}/`: six final checkpoints plus configurations, timings and training logs. The hybrid reuses standard DQN checkpoints; it is not an additional trained network.
- Root-level `models/`, `results/` and `demo/` hold the supplementary experiment set. The project report is in `reports/final/`.

## Important fields

| Field or group | Meaning and units |
| --- | --- |
| `scenario`, `bg_seed`, `route_seed`, `route_sha256` | Scenario identity and exact traffic provenance |
| `controller`, `model_seed`, `experiment_id` | Controller, learner seed (-1 for non-learned baselines) and schema family |
| `ambulance_requested_depart`, `ambulance_depart_actual` | Scheduled and actual insertion times in simulated seconds |
| `ambulance_entry`, `ambulance_clear` | First detection within 250m and first outgoing-road sample, in simulated seconds |
| `ambulance_passage_seconds` | Detection-to-clear interval, not total journey time |
| `ambulance_scheduled_to_clear_seconds` | Requested-departure-to-clear interval, including insertion delay and travel before detection |
| `ambulance_passage_status`, `recovery_status` | Completion, not-applicable, or censoring status; empty numeric fields are not zeros |
| `regular_wait_mean_inserted` | Observed stopped vehicle-seconds divided by inserted regular vehicles, including observed wait of unfinished trips |
| `regular_not_inserted`, `regular_unfinished` | Scheduled vehicles never inserted, and inserted vehicles not completed by 1200s |
| `stopped_plus_pending_seconds_per_requested_vehicle` | Sum of stopped and pending vehicle counts over the horizon divided by requested vehicles; a congestion proxy |
| `queue_recovery_seconds` | Time from ambulance clearance to the beginning of 15 consecutive seconds at/below the recovery threshold; censored if not observed |
| `post_clearance_excess_queue_area_120s` | Queue above threshold integrated for up to 120s after clearance, in vehicle-seconds |
| `ambulance_red_crossings`, `colliding_vehicle_events`, `teleported_vehicles` | Recorded simulator events; zero observations do not establish general safety |

Trace files additionally expose `time`, actual signal string and phase, total queue, pending count, per-approach queue/wait, and emergency detection features. Model ZIP files are SB3 checkpoints, not traffic datasets. Only load checkpoints from sources you trust.

## Interactive scenarios

`ui_scenarios.py` writes generated configurations and routes to ignored `results/ui_runs/scenarios/`. Dated run directories contain actual JSONL vehicle coordinates and logs. These are local demonstration recordings, not held-out evaluation. The UI's arbitrary heavy-direction controls increase selected rates by 45% without reducing the other approaches. This differs deliberately from the fixed benchmark's axis-based imbalance.

## Reproduction and license

Run `python verify_repository.py` to check repository-contained inputs, model hashes and independently recomputed result summaries. Run `python generate_routes.py --experiment v2` only if intentionally regenerating the benchmark with the pinned environment. To preserve supplied artifacts, use separate training/evaluation output directories as described in `CONTRIBUTING.md`.

Authored synthetic routes and local code are under the root MIT license. Adapted network assets retain SUMO-RL's MIT notice in `vendor/sumo-rl/LICENSE`. Installed SUMO binaries are separate dependencies under their upstream terms; see `THIRD_PARTY_NOTICES.md`.
