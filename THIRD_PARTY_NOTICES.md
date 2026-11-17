# Attribution and licenses

## SUMO-RL

Lucas N. Alegre and SUMO-RL contributors. https://github.com/LucasAlegre/sumo-rl

Vendored source revision: `86f0fc41401eeca724b2e8878f5020dc6d70c9b4`; package version 1.4.5. Original MIT notice is retained in `vendor/sumo-rl/LICENSE`. Source files are reused without local algorithm edits. The junction geometry in `data/network` derives from its two-way single-intersection network; we extended the approaches and regenerated the geometry while preserving connections and signal groups. Plain export/adaptation inputs and provenance are retained.

The local upstream checkout also contains unrelated assets, including RESCO networks, with their own provenance. These are excluded from the submission Git snapshot/ZIP. Only the core package and identified single-intersection network assets are needed and included. The experiments use this network and locally generated route files. Do not infer that all upstream datasets share the code license or were used here.

## Installed dependencies

Eclipse SUMO 1.28.0 is installed separately; its runtime reports `EPL-2.0 OR GPL-2.0-or-later`. SUMO binaries are excluded from the submission ZIP. Stable-Baselines3 is MIT licensed; PyTorch, NumPy, pandas, Matplotlib, Pillow, ReportLab and other installed packages retain their respective licenses. Dependency installation does not transfer their authorship to this project.

## Project contributions

Project contributions include scenario generation, ambulance observation/reward definitions, legal preemption, telemetry and metrics, training/evaluation orchestration, validation tests, the desktop application, plots and report generation. Generated routes are synthetic; no measured city traffic or human subject data is claimed. The six primary and six supplementary final checkpoints have training run records and hashes.
