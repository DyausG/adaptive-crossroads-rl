# Matched recorded comparison

Genuine SUMO-GUI screenshots on scenario_3002: the first held-out asymmetric rush-hour ambulance scenario. Rule preemption versus emergency-aware model seed 0. Both GUI replays exactly matched headless passage time, regular waiting and unfinished/uninserted counts.

36 paired frames, requested at simulation times 200 through 900s in 20s intervals. SUMO saves a requested screenshot on the following simulation step. GIF playback is a sampled time-compressed recording, not a live simulation. Original PNGs, screenshot hashes and controller metrics are in raw/*/capture.json. No generated vehicle imagery or fabricated improvement overlays are used.

Run `python capture_gui.py` with the project environment from this folder's parent, then `python compose_demo.py`. For live playback use `python capture_gui.py --controllers emergency --seed 0 --scenario scenario_3002 --no-capture --delay 0.1`. TraCI is used and LIBSUMO_AS_TRACI is unset for capture. Visual settings change colors/sizes only; checked dynamics match headless outputs.
