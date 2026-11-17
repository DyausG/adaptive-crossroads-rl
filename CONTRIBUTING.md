# Development and reproducibility

Use Windows x64 and Python 3.12 for the tested setup. Follow the README installation steps and retain the pinned dependency versions when checking byte-identical route regeneration.

## Before a change

Keep historical datasets, models, reports and evaluation outputs intact. Use a separate directory for new research runs; never replace supplied evidence merely to make a check pass. Keep interactive sandbox results separate from held-out benchmark claims.

```powershell
python -m pytest tests -q -p no:cacheprovider
python verify_repository.py
```

`validate_ui.py --screenshots` invokes actual Tk controls and SUMO workers; it requires an interactive Windows desktop. `--quick` omits the full 1200s completion repeat while exercising layout and interactions. The supplied Windows CI template runs the pytest suite and repository verifier after it is activated as documented in the README.

## New experiments

`train.py` refuses to overwrite a completed run. Train separately with, for example:

```powershell
python train.py --experiment v2 --steps 40000 --seeds 0 1 2 --group results/reproduction/models
```

The existing evaluator loads the registry's supplied checkpoints. To reproduce their evaluation without overwriting evidence, call its function with a separate output root:

```powershell
python -c "from evaluate import evaluate; evaluate('emergency', 'test', 0, output_root='results/reproduction/emergency_seed0', experiment='v2')"
```

Do not describe the `--group` training output as evaluated by that command: evaluating a newly trained model requires explicitly configuring a separate experiment/model location. Keep seeds, input hashes, checkpoint hashes, schema versions and metrics with every new result. Report sample counts, aggregation and censoring. Do not infer causality from v1-to-v2 comparisons or hide a strong deterministic baseline.

## Pull requests

Explain the problem, resulting behavior and checks performed. For UI changes verify controller exclusivity, configuration staging, matching comparison inputs, time barriers, pause/seek, injections, Stop/Restart and startup cancellation. Do not alter raw traces to create a smoother animation. Preserve third-party notices and identify any upstream-derived work.
