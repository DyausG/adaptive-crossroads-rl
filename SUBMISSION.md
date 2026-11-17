# Submission guide

**Project:** Adaptive Crossroads Control with Emergency Vehicle Priority

**Student:** Dyaus Ghangas, 102308054

**Course:** Reinforcement Learning - UCS760

## Submit these

1. Repository URL: https://github.com/DyausG/adaptive-crossroads-rl
2. Report: `reports/final/Adaptive_Crossroads_Project_Report.pdf`
3. If the submission portal asks for a source ZIP, build it using the command below. It contains the committed code, datasets, checkpoints, report and result evidence. It excludes virtual environments, caches, private development records and local interactive recordings.

The repository uses file-based datasets and results, not a database. No database dump, credentials or service setup is needed. The primary study includes 96 synthetic routes and six trained DQN checkpoints; the supplementary experiment fixtures are also included.

## Run for evaluation

Follow the README's Windows/Python 3.12 setup, then run:

```powershell
& '.venv\Scripts\python.exe' app.py
& '.venv\Scripts\python.exe' verify_repository.py --replay
& '.venv\Scripts\python.exe' -m pytest tests -q -p no:cacheprovider
```

For a short demonstration select Side-by-side, choose Emergency-aware RL and Rule-based preemption, select Rush hour and Start. Both views use identical traffic. Inject an ambulance using N/S/E/W at the live edge, pause or scrub recorded history, then Stop. Insertion may wait for entrance space. Results from this interactive demonstration are not benchmark measurements.

## Build a source archive

After committing the intended submission:

```powershell
& '.venv\Scripts\python.exe' scripts\build_submission.py
```

Output: `output/submission/adaptive-crossroads-rl-submission.zip`, with a SHA-256 receipt beside it. The archive is generated from the committed Git tree, not from unrelated local files. `SUBMISSION_CONTENTS.json` inside it lists per-file hashes and the source commit. Extract the ZIP and follow the same README setup; the virtual environment is intentionally excluded.

## Report source

`reports/final/Adaptive_Crossroads_Project_Report.md` is the editable report source. `metadata.json` stores the supplied student details; no institution or supervisor is invented. Numeric tables come from `report_evidence.json`, independently recomputed from raw episode records.

To regenerate the PDF and Markdown with the documented report dependencies:

```powershell
python -m pip install -r requirements-report.txt
python scripts/build_project_report.py
```

The PDF has a contents page, metric definitions, methodology, controller/training details, results, charts, actual application capture, implementation difficulties, verification, limitations and references. Retain the code/data licenses when redistributing the submission.

## Before uploading

Check the portal's file-size and naming rules and whether it requires a specific institutional cover sheet. The supplied report includes the name, roll number and course given for this project. Submission to a university portal is separate from publishing this repository; no portal upload is claimed by the repository.
