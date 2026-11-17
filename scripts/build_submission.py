"""Package only the committed project tree and verify every archived file."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    commit = git('rev-parse', 'HEAD').decode().strip()
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT).returncode:
        raise SystemExit('Commit tracked changes before building a submission archive.')
    names = [p for p in git('ls-tree', '-r', '--name-only', '-z', 'HEAD').decode().split('\0') if p]
    prohibited = ('.venv/', '.git/', 'cache/', 'tmp/', 'pilot/', 'reports/final/qa/', 'results/ui_runs/')
    assert not any(n.startswith(prohibited) for n in names)
    for required in ('README.md', 'LICENSE', 'DATASET.md', 'app.py', 'reports/final/Adaptive_Crossroads_Project_Report.pdf'):
        assert required in names, required
    records = [dict(path=n, bytes=(ROOT/n).stat().st_size,
                    sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in names]
    out = ROOT / 'output/submission'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'adaptive-crossroads-rl-submission.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=adaptive-crossroads-rl/',
                    '-o',str(target),commit], cwd=ROOT, check=True)
    manifest = dict(project='Adaptive Crossroads Control with Emergency Vehicle Priority',
                    author='Dyaus Ghangas', roll_number='102308054', course='Reinforcement Learning - UCS760',
                    repository='https://github.com/DyausG/adaptive-crossroads-rl', commit=commit, files=records)
    with zipfile.ZipFile(target,'a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('adaptive-crossroads-rl/SUBMISSION_CONTENTS.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for record in records:
            assert hashlib.sha256(archive.read('adaptive-crossroads-rl/'+record['path'])).hexdigest()==record['sha256']
    receipt = dict(passed=True, file=target.name, bytes=target.stat().st_size,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest(),commit=commit,files=len(records)+1)
    (out/'submission_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
