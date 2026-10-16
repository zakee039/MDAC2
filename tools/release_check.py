"""Repeatable engineering regression report. Does not authorize a release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New report directory (must not exist)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    npm = shutil.which('npm.cmd') or shutil.which('npm')
    if npm is None:
        parser.error('npm is required; no partial result can pass')
    steps = [
        ('backend', [sys.executable, '-m', 'pytest', '-q', '--junitxml='+str(output/'backend.xml')], root),
        ('frontend', [npm, 'test'], root/'web'),
        ('build', [npm, 'run', 'build'], root/'web'),
    ]
    # Fingerprint the checked inputs, including uncommitted source. Exclude models,
    # runtime data, generated artifacts and caches; never copy private assets.
    files = [p for base in ('src', 'tests', 'web/src', 'web/tests')
             for p in (root/base).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)]
    files += [*root.glob('*.lock'), root/'pyproject.toml', root/'web/package.json',
              root/'web/package-lock.json', Path(__file__).resolve()]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(files))}
    report = {'schema_version': 1, 'started_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic engineering regression; no music or game acceptance',
              'inputs': hashes, 'steps': [], 'engineering_passed': False,
              'release_authorized': False,
              'external_acceptance': 'Not evaluated by this tool. See construction log and human/game reports.'}
    report_path = output/'report.json'

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    save()
    for name, command, cwd in steps:
        print('Running '+name, flush=True)
        start = time.monotonic()
        try:
            with (output/(name+'.log')).open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=1800, check=False)
            code = completed.returncode
            error = None
        except (OSError, subprocess.TimeoutExpired) as exc:
            code, error = None, type(exc).__name__
        report['steps'].append({'name': name, 'returncode': code, 'error': error,
                                'seconds': round(time.monotonic()-start, 3)})
        save()
        print(name+': '+('passed' if code == 0 else 'FAILED'), flush=True)
    report['inputs_unchanged'] = all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.relative_to(root).as_posix()] for p in files)
    report['engineering_passed'] = report['inputs_unchanged'] and all(s['returncode'] == 0 for s in report['steps'])
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(str(report_path), flush=True)
    return 0 if report['engineering_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
