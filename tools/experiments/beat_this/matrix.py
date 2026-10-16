"""Run the six real CPU experiments serially with a per-process timeout."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
python=sys.executable
for model in ('final0','small0'):
    for case in ('fixed_4_4','pickup_4_4','compound_6_8'):
        result=subprocess.run([str(python),str(Path(__file__).with_name('run.py')),'--model',model,'--case',case],
            cwd=ROOT,timeout=180,capture_output=True,text=True,encoding='utf-8')
        if result.returncode:
            print(result.stderr,flush=True)
            raise SystemExit(result.returncode)
        metrics=json.loads(result.stdout.strip().splitlines()[-1])
        print(json.dumps({k:metrics[k] for k in ('model','case','elapsed_seconds','peak_working_set_bytes','beat_f_measure_70ms','downbeat_f_measure_70ms','fit_accepted')}),flush=True)
