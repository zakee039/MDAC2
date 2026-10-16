"""Finite local completion pipeline for the authorized v4/v4b experiments.
No training restarts, source edits, publication, or production promotion.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PYTHONUTF8','1')

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def main(variant):
    if variant not in ('v4','v4b'):raise ValueError('Bounded variants only')
    r=ROOT/f'tools/private/community/supervised-dual-{variant}-20260930';started=time.monotonic();commands=[]
    def status(stage,**extra):write(r/'completion-status.json',dict(stage=stage,variant=variant,time=time.strftime('%Y-%m-%d %H:%M:%S'),commands=commands,**extra))
    status('waiting_for_training')
    while True:
        if time.monotonic()-started>5400:raise TimeoutError('90-minute completion bound exceeded')
        p=r/'formal-arrangement/status.json'
        if p.exists():
            state=json.loads(p.read_text(encoding='utf-8'))
            if state['status']=='completed':break
            if state['status'] in ('failed','cancelled'):raise RuntimeError('Training stopped: '+state['status'])
        time.sleep(15)
    def run(stage,tool,args,expected):
        if expected.exists():commands.append(dict(stage=stage,reused=str(expected)));return
        command=[sys.executable,str(ROOT/'tools'/tool),*map(str,args)];commands.append(dict(stage=stage,command=command));status(stage)
        with (r/(stage+'-pipeline.log')).open('w',encoding='utf-8') as stream:
            subprocess.run(command,cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        if not expected.exists():raise RuntimeError('Missing stage output '+str(expected))
    run('evaluation',f'evaluate_arrangement_context_{variant}.py',['--experiment',r,'--output',r/'evaluation'],r/'evaluation/validation.json')
    run('comparison','compare_arrangement_context_v4.py',['--experiment',r,'--old-evaluation',ROOT/'tools/private/community/supervised-dual-v2-20260928/calibration-v1/alpha-0.5','--output',r/'comparison.json'],r/'comparison.json')
    run('hierarchy','diagnose_arrangement_hierarchy_v4.py',['--experiment',r,'--output',r/'hierarchy-diagnosis.json'],r/'hierarchy-diagnosis.json')
    run('audio',f'diagnose_arrangement_context_audio_{variant}.py',['--experiment',r,'--output',r/'audio-dependence.json'],r/'audio-dependence.json')
    run('review','build_arrangement_context_review_v4.py',['--experiment',r,'--evaluation',r/'evaluation','--output',r/f'review-{variant}'],r/f'review-{variant}/review.json')
    run('pipaxing',f'predict_arrangement_context_cached_song_{variant}.py',['--experiment',r,'--evaluation',r/'evaluation','--cached',ROOT/'tools/private/playtests/20260927-pipaxing-v1','--output',r/f'pipaxing-review-{variant}'],r/f'pipaxing-review-{variant}/report.json')
    status('completed',human_listening='pending',game_acceptance='not_evaluated',production_promoted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['v4','v4b']);a=p.parse_args()
    try:main(a.variant)
    except BaseException as error:
        write(ROOT/f'tools/private/community/supervised-dual-{a.variant}-20260930/completion-error.json',dict(error=repr(error),time=time.strftime('%Y-%m-%d %H:%M:%S')));raise
