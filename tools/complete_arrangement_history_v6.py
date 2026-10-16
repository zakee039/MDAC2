"""Bounded completion pipeline: preserve outputs and never restart training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('PYTHONUTF8','1')
def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def main():
 r=ROOT/'tools/private/community/supervised-dual-v6-20260930';started=time.monotonic();commands=[]
 def status(stage,**extra):write(r/'completion-status.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),commands=commands,**extra))
 status('waiting_for_training')
 while True:
  if time.monotonic()-started>10800:raise TimeoutError('Three-hour completion bound exceeded')
  p=r/'formal-arrangement/status.json'
  if p.exists():
   state=json.loads(p.read_text(encoding='utf-8'))
   if state['status']=='completed':break
   if state['status'] in ('failed','cancelled'):raise RuntimeError('Training stopped: '+state['status'])
  time.sleep(15)
 def run(stage,tool,args,expected):
  if expected.exists():commands.append(dict(stage=stage,reused=str(expected)));return
  command=[sys.executable,str(ROOT/'tools'/tool),*map(str,args)];commands.append(dict(stage=stage,command=command));status(stage)
  with (r/(stage+'-pipeline.log')).open('w',encoding='utf-8') as stream:subprocess.run(command,cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
  if not expected.exists():raise RuntimeError('Missing output '+str(expected))
 run('evaluation','evaluate_arrangement_context_v6.py',['--experiment',r,'--output',r/'evaluation'],r/'evaluation/validation.json')
 run('history-ablation','evaluate_arrangement_context_v6.py',['--experiment',r,'--output',r/'evaluation-without-history','--history-disabled'],r/'evaluation-without-history/validation.json')
 run('comparison','compare_arrangement_context_v6.py',['--experiment',r,'--old-evaluation',ROOT/'tools/private/community/supervised-dual-v4b-20260930/evaluation','--output',r/'comparison.json'],r/'comparison.json')
 run('relations','diagnose_arrangement_relations_v6.py',['--experiment',r,'--output',r/'relations-diagnosis.json'],r/'relations-diagnosis.json')
 run('hierarchy','diagnose_arrangement_hierarchy_v6.py',['--experiment',r,'--output',r/'hierarchy-diagnosis.json'],r/'hierarchy-diagnosis.json')
 run('audio','diagnose_arrangement_context_audio_v6.py',['--experiment',r,'--output',r/'audio-dependence.json'],r/'audio-dependence.json')
 run('review','build_arrangement_context_review_v6.py',['--experiment',r,'--evaluation',r/'evaluation','--output',r/'review-v6'],r/'review-v6/review.json')
 run('pipaxing','predict_arrangement_context_cached_song_v6.py',['--experiment',r,'--evaluation',r/'evaluation','--cached',ROOT/'tools/private/playtests/20260927-pipaxing-v1','--output',r/'pipaxing-review-v6'],r/'pipaxing-review-v6/report.json')
 run('audit','audit_arrangement_results_v6.py',[],r/'acceptance.json')
 status('completed',human_listening='pending',game_acceptance='not_evaluated',production_promoted=False)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(ROOT/'tools/private/community/supervised-dual-v6-20260930/completion-error.json',dict(error=repr(error),time=time.strftime('%Y-%m-%d %H:%M:%S')));raise
