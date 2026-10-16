"""Run one independently checkpointed MDAC2 dual-model stage."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_training_v4 import train,overfit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--overfit',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if c['device']!='cuda' or c['max_epochs']<=0 or c['patience']<=0:raise ValueError('Bounded CUDA configuration required')
    overfit(c,a.run_dir,a.steps) if a.overfit else train(c,a.run_dir,a.resume)
