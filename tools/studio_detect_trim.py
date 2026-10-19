"""Isolated, time-bounded acoustic structure suggestion; no model/API calls."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdac2.application.studio_trim import suggest
from mdac2.data_preparation.audio_features import sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert sha(a.audio)==a.sha
    result=suggest(a.audio)
    assert sha(a.audio)==a.sha
    result.update(source_sha256=a.sha)
    a.output.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
