"""Separate composition errors from orientation errors at reference onsets."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_hierarchy_v4 import GROUPS,GROUP_IDS,ASYMMETRIC,mirror
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.audio_features import sha

def main(root,output):
    if output.exists():raise ValueError('Preserve existing diagnostic')
    config=json.loads((root/'arrangement.json').read_text(encoding='utf-8'));derived=Path(config['derived'])
    evaluation=json.loads((root/'evaluation/validation.json').read_text(encoding='utf-8'))
    group=np.asarray(GROUP_IDS);asymmetric=np.asarray(ASYMMETRIC);mirrors=np.asarray([PATTERNS.index(mirror(p)) for p in PATTERNS])
    matrix=np.zeros((len(GROUPS),len(GROUPS)),np.int64);counts=np.zeros(5,np.int64)
    for path in sorted((root/'evaluation/scores-reference_onsets').glob('*.npz')):
        if sha(path)!=evaluation['score_cache_sha256'][path.relative_to(root/'evaluation').as_posix()]:raise ValueError('Score identity changed')
        with np.load(path) as a:s=a['logits'];times=a['times']
        with np.load(derived/path.name) as a:truth=a['tokens'];valid=a['valid'];assert np.array_equal(times,a['times'])
        pred=s.argmax(1);np.add.at(matrix,(group[truth[valid]],group[pred[valid]]),1)
        take=np.flatnonzero(valid&asymmetric[truth]);t=truth[take];other=mirrors[t]
        direction=s[take,t]>s[take,other];ties=s[take,t]==s[take,other]
        counts+=np.array([valid.sum(),(valid&(truth==pred)).sum(),len(take),direction.sum(),ties.sum()],np.int64)
    classes={}
    for i,g in enumerate(GROUPS):
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum()-tp);fn=int(matrix[i].sum()-tp)
        classes[str(g)]=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn),reference_count=int(matrix[i].sum()),predicted_count=int(matrix[:,i].sum()))
    light=GROUPS.index((0,1,0));gemini=GROUPS.index((5,5,0));double=GROUPS.index((2,2,0))
    atomic_json(output,dict(role='validation_reference_onsets',valid_events=int(counts[0]),raw_combo_accuracy=float(counts[1]/max(1,counts[0])),
        composition_accuracy=float(np.trace(matrix)/max(1,matrix.sum())),orientation_with_correct_type_accuracy=float(counts[3]/max(1,counts[2])),orientation_ties=int(counts[4]),
        orientation_events=int(counts[2]),groups=[list(g) for g in GROUPS],confusion=matrix.tolist(),per_group=classes,
        gemini_as_single_light=int(matrix[gemini,light]),single_light_as_gemini=int(matrix[light,gemini]),gemini_as_double=int(matrix[gemini,double]),double_as_gemini=int(matrix[double,gemini]),
        caveat='Ranking includes a minimal boost binding the actual history-feedback legal choice; oracle-type direction is diagnostic, not full-chain success or calibrated probability',locked_test_opened=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.experiment.resolve(),a.output.resolve())
