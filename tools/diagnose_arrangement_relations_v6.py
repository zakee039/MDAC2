import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.supervised_training import atomic_json

def side(token):
    if token<0:return None
    g,a,h=PATTERNS[token]
    if h:return None
    if g in (1,2) and a==0:return 0
    if a in (1,2) and g==0:return 1
    return None

def main(root,output):
    if output.exists():raise ValueError('Preserve diagnostic')
    config=json.loads((root/'arrangement.json').read_text(encoding='utf-8'));derived=Path(config['derived']);reports={}
    for name,evaluation in [('v4b',ROOT/'tools/private/community/supervised-dual-v4b-20260930/evaluation'),('v6',root/'evaluation'),('v6_without_history',root/'evaluation-without-history')]:
        totals=np.zeros(6,np.int64);charts=[]
        for path in sorted((evaluation/'charts-reference_onsets').glob('*.json')):
            chart=json.loads(path.read_text(encoding='utf-8'));decisions={round(d['start'],3):d['token'] for d in chart['audit']['decisions']}
            with np.load(derived/(path.stem+'.npz')) as a:truth=a['tokens'];times=a['times'];valid=a['valid']
            c=np.zeros(6,np.int64)
            for i in range(1,len(truth)):
                r0=side(int(truth[i-1]));r1=side(int(truth[i]))
                if not(valid[i-1] and valid[i]) or r0 is None or r1 is None:continue
                p0=side(decisions.get(round(float(times[i-1]),3),-1));p1=side(decisions.get(round(float(times[i]),3),-1));switch=r0!=r1
                c[0]+=1;c[1]+=switch
                if p0 is None or p1 is None:c[5]+=1;continue
                ps=p0!=p1;c[2]+=switch and ps;c[3]+=not switch and ps;c[4]+=switch and not ps
            totals+=c;charts.append(dict(chart_sha256=path.stem,counts=c.tolist()))
        n,r,tp,fp,fn,missing=map(int,totals)
        # Missing predicted single types are penalized as switch false negatives where applicable in total recall.
        reports[name]=dict(reference_adjacent_single_pairs=n,reference_switches=r,switch_tp=tp,switch_fp=fp,
            switch_recall=tp/max(1,r),switch_precision=tp/max(1,tp+fp),switch_f1=2*tp/max(1,r+tp+fp),
            predicted_non_single_pairs=missing,type_coverage=(n-missing)/max(1,n),per_chart=charts,
            caveat='Relation may be correct with both absolute lanes wrong; not absolute-lane or end-to-end accuracy')
    atomic_json(output,dict(role='validation',reference_history_input=False,conditions=reports,locked_test_opened=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.experiment.resolve(),a.output.resolve())
