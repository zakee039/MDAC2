import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.arrangement_lane_prior_v3 import apply_lane_prior,decode_with_lane_prior
from mdac2.data_preparation.supervised_training import atomic_json
p=PATTERNS.index((1,0,0));q=PATTERNS.index((0,1,0));dual=PATTERNS.index((5,5,0))
a=np.full((100,len(PATTERNS)),-20.);a[:,p]=2.;a[:,q]=2.-np.linspace(.001,.5,100)
z,zero=apply_lane_prior(a,0);assert np.array_equal(z,a)
b,info=apply_lane_prior(a,1);assert info['changed_orientation_count']>0 and info['after']['dual_positions']==0
strong=a.copy();strong[:,q]=-20.;b,info2=apply_lane_prior(strong,1);assert np.array_equal(b.argmax(1),strong.argmax(1))
a[:10,dual]=30.;b,info3=apply_lane_prior(a,1);assert np.all(b[:10].argmax(1)==dual)
for token in (PATTERNS.index((1,0,1)),PATTERNS.index((3,0,0)),PATTERNS.index((2,2,0))):
    x=np.full((10,len(PATTERNS)),-20.);x[:,token]=20.;y,_=apply_lane_prior(x,1);assert np.all(y.argmax(1)==token)
times=np.linspace(.1,9.9,100);lengths=np.zeros((100,2));events,audit=decode_with_lane_prior(a,lengths,times,10.,1.)
assert len(events)==100 and sum(e.kind=='gemini' for e in events)==10 and not audit['rule_issues']
assert sorted(e.start for e in events)==sorted(round(float(t),3) for t in times)
atomic_json(Path(sys.argv[1]),dict(passed=True,zero_exact=True,weak_orientation_can_change=True,strong_confidence_preserved=True,
    gemini_preserved=True,weak_case=info,strong_case=info2,locked_test_opened=False))
