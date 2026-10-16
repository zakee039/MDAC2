from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.arrangement_context_v4 import decode
from mdac2.data_preparation.arrangement_gemini_margin_v5 import decode_uncertain_gemini,GEMINI
from mdac2.data_preparation.supervised_training import atomic_json
shapes=[(0,0,1),(5,5,0),(5,5,0),(3,0,0),(2,2,0)]
s=np.full((5,len(PATTERNS)),-20.);times=[0.,2.,4.,6.,8.];lengths=np.zeros((5,2));lengths[0,0]=np.log1p(1.)
for i,p in enumerate(shapes):s[i,PATTERNS.index(p)]=2.
s[1,PATTERNS.index((1,0,0))]=1.8;s[2,PATTERNS.index((0,1,0))]=0.
a,_=decode(s,lengths,times,10.);b,b0=decode_uncertain_gemini(s,lengths,times,10.,0.)
assert [e.to_dict() for e in a]==[e.to_dict() for e in b]
c,report=decode_uncertain_gemini(s,lengths,times,10.,.5)
assert report['gemini_margin']['changes']==1
assert len([e for e in c if e.kind=='gemini'])==1
assert any(e.start==2 and e.kind=='light' and e.lanes==('ground',) for e in c)
assert [e.to_dict() for e in a if e.start!=2]==[e.to_dict() for e in c if e.start!=2]
assert sorted({e.start for e in a})==sorted({e.start for e in c})
atomic_json(Path(sys.argv[1]),dict(passed=True,zero_margin_exact=True,strong_gemini_preserved=True,hold_special_double_preserved=True,onsets_unchanged=True,only_weak_gemini_changed=True))
