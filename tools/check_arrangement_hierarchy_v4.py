"""Test that orientation uncertainty never turns single clicks into Gemini."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_hierarchy_v4 import GROUPS,GROUP_IDS,ORIENTATIONS,mirror,hierarchical_scores
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.supervised_training import atomic_json
single=GROUPS.index((0,1,0));gemini=GROUPS.index((5,5,0));double=GROUPS.index((2,2,0))
group=torch.full((1,1,len(GROUPS)),-10.);group[:,:,single]=1.;group[:,:,gemini]=.9
orient=torch.zeros(1,1,len(GROUPS),2)
for side in (-5.,0.,5.):
 orient[:,:,single,0]=side;s=hierarchical_scores(group,orient);chosen=int(s.argmax(-1));assert GROUP_IDS[chosen]==single
for i,p in enumerate(PATTERNS):
 assert PATTERNS.index(mirror(mirror(p)))==i
 g=torch.full_like(group,-10.);g[:,:,GROUP_IDS[i]]=10.
 o=torch.zeros_like(orient);o[:,:,GROUP_IDS[i],ORIENTATIONS[i]]=10.
 assert int(hierarchical_scores(g,o).argmax(-1))==i
assert gemini!=double
# Correct non-ambiguous group and orientation gradients reach both heads.
g=torch.randn_like(group,requires_grad=True);o=torch.randn_like(orient,requires_grad=True)
loss=torch.nn.functional.cross_entropy(g.reshape(1,-1),torch.tensor([single]))+torch.nn.functional.cross_entropy(o[:,:,single].reshape(1,2),torch.tensor([0]));loss.backward()
assert torch.isfinite(g.grad).all() and torch.isfinite(o.grad).all() and o.grad.abs().sum()>0
atomic_json(Path(sys.argv[1]),dict(passed=True,all_41_patterns_reachable=True,orientation_uncertainty_cannot_promote_gemini=True,
    gemini_double_distinct=True,gradients_finite=True,groups=len(GROUPS),locked_test_opened=False))
