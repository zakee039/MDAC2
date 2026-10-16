import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v6 import ContextArrangement,decode
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS,BOS
from mdac2.data_preparation.supervised_training import seed_all,atomic_json
seed_all(123);m=ContextArrangement().cuda().eval()
z=torch.randn(2,8,1024,device='cuda');g=torch.ones(2,8,2,device='cuda')*.2;t=torch.arange(8,device='cuda')[None].expand(2,-1)*.25;n=torch.tensor([8,8])
a=m.rollout(z,g,t,n);changed=z.clone();changed[:,5:]+=10;b=m.rollout(changed,g,t,n)
assert torch.equal(a['logits'][:,:5],b['logits'][:,:5]),'History layer reads future events'
none=m.rollout(z,g,t,n,False)
assert not torch.equal(a['logits'],none['logits']),'History input has no effect'
loss=a['logits'].square().mean();loss.backward()
assert m.embedding.weight.grad.abs().sum()>0 and m.history.weight_hh.grad.abs().sum()>0
# Long holds persist across conceptual 16-second audio boundaries; clicks remain legal under them.
s=m.state(1,'cuda');hold=PATTERNS.index((0,0,1));click=PATTERNS.index((1,0,0));extra=PATTERNS.index((3,0,0))
s=m.advance(s,torch.ones(1,256,device='cuda'),torch.tensor([hold],device='cuda'),torch.tensor([[np.log1p(30),0]],device='cuda'),torch.tensor([1.],device='cuda'),torch.tensor([True],device='cuda'))
scores=torch.zeros(1,len(PATTERNS),device='cuda');scores[0,hold]=10;scores[0,extra]=9;scores[0,click]=8
assert int(m.choose({'logits':scores},torch.tensor([17.],device='cuda'),s))==click
assert s['active'][0,0]>17
# Explicit Gemini and double-heavy are distinct feedback tokens.
assert PATTERNS.index((5,5,0))!=PATTERNS.index((2,2,0))
assert m.config['teacher_forcing']==0 and not m.config['label_history_input']
atomic_json(Path(sys.argv[1]),dict(passed=True,causal_history=True,history_input_changes_output=True,
    history_gradients_nonzero=True,reference_history_never_input=True,teacher_forcing=0,
    cross_window_hold_state_preserved=True,hold_plus_click_allowed=True,extra_hold_exclusion=True,
    gemini_double_heavy_distinct=True,parameters=sum(p.numel() for p in m.parameters())))
print('history contracts passed')
