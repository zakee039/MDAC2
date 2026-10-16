"""Executable contracts for masking, independent inputs, groups and long holds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.dual_data import derive,PATTERNS,BOS
from mdac2.data_preparation.dual_training import make_model
from mdac2.data_preparation.dual_inference import pattern_events,arrange,acoustic_predictions
from mdac2.data_preparation.supervised_training import atomic_json,seed_all
from mdac2.data_preparation.generated_constraints import validate_generated_events

def main(output):
    seed_all(20260928)
    events=[dict(kind='hold',lanes=['ground'],start=.105,end=20.5),
            dict(kind='light',lanes=['ground'],start=.105,end=.105),
            dict(kind='gemini',lanes=['ground','air'],start=.3,end=.3),
            dict(kind='heavy',lanes=['ground'],start=.5,end=.5),
            dict(kind='heavy',lanes=['air'],start=.5,end=.5),
            dict(kind='light',lanes=['ground'],start=.7,end=.7),
            dict(kind='heavy',lanes=['ground'],start=.7,end=.7)]
    original=dict(onsets=np.zeros((2100,2,6),np.float32),onset_mask=np.ones((2100,2,6),bool))
    original['onsets'][[10,30,50,70],0,0]=1;original['onset_mask'][90]=False
    lab=derive(dict(duration=21.,events=events),original)
    assert lab['frames'].tolist()==[10,30,50,70]
    assert PATTERNS[lab['tokens'][0]]==(1,0,1)
    assert PATTERNS[lab['tokens'][1]]==(5,5,0)
    assert PATTERNS[lab['tokens'][2]]==(2,2,0)
    assert not lab['valid'][3] and not lab['mask'][90]
    assert lab['previous'].tolist()==[BOS,*lab['tokens'][:3]]
    assert np.isclose(np.expm1(lab['context'][1,2]),20.2)
    gemini=pattern_events(PATTERNS.index((5,5,0)),[0,0],1.,30.,1)
    double=pattern_events(PATTERNS.index((2,2,0)),[0,0],2.,30.,2)
    held=pattern_events(PATTERNS.index((1,0,1)),[20.,0],15.,40.,3)
    assert len(gemini)==1 and gemini[0].kind=='gemini'
    assert len(double)==2 and double[0].group_id==double[1].group_id
    assert any(e.kind=='hold' and e.end==35 for e in held)
    assert not validate_generated_events(gemini+double+held,40.)
    onset=make_model('onset');layout=make_model('arrangement');onset.eval();layout.eval()
    assert 3e6<=sum(p.numel() for p in onset.parameters())<=5e6
    assert 5e6<=sum(p.numel() for p in layout.parameters())<=10e6
    for model,streams in ((onset,3),(layout,2)):
        try:
            if model is onset:model(torch.zeros(1,streams,96,32))
            else:model.encode(torch.zeros(1,streams,96,32))
        except ValueError:pass
        else:raise AssertionError('Incorrect input stream signature accepted')
    class ConstantOnset:
        def eval(self):pass
        def __call__(self,x):return dict(logits=x[:,0,0],offsets=torch.zeros_like(x[:,0,0]))
    class ConstantArrangement:
        def eval(self):pass
        def encode(self,x):return x[:,0,0,:,None].expand(-1,-1,128)
    features=np.zeros((3,96,4000),np.float32);features[:,0,:]=np.arange(4000)[None,:]/4000
    pred=acoustic_predictions(ConstantOnset(),ConstantArrangement(),features,4000,device='cpu',amp=False)
    assert np.allclose(pred['acoustic'][:,0],features[0,0])
    assert np.allclose(pred['probability'],torch.tensor(features[1,0]).sigmoid().numpy())
    atomic_json(output,dict(passed=True,label_union=True,ambiguous_pattern_masked=True,excluded_negative_masked=True,
         shifted_history_no_current_token=True,long_hold_crosses_windows=True,gemini_double_distinct=True,
         hold_plus_click_preserved=True,wrong_input_streams_rejected=True,unique_center_stitch=True,
         bpm_inputs=False,grid=False,locked_test_opened=False))
    print('Dual contracts passed')

if __name__=='__main__':main(Path(sys.argv[1]))
