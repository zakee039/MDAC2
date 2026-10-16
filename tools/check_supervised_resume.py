"""Actual CUDA/dropout/AdamW cancel-resume equivalence on a tiny fixture."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation import supervised_training as s


class TinyDataset:
    rows = [dict(),dict()]
    index = [(0,-800),(0,800),(1,-800),(1,800)]
    def __init__(self,*args,**kwargs):
        pass
    def __len__(self):
        return 4
    def __getitem__(self,index):
        rng = np.random.default_rng(index)
        onset = np.zeros((32,2,6),np.float32)
        onset[16,0,0] = 1
        ends = np.zeros((32,2),np.float32)
        return dict(features=rng.random((3,96,32),dtype=np.float32), timing=np.zeros((32,4),np.float32),
            onsets=onset, offsets=onset*.3, onset_mask=np.ones_like(onset,bool), offset_mask=onset>0,
            hold_ends=ends, hold_end_offsets=ends, hold_end_mask=np.ones_like(ends,bool),
            hold_end_offset_mask=ends>0, occupancy=ends, occupancy_mask=np.ones_like(ends,bool))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        diagnostic = root/'gate.json'
        s.atomic_json(diagnostic,dict(passed=True,identity={'fixture':'resume-cuda-v1'}))
        config = dict(manifest='fixture',cache='fixture',tensors='fixture',device='cuda',amp=True,
            seed=123,batch_size=1,learning_rate=.001,weight_decay=.0001,max_epochs=1,patience=2,
            min_delta=.0005,log_every=1,gradient_clip=1.,overfit_report=str(diagnostic))
        balance = dict(positive_weights=torch.ones(6,device='cuda'),hold_end_positive_weights=torch.ones(2,device='cuda'))
        with patch.object(s,'WindowDataset',TinyDataset), patch.object(s,'data_identity',return_value={'fixture':'resume-cuda-v1'}), patch.object(s,'weights',return_value=balance):
            s.train(config,root/'continuous')
            original = torch.optim.AdamW.step
            calls = [0]
            def cancelling_step(optimizer,*args,**kwargs):
                value = original(optimizer,*args,**kwargs)
                calls[0] += 1
                if calls[0] == 2:
                    (root/'resumed'/'CANCEL').touch()
                return value
            with patch.object(torch.optim.AdamW,'step',cancelling_step):
                first = s.train(config,root/'resumed')
            assert first['status'] == 'cancelled' and first['cursor'] == 2
            (root/'resumed'/'CANCEL').unlink()
            s.train(config,root/'resumed',root/'resumed'/'last.pt')
        a = torch.load(root/'continuous'/'last.pt',weights_only=False,map_location='cpu')
        b = torch.load(root/'resumed'/'last.pt',weights_only=False,map_location='cpu')
        assert a['state'] == b['state']
        assert all(torch.equal(a['model'][name],b['model'][name]) for name in a['model'])
        for key, value in a['optimizer']['state'].items():
            for name, tensor in value.items():
                assert torch.equal(tensor,b['optimizer']['state'][key][name])
        # A changed identity must fail before any resumed optimizer update.
        with patch.object(s,'WindowDataset',TinyDataset), patch.object(s,'data_identity',return_value={'fixture':'changed'}):
            try:
                s.train(config,root/'resumed',root/'resumed'/'last.pt')
            except ValueError:
                pass
            else:
                raise AssertionError('Changed data identity was accepted')
        report = dict(passed=True,cancel_at_step=2,total_steps=4,model_bitwise_equal=True,
            optimizer_bitwise_equal=True,identity_change_rejected=True,device='cuda',
            full_training_inputs_used=False,locked_test_opened=False,trainer_sha256=s.sha(Path(s.__file__)))
        output = Path(sys.argv[1])
        s.atomic_json(output,report)
        print(report)


if __name__ == '__main__':
    main()
