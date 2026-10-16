"""Analytic loss/mask checks in the isolated Torch runtime, without gradients."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from mdac2.data_preparation.joint_losses import joint_losses
    from mdac2.data_preparation.community import digest,write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    onset=torch.zeros((1,3,2,6));tail=torch.zeros((1,3,2))
    out={'onset_logits':onset.clone(),'offsets':onset.clone(),
         'hold_end_logits':tail.clone(),'hold_end_offsets':tail.clone(),'occupancy_logits':tail.clone()}
    target={'onsets':onset.clone(),'offsets':onset.clone(),'onset_mask':torch.zeros_like(onset,dtype=torch.bool),
            'offset_mask':torch.zeros_like(onset,dtype=torch.bool),
            'hold_ends':tail.clone(),'hold_end_offsets':tail.clone(),
            'hold_end_mask':torch.zeros_like(tail,dtype=torch.bool),
            'hold_end_offset_mask':torch.zeros_like(tail,dtype=torch.bool),
            'occupancy':tail.clone(),'occupancy_mask':torch.zeros_like(tail,dtype=torch.bool)}
    checks=[]
    with torch.inference_mode():
        empty=joint_losses(out,target)
        assert all(float(v)==0 for v in empty.values());checks.append('empty masks give finite zero')
        target['onsets'][0,1,0,2]=1;target['onset_mask'][0,1,0,2]=True
        weights=torch.ones(6);weights[2]=4
        losses=joint_losses(out,target,positive_weights=weights)
        assert math.isclose(float(losses['onsets']),4*math.log(2),rel_tol=1e-6)
        checks.append('positive weight applies only to intended class')
        target['onsets'][0,1,0,2]=0
        losses=joint_losses(out,target,positive_weights=weights)
        assert math.isclose(float(losses['onsets']),math.log(2),rel_tol=1e-6)
        checks.append('negative label is not multiplied by positive weight')
        out['onset_logits'][0,0]=1000
        changed=joint_losses(out,target,positive_weights=weights)
        assert float(changed['total'])==float(losses['total'])
        checks.append('ignored context logits cannot change masked loss')
        target['hold_ends'][0,1,1]=1;target['hold_end_mask'][0,1,1]=True
        losses=joint_losses(out,target,hold_end_positive_weights=torch.tensor([2.,3.]))
        assert math.isclose(float(losses['hold_ends']),3*math.log(2),rel_tol=1e-6)
        checks.append('hold tail positive weights broadcast by lane')
        for bad in (torch.zeros(6),torch.full((6,),float('nan')),torch.ones(5)):
            try:joint_losses(out,target,positive_weights=bad)
            except ValueError:pass
            else:raise AssertionError('Invalid weight accepted')
        checks.append('invalid weights rejected')
    result={'passed':len(checks),'checks':checks,'training_started':False,'backward_called':False,
            'loss_code_sha256':digest(ROOT/'src/mdac2/data_preparation/joint_losses.py'),
            'check_code_sha256':digest(Path(__file__))}
    write_json(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
