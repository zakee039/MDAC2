"""Use a real prepared training window for a forward-only model/loss check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    import numpy as np
    import torch
    from mdac2.data_preparation.window_dataset import WindowDataset
    from mdac2.data_preparation.joint_model import JointChartModel
    from mdac2.data_preparation.joint_losses import joint_losses
    from mdac2.data_preparation.community import write_json,digest
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--tensor-cache',type=Path)
    p.add_argument('--allow-partial-cache',action='store_true')
    args=p.parse_args();torch.manual_seed(20260927);torch.set_num_threads(4)
    dataset=WindowDataset(args.manifest,args.cache,'train',tensor_cache=args.tensor_cache,
                          allow_partial_cache=args.allow_partial_cache)
    # During ongoing cache preparation choose a completed training song; do not
    # fall through to validation/locked-test data for convenient debugging.
    position=next(i for i,(ri,_) in enumerate(dataset.index)
                  if dataset.rows[ri]['pcm']['sha256'] in dataset.feature_index)
    sample=dataset[position]
    batch={name:torch.from_numpy(value[None]).to(args.device) for name,value in sample.items()}
    model=JointChartModel().to(args.device).eval()
    before=state_hash(model);started=time.perf_counter()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output=model(batch['features'],batch['timing'])
        losses=joint_losses(output,batch)
    if args.device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    if any(not torch.isfinite(x).all() for x in output.values()):raise ValueError('Nonfinite forward output')
    if any(not torch.isfinite(x) for x in losses.values()):raise ValueError('Nonfinite masked loss')
    after=state_hash(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected parameter update/gradient')
    expected={'onset_logits':(1,3200,2,6),'offsets':(1,3200,2,6),'hold_end_logits':(1,3200,2),
              'hold_end_offsets':(1,3200,2),'occupancy_logits':(1,3200,2),'vocal_activity_prior':(1,3200)}
    if any(tuple(output[k].shape)!=shape for k,shape in expected.items()):raise ValueError('Output geometry mismatch')
    report={'schema_version':1,'model_config':model.config,'parameter_count':sum(p.numel() for p in model.parameters()),
            'device':args.device,'seconds':seconds,'manifest_sha256':digest(args.manifest),
            'training_windows':len(dataset),'sample_window_index':position,'sample_role':'train',
            'output_shapes':{k:list(v.shape) for k,v in output.items()},
            'untrained_loss_values':{k:float(v) for k,v in losses.items()},
            'parameters_before_sha256':before,'parameters_after_sha256':after,
            'backward_called':False,'optimizer_created':False,'checkpoint_saved':False,'training_started':False,
            'cuda_peak_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
            'model_code_sha256':digest(ROOT/'src/mdac2/data_preparation/joint_model.py'),
            'loss_code_sha256':digest(ROOT/'src/mdac2/data_preparation/joint_losses.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
