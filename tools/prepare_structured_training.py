"""Read only training-role tensors to describe conditional pattern supervision."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.structured_joint import PATTERNS,structured_targets
from mdac2.data_preparation.audio_features import sha


def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'tools/private/community/supervised-joint-v1-20260927/baseline-lr3e4.json').read_text(encoding='utf-8'))
    # Contract probe: single, double, Gemini, hold+click, and ambiguous stacking.
    target={'onsets':torch.zeros(1,5,2,6),'onset_mask':torch.ones(1,5,2,6,dtype=torch.bool)}
    target['onsets'][0,0,0,0]=1
    target['onsets'][0,1,:,1]=1
    target['onsets'][0,2,:,3]=1
    target['onsets'][0,3,1,[0,4]]=1
    target['onsets'][0,4,0,[0,1]]=1
    before=target['onsets'].clone()
    d=structured_targets(target)
    assert d['pattern_mask'].tolist()==[[True,True,True,True,False]]
    assert [PATTERNS[i] for i in d['pattern_label'][0,:4].tolist()]==[(1,0),(2,2),(5,5),(0,1)]
    assert torch.equal(before,target['onsets'])
    manifest=json.loads(Path(config['manifest']).read_text(encoding='utf-8'))
    records=json.loads((Path(config['tensors'])/'report.json').read_text(encoding='utf-8'))['charts']
    permitted={r['chart_sha256'] for r in manifest['charts'] if r['training_eligible'] and r['split']=='train'}
    counts=np.zeros(len(PATTERNS),np.int64)
    valid=positive=ambiguous=0
    tensor_ids={}
    torch.set_num_threads(4)
    for rec in records:
        if rec['chart_sha256'] not in permitted:
            continue
        path=Path(config['tensors'])/rec['file']
        if sha(path)!=rec['sha256']:
            raise ValueError('Training tensor changed')
        with np.load(path,allow_pickle=False) as npz:
            data={k:torch.from_numpy(npz[k]) for k in ('onsets','onset_mask')}
        t=structured_targets(data)
        valid+=int(t['event_mask'].sum())
        positive+=int(((t['event_target']>0)&t['event_mask']).sum())
        ambiguous+=int(t['ambiguous_pattern'].sum())
        counts+=torch.bincount(t['pattern_label'][t['pattern_mask']],minlength=len(PATTERNS)).numpy()
        tensor_ids[rec['chart_sha256']]=rec['sha256']
    assert len(tensor_ids)==len(permitted)==172
    stats=dict(role='train',manifest_sha256=sha(Path(config['manifest'])),charts=len(permitted),
        event_valid_frames=valid,event_positive_frames=positive,event_positive_weight=math.sqrt((valid-positive)/positive),
        patterns=[dict(pattern=list(p),count=int(n)) for p,n in zip(PATTERNS,counts)],ambiguous_pattern_frames=ambiguous,
        contract_probe_passed=True,source_tensors=tensor_ids,source_modified=False,locked_test_opened=False)
    (output/'target-statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    config.update(learning_rate=.0001,max_epochs=30,patience=7,min_delta=.0001,
        target_statistics=str((output/'target-statistics.json').resolve()),
        overfit_report=str((output/'overfit/report.json').resolve()))
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k!='source_tensors'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
