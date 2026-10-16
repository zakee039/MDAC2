"""Read-only, role-isolated dataset usable by Sol's future trainer."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from .audio_features import validate_cache,sha
from . import training_tensors
from .training_chart import CLASSES,POLICY_ID
from .training_tensors import frame_targets,timing_features,window_starts,slice_window


class WindowDataset:
    def __init__(self,manifest,cache,role,*,allow_locked_test=False,tensor_cache=None,allow_partial_cache=False):
        if role not in ('train','validation','locked_test'):
            raise ValueError('Explicit dataset role required')
        if role=='locked_test' and not allow_locked_test:
            raise ValueError('Locked test must be opened explicitly for final evaluation')
        self.path=Path(manifest);self.root=self.path.parent;self.cache=Path(cache)
        data=json.loads(self.path.read_text(encoding='utf-8'))
        if data['schema_version']!=1 or data['policy']!=POLICY_ID or tuple(data['classes'])!=CLASSES:
            raise ValueError('Explicit compatible label policy/class order required')
        self.tensor_cache=Path(tensor_cache) if tensor_cache is not None else None
        self.target_index={};self.verified_audio=set();self.verified_targets=set()
        feature_report=json.loads((self.cache/'report.json').read_text(encoding='utf-8'))
        if feature_report['status']!='completed' and not allow_partial_cache:
            raise ValueError('Complete feature preparation required before training')
        if not allow_partial_cache and (feature_report.get('manifest_sha256')!=sha(self.path) or feature_report.get('errors')):
            raise ValueError('Feature report identity/errors require preparation audit')
        self.feature_index={r['pcm_sha256']:r for r in feature_report['complete']}
        expected_audio={r['pcm']['sha256'] for r in data['charts'] if r['training_eligible']}
        if not allow_partial_cache and not expected_audio.issubset(self.feature_index):
            raise ValueError('Feature report missing eligible audio')
        if self.tensor_cache is not None:
            tensor_report=json.loads((self.tensor_cache/'report.json').read_text(encoding='utf-8'))
            if tensor_report['manifest_sha256']!=sha(self.path) or tensor_report['implementation_sha256']!=sha(Path(training_tensors.__file__)):
                raise ValueError('Stale tensor cache')
            self.target_index={r['chart_sha256']:r for r in tensor_report['charts']}
        self.rows=[r for r in data['charts'] if r['split']==role and r['training_eligible']]
        self.index=[];self.loaded=OrderedDict()
        for i,row in enumerate(self.rows):
            label=self.root/row['labels']
            if sha(label)!=row['labels_sha256']:raise ValueError('Labels changed')
            chart=json.loads(label.read_text(encoding='utf-8'))
            count=int(np.ceil(chart['duration']*100-1e-8))
            self.index.extend((i,start) for start in window_starts(count))

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        row_index,start=self.index[index];row=self.rows[row_index]
        if row_index not in self.loaded:
            label_path=self.root/row['labels']
            if sha(label_path)!=row['labels_sha256']:raise ValueError('Labels changed after dataset initialization')
            chart=json.loads(label_path.read_text(encoding='utf-8'))
            if not chart['training_eligible']:raise ValueError('Unsafe labels in eligible manifest row')
            if chart['policy']!=POLICY_ID or chart['chart_sha256']!=row['chart_sha256']:
                raise ValueError('Chart label policy/identity mismatch')
            if self.tensor_cache is None:
                target=frame_targets(chart)
            else:
                record=self.target_index[row['chart_sha256']]
                tensor_path=self.tensor_cache/record['file']
                if record['labels_sha256']!=row['labels_sha256']:raise ValueError('Tensor label identity mismatch')
                if row_index not in self.verified_targets:
                    if sha(tensor_path)!=record['sha256']:raise ValueError('Corrupt target tensors')
                    self.verified_targets.add(row_index)
                with np.load(tensor_path,allow_pickle=False) as archive:
                    target={k:archive[k] for k in archive.files}
            pcm=row['pcm']['sha256'];path=self.cache/pcm
            if sha(path/'metadata.json')!=self.feature_index[pcm]['metadata_sha256']:
                raise ValueError('Feature metadata identity mismatch')
            meta=validate_cache(path,row,verify_hashes=pcm not in self.verified_audio)
            if meta['frame_seconds']!=.01 or meta['feature_origin_seconds']!=0 or meta['feature_shape'][:2]!=[3,96]:
                raise ValueError('Feature temporal/stream contract mismatch')
            self.verified_audio.add(pcm)
            features=np.load(path/'features.npy',mmap_mode='r',allow_pickle=False)
            if abs(features.shape[-1]-len(target['onsets']))>1:
                raise ValueError('Feature/target duration mismatch')
            timing=target.pop('timing',None)
            if timing is None:timing=timing_features(chart,row['rating'],features.shape[-1])
            self.loaded[row_index]=(features,timing,target)
            while len(self.loaded)>2:self.loaded.popitem(last=False)
        self.loaded.move_to_end(row_index)
        features,timing,target=self.loaded[row_index]
        audio,valid=slice_window(features,start,time_axis=2)
        beat,_=slice_window(timing,start)
        answer={'features':audio.astype(np.float32),'timing':beat,'loss_region':valid}
        for name,array in target.items():
            if name=='collisions':continue
            value,_=slice_window(array,start)
            if 'mask' in name:value &= valid.reshape((-1,)+(1,)*(value.ndim-1))
            answer[name]=value
        # Deliberately no title, source code, author, label identity or event ids
        # among model inputs; the caller may retrieve provenance separately.
        return answer
