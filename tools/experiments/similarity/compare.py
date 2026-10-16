"""Offline labeled-feature comparison; never downloads songs or invents labels."""
from pathlib import Path
import argparse
import json
import sys
import time
import tracemalloc
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from mdac2.analysis.similarity_search import search,feature_rows
from mdac2.analysis.phrase_alignment import align
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.structure import Span
from fractions import Fraction as F


def legacy_mel(features):
    """Documented old fallback reconstruction: 8 quarter window / 4 quarter step."""
    import numpy as np
    bars=feature_rows(features)
    if any(b['meter']!=[4,4] for b in bars):raise ValueError('Legacy fixed-window comparison currently restricted to 4/4')
    windows=[]
    for i in range(len(bars)-1):
        a,b=bars[i:i+2]
        if a['span']['end_q']!=b['span']['start_q']:continue
        vector=np.array([v for row in (a,b) for c in row['cells'] for v in c['mel']])
        windows.append((Span(F(a['span']['start_q']),F(b['span']['end_q'])),vector))
    pairs=[]
    for i,(a,x) in enumerate(windows):
        for b,y in windows[i+1:]:
            norm=np.linalg.norm(x)*np.linalg.norm(y)
            if not a.overlaps(b) and norm>0 and float(x@y/norm)>=.82:pairs.append((a,b))
    return pairs


def metrics(predicted,labels):
    def iou(a,b):return float(max(F(0),min(a.end_q,b.end_q)-max(a.start_q,b.start_q))/(max(a.end_q,b.end_q)-min(a.start_q,b.start_q)))
    matches=[]
    for i,(a,b) in enumerate(predicted):
        for j,(c,d) in enumerate(labels):
            score=min(iou(a,c),iou(b,d))
            if score>=.5:matches.append((score,i,j))
    used_p=set();used_l=set();errors=[]
    for _,i,j in sorted(matches,reverse=True):
        if i in used_p or j in used_l:continue
        used_p.add(i);used_l.add(j)
        a,b=predicted[i];c,d=labels[j]
        errors.append(float((abs(a.start_q-c.start_q)+abs(a.end_q-c.end_q)+abs(b.start_q-d.start_q)+abs(b.end_q-d.end_q))/4))
    return {'precision':len(used_p)/len(predicted) if predicted else None,'recall':len(used_l)/len(labels) if labels else None,
        'true_positives':len(used_p),'candidate_count':len(predicted),'label_count':len(labels),
        'mean_boundary_error_quarters':sum(errors)/len(errors) if errors else None,'pair_min_iou':.5}


def compare(features,labels):
    reports={}
    for name in ('legacy_mel_reconstruction','librosa_recurrence','librosa_recurrence_dtw'):
        tracemalloc.start();start=time.perf_counter()
        try:
            if name=='legacy_mel_reconstruction':pairs=legacy_mel(features)
            else:
                candidates=search(features).proposals
                if name.endswith('_dtw'):candidates=tuple(p for p in candidates if align(features,p).accepted)
                pairs=[(p.source,p.target) for p in candidates]
            elapsed=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory()
        finally:tracemalloc.stop()
        reports[name]={**metrics(pairs,labels),'wall_seconds':elapsed,'python_traced_peak_bytes':peak}
    return {'feature_hash':features.artifact_hash,'methods':reports,'limitations':['Legacy fallback reconstruction, not recovered CNN weights.',
        'Python traced memory excludes native library allocations; not process RSS.','Cold import/JIT costs may differ; not a latency ranking.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1 or manifest.get('label_origin')!='human_reviewed' or not manifest.get('reviewer') or not manifest.get('license_basis') or not manifest.get('songs'):raise ValueError('Human reviewed labels, reviewer and license basis required')
    if args.output.exists():raise ValueError('Choose a new report path; existing reports are immutable')
    reports=[]
    for song in manifest['songs']:
        path=(args.manifest.parent/song['features']).resolve()
        features=ArtifactEnvelope.from_json(path.read_bytes())
        if features.artifact_hash!=song['feature_hash']:raise ValueError('Feature identity mismatch')
        labels=[(Span(F(p['source'][0]),F(p['source'][1])),Span(F(p['target'][0]),F(p['target'][1]))) for p in song['positive_pairs']]
        if any(a.overlaps(b) or a.start_q>=b.start_q for a,b in labels):raise ValueError('Ordered nonoverlapping annotated pairs required')
        reports.append({'song_id':song['song_id'],**compare(features,labels)})
    args.output.write_text(json.dumps({'schema_version':1,'reviewer':manifest['reviewer'],'songs':reports},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
