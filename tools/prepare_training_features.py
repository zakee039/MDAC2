"""Resumable offline GPU inference/cache preparation; no supervised training."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import OfflineFeatureBuilder,config_hash,validate_cache
from mdac2.data_preparation.community import digest,write_json


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--smoke-seconds',type=float)
    p.add_argument('--include-quarantined',action='store_true')
    args=p.parse_args();raw=args.manifest.read_bytes();manifest=json.loads(raw)
    unique={}
    for row in manifest['charts']:
        if row['training_eligible'] or args.include_quarantined:
            unique.setdefault(row['pcm']['sha256'],row)
    rows=sorted(unique.values(),key=lambda r:(r['split']!='train',r['pcm']['sha256']))
    if args.limit:rows=rows[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    def denied(*a,**k):raise RuntimeError('Feature inference network disabled')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=denied
    engine=OfflineFeatureBuilder(checkpoint=ROOT/'tools/private/separation/MelBandRoformer.ckpt',
        config_path=ROOT/'tools/private/separation/roformer-author-config.yaml',
        ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe')
    report={'schema_version':1,'manifest_sha256':digest(args.manifest),'config_sha256':config_hash(),
            'environment':engine.environment,'planned':len(rows),'complete':[],'errors':[],
            'training_started':False,'status':'running','started_at_unix':time.time(),
            'scope':'smoke_excerpt' if args.smoke_seconds else ('limited' if args.limit else
                ('all_selected_audio' if args.include_quarantined else 'all_eligible_audio'))}
    for i,row in enumerate(rows,1):
        key=row['pcm']['sha256'];target=args.output/key;stage=args.output/(key+'.staging')
        try:
            if digest(Path(row['audio_path_absolute']))!=row['audio_sha256']:
                raise ValueError('Source audio changed')
            if target.exists():
                metadata=validate_cache(target,row)
                if metadata['environment']!=engine.environment:
                    raise ValueError('Feature inference environment changed')
            else:
                if stage.exists():raise ValueError('Existing incomplete staging; inspect before retry')
                metadata=engine.process(row,stage,args.smoke_seconds)
                os.replace(stage,target)
            report['complete'].append({'pcm_sha256':key,'path':target.resolve().as_posix(),
                'metadata_sha256':digest(target/'metadata.json'),'diagnostics':metadata['diagnostics']})
            print(json.dumps({'done':i,'total':len(rows),'title':row['title'],**metadata['diagnostics']},ensure_ascii=False),flush=True)
        except Exception as exc:
            report['errors'].append({'pcm_sha256':key,'error':type(exc).__name__+': '+str(exc)})
            print(json.dumps(report['errors'][-1]),flush=True)
        write_json(args.output/'report.json',report)
    if args.manifest.read_bytes()!=raw:raise ValueError('Manifest changed during preparation')
    report['status']='completed' if not report['errors'] else 'completed_with_errors'
    report['finished_at_unix']=time.time();write_json(args.output/'report.json',report)
    return int(bool(report['errors']))


if __name__=='__main__':raise SystemExit(main())
