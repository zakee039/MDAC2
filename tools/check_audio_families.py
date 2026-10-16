"""Local full-song acoustic duplicate candidates; no training/model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.signal import stft

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.community import digest,write_json


def fingerprint(audio):
    _,_,z=stft(audio,fs=8000,nperseg=512,noverlap=112,boundary='zeros')
    power=np.abs(z).astype(np.float64)**2
    # Group linear spectral bins into logarithmically spaced frequency bands.
    edges=np.unique(np.round(np.geomspace(1,power.shape[0],49)).astype(int))
    bands=np.stack([np.log1p(10000*power[a:b].mean(axis=0)) for a,b in zip(edges[:-1],edges[1:])])
    original=np.linspace(0,1,bands.shape[1]);target=np.linspace(0,1,128)
    fixed=np.stack([np.interp(target,original,band) for band in bands])
    fixed-=fixed.mean(axis=1,keepdims=True)
    flat=fixed.flatten();norm=np.linalg.norm(flat)
    return (flat/norm if norm>1e-8 else np.zeros_like(flat)).astype(np.float32)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    unique={r['pcm']['sha256']:r for r in manifest['charts']};rows=list(unique.values());vectors=[]
    ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    for i,r in enumerate(rows,1):
        path=Path(r['audio_path_absolute'])
        if digest(path)!=r['audio_sha256']:raise ValueError('Source changed')
        result=subprocess.run([str(ffmpeg),'-v','error','-i',str(path),'-map','0:a:0','-ac','1',
            '-ar','8000','-f','f32le','pipe:1'],capture_output=True,check=True,timeout=60)
        signal=np.frombuffer(result.stdout,dtype='<f4')
        if not np.isfinite(signal).all():raise ValueError('Nonfinite fingerprint audio')
        vectors.append(fingerprint(signal))
        if i%50==0:print(f'Acoustic fingerprints {i}/{len(rows)}',flush=True)
    vectors=np.stack(vectors);np.save(a.output/'fingerprints.npy',vectors,allow_pickle=False)
    correlation=vectors@vectors.T;pairs=[]
    for i in range(len(rows)):
        for j in range(i):
            duration_delta=abs(rows[i]['pcm']['seconds']-rows[j]['pcm']['seconds'])
            if correlation[i,j]>=.98 and duration_delta<=1.:
                pairs.append({'a':rows[i]['pcm']['sha256'],'b':rows[j]['pcm']['sha256'],
                    'similarity':float(correlation[i,j]),'duration_delta':duration_delta,
                    'family_a':rows[i]['provisional_family'],'family_b':rows[j]['provisional_family'],
                    'role_a':rows[i]['split'],'role_b':rows[j]['split'],
                    'title_a':rows[i]['title'],'title_b':rows[j]['title']})
    report={'schema_version':1,'manifest_sha256':digest(a.manifest),'source_code_sha256':digest(Path(__file__)),
            'audio_order':[r['pcm']['sha256'] for r in rows],'fingerprints_sha256':digest(a.output/'fingerprints.npy'),
            'threshold':.98,'max_duration_delta':1.,'pairs':pairs,
            'cross_role_pairs':sum(x['role_a']!=x['role_b'] for x in pairs),
            'scope':'near-identical full recordings; does not certify covers/remixes are independent',
            'training_started':False}
    write_json(a.output/'report.json',report)
    print(json.dumps({'recordings':len(rows),'pairs':pairs,'cross_role_pairs':report['cross_role_pairs']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
