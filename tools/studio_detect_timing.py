"""Bounded local BeatThis child for the automatic studio; no network loader."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from mdac2.infrastructure.timing_models import TimingModels
    from mdac2.application.studio_timing import fit_beats
    from mdac2.data_preparation.audio_features import sha
    from mdac2.adapters.beat_this_local import LocalBeatThisBackend
    from mdac2.application.studio_cover import binary
    _,registry,producer=TimingModels(ROOT).configuration('final0',check_dependencies=True)
    checkpoint,_=registry.require('final0')
    assert sha(args.audio)==args.sha,'Audio identity changed'
    # The existing BeatThis runtime does not include soundfile. Reuse bundled FFmpeg.
    rate=44100;channels=2
    pcm=subprocess.run([str(binary(ROOT,'ffmpeg.exe')),'-v','error','-nostdin','-protocol_whitelist','file,pipe',
        '-i',str(args.audio),'-t','601','-ar',str(rate),'-ac',str(channels),'-f','s16le','pipe:1'],
        capture_output=True,check=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
    if len(pcm)%4 or not 0<len(pcm)/4/rate<=600:raise ValueError('Invalid audio')
    def forbidden(*a,**kw):raise RuntimeError('Network forbidden for local timing inference')
    socket.socket=forbidden
    result=LocalBeatThisBackend().infer(checkpoint=checkpoint,pcm=pcm,sample_rate=rate,channels=channels,
        options=dict(device='cpu',precision='float32',postprocessor='minimal'),include_logits=False)
    assert result['model_hash']==producer['model_hash'] and sha(args.audio)==args.sha
    summary=fit_beats(result['beat_times'])
    summary.update(producer=producer,source_sha256=args.sha,downbeat_times=result['downbeat_times'])
    args.output.write_text(json.dumps(summary),encoding='utf-8')


if __name__=='__main__':main()
