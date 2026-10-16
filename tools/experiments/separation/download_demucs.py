"""Explicit setup-only download. Never imported by the inference backend."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[2]/'private'/'separation'
URL='https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    destination=ROOT/'955717e8-8726e21a.th'
    if not destination.exists():
        temporary=destination.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=60) as response,temporary.open('wb') as output:
            size=0
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>500*1024*1024:raise ValueError('Unexpected checkpoint size')
                output.write(chunk)
        with temporary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if not digest.startswith('8726e21a'):raise ValueError('Upstream filename checksum mismatch')
        temporary.rename(destination)
    with destination.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if not digest.startswith('8726e21a'):raise ValueError('Local checkpoint checksum mismatch')
    sources={}
    for name in ('LICENSE','README.md','demucs/remote/htdemucs.yaml','demucs/remote/files.txt'):
        url='https://raw.githubusercontent.com/facebookresearch/demucs/main/'+name
        data=urllib.request.urlopen(url,timeout=30).read();path=ROOT/('source-'+name.replace('/','-'))
        path.write_bytes(data);sources[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'local_file':path.name}
    manifest={'model_id':'htdemucs-955717e8','file':destination.name,'sha256':digest,'byte_count':destination.stat().st_size,
        'url':URL,'license_id':'MIT','license_basis':'Official Demucs distribution README and LICENSE; snapshots retained.',
        'allowed_uses':['inference'],'pretrained_overlap':'unknown','upstream_checksum_prefix':'8726e21a','source_documents':sources,
        'notes':['Full SHA256 pins downloaded bytes; upstream only publishes an eight-character checksum in filename.','No training songs downloaded.']}
    (ROOT/'demucs-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(destination),'sha256':digest,'bytes':destination.stat().st_size}))


if __name__=='__main__':main()
