"""Explicit, pinned-revision author checkpoint download, outside inference."""
from pathlib import Path
import hashlib
import json
import urllib.request
ROOT=Path(__file__).resolve().parents[2]/'private'/'separation'
REVISION='ac9b0614ab3cd7f77219e18ba494dfd93956c348'
SHA256='87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
BASE='https://huggingface.co/KimberleyJSN/melbandroformer/resolve/'+REVISION+'/'


def main():
    ROOT.mkdir(parents=True,exist_ok=True);destination=ROOT/'MelBandRoformer.ckpt'
    readme=urllib.request.urlopen(BASE+'README.md',timeout=30).read()
    if b'license: mit' not in readme:raise ValueError('Pinned license declaration differs')
    (ROOT/'roformer-author-README.md').write_bytes(readme)
    if not destination.exists():
        temporary=destination.with_suffix('.download')
        with urllib.request.urlopen(BASE+destination.name,timeout=60) as response,temporary.open('wb') as output:
            size=0;last=0
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>913106900:raise ValueError('Unexpected checkpoint size')
                output.write(chunk)
                if size-last>=64*1024*1024:print(f'{size}/{913106900} bytes',flush=True);last=size
        with temporary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=SHA256:raise ValueError('Author LFS SHA256 mismatch')
        temporary.rename(destination)
    with destination.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=SHA256:raise ValueError('Local checkpoint mismatch')
    manifest={'model_id':'kim-mel-band-roformer','file':destination.name,'sha256':digest,'byte_count':destination.stat().st_size,
        'url':BASE+destination.name,'revision':REVISION,'license_id':'MIT','license_source':BASE+'README.md',
        'license_document_sha256':hashlib.sha256(readme).hexdigest(),'allowed_uses':['inference'],'pretrained_overlap':'unknown'}
    (ROOT/'roformer-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))


if __name__=='__main__':main()
