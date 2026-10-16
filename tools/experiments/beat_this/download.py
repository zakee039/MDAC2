from pathlib import Path
import hashlib,json,urllib.request
root=Path('tools/private/beat-this')
records=[]
for name in ('final0','small0'):
    url=f'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/{name}.ckpt'
    target=root/(name+'.ckpt')
    if not target.exists():
        with urllib.request.urlopen(url,timeout=60) as response,target.with_suffix('.partial').open('wb') as out:
            while chunk:=response.read(1024*1024):out.write(chunk)
        target.with_suffix('.partial').rename(target)
    with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    records.append({'model_id':name,'relative_path':target.name,'sha256':digest,'byte_count':target.stat().st_size,'source':url,'license_id':'MIT','allowed_uses':['inference'],'license_source':'https://github.com/CPJKU/beat_this#license'})
    print(name,target.stat().st_size,digest,flush=True)
(root/'download-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
