"""Read-only hash verification of copied and expanded preparation assets."""
import argparse
import json
from pathlib import Path
from mdac2.data_preparation.community import digest


def verify(root):
    root=root.resolve(strict=True)
    report=json.loads((root/'inventory.json').read_text(encoding='utf-8'))
    checked={'copied_files':0,'expanded_files':0}
    pairs=[('copied_files',root/'copied',report['files'])]
    pairs.extend(('expanded_files',root/'expanded'/p['sha256'],p['files'])
                 for p in report['packages'] if p['status']=='extracted')
    for key,base,files in pairs:
        for item in files:
            path=(base/item.get('relative_path',item.get('path'))).resolve()
            if base.resolve() not in path.parents:
                raise ValueError('Unsafe inventory path')
            if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('Hash/length mismatch: '+str(path))
            checked[key]+=1
    if any(p['status']!='extracted' for p in report['packages']):
        raise ValueError('Incomplete package extraction')
    return checked


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(verify(parser.parse_args().root)))
