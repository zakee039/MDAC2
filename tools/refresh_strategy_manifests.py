"""Explicitly register current studio policies without rewriting frozen V2/V3."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mdac2 import __version__


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def refresh():
    destination = ROOT / 'recipes/strategies'
    destination.mkdir(parents=True, exist_ok=True)
    for sid in ('r2', 'v2', 'v3', 'v4'):
        parent = 'v2' if sid == 'v2' else 'v3'
        base = json.loads((ROOT / f'models/{parent}/manifest.json').read_text(encoding='utf-8'))
        post = base['postprocessing']
        sources = {name for name in post['sources'] if name.startswith('src/')}
        sources.update([
            'src/mdac2/application/studio_models.py',
            'src/mdac2/application/studio_timing.py',
            'src/mdac2/data_preparation/playability_audit.py',
            'tools/studio_generate.py',
        ])
        if sid == 'v4':
            sources.update([
                'src/mdac2/application/strategies/v4.py',
                'src/mdac2/application/studio_timing_v4.py',
                'src/mdac2/data_preparation/global_postprocess_v4.py',
                'src/mdac2/data_preparation/playability_audit_v4.py',
            ])
        else:
            sources.update(p.relative_to(ROOT).as_posix() for p in
                           (ROOT / f'src/mdac2/application/strategies/{sid}').glob('*.py'))
        manifest = dict(
            schema_version=1, strategy_id=sid, weight_model='v2' if sid in ('r2', 'v2') else 'v3',
            parent_manifest=f'models/{parent}/manifest.json',
            parent_manifest_sha256=digest(f'models/{parent}/manifest.json'),
            postprocessing=dict(
                prior_path=post['prior_path'], prior_sha256=post['prior_sha256'],
                sources={name: dict(sha256=digest(name)) for name in sorted(sources)}),
        )
        (destination / f'{sid}.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    # Keep the V4 release registration in sync with its strategy sources.
    path = ROOT / 'models/v4/manifest.json'
    release = json.loads(path.read_text(encoding='utf-8'))
    registered = json.loads((destination / 'v4.json').read_text(encoding='utf-8'))
    release['postprocessing']['sources'] = registered['postprocessing']['sources']
    release['strategy_manifest'] = 'recipes/strategies/v4.json'
    release['application_version'] = __version__
    path.write_text(json.dumps(release, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'models/registry.json'
    registry = json.loads(path.read_text(encoding='utf-8'))
    for sid, spec in registry['strategies'].items():
        spec['manifest'] = f'recipes/strategies/{sid}.json'
        spec['manifest_sha256'] = digest(spec['manifest'])
    registry['models']['v4']['manifest_sha256'] = digest('models/v4/manifest.json')
    registry['models']['v4']['application_version'] = __version__
    path.write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    refresh()
