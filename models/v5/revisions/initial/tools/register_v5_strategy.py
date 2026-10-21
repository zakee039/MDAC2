"""Register only the new V5 strategy; never refresh frozen V2/V3/V4 hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
    def digest(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    def save(name,value):
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    parent=read('recipes/strategies/v4.json')
    # Keep the inherited dependency identities and add only V5-owned adapters.
    sources=dict(parent['postprocessing']['sources'])
    new=[p.relative_to(ROOT).as_posix() for p in (ROOT/'src/mdac2/application/strategies').glob('v5*.py')]
    new+=['src/mdac2/application/studio_strategy_catalog.py',
          'src/mdac2/application/studio_difficulty_v5.py','tools/studio_generate_v5.py']
    for name,item in sources.items():
        if digest(name)!=item['sha256']:raise ValueError('Frozen V4 source changed: '+name)
    sources.update({name:dict(sha256=digest(name)) for name in new})
    strategy=dict(schema_version=1,strategy_id='v5',weight_model='v3',
        parent_manifest=parent['parent_manifest'],parent_manifest_sha256=parent['parent_manifest_sha256'],
        base_strategy='recipes/strategies/v4.json',base_strategy_sha256=digest('recipes/strategies/v4.json'),
        postprocessing={**parent['postprocessing'],'sources':dict(sorted(sources.items()))})
    save('recipes/strategies/v5.json',strategy)
    weights=read('models/v3/manifest.json')
    save('models/v5/manifest.json',dict(schema_version=1,product='MDAC2',strategy_id='v5',weight_model='v3',
        application_version='0.4.0',release_date='2026-10-21',status='optional_strategy_playtest_pending',
        parent_model='v4',parent_manifest_sha256=digest('models/v4/manifest.json'),
        strategy_manifest='recipes/strategies/v5.json',strategy_manifest_sha256=digest('recipes/strategies/v5.json'),
        checkpoints=weights['checkpoints'],postprocessing=strategy['postprocessing'],
        policy=dict(version='studio-v5-sustain-masher-1ms-v1',hold_detection=['vocals','mix'],
            hold_interior='both lanes: masher only',dense_segments='N logical notes -> one masher, target 2N',
            masher_export='original interval; no game-enforced custom hit counter',
            soft_edges='first/last 5 seconds, local power below 3% of song p90',storage_ms=1),
        validation=dict(human_playtest='pending',game_acceptance='pending'),
        version_note='V4 weights and onset/phrase stages; independent sustained-region detection, dense-segment mashers and quiet-edge collectibles. No retraining. V4 stays default.'))
    registry=read('models/registry.json')
    registry['strategies']['v5']=dict(manifest='recipes/strategies/v5.json',weight_model='v3',
        status='optional_strategy_playtest_pending',manifest_sha256=digest('recipes/strategies/v5.json'))
    registry['models']['v5']=dict(manifest='models/v5/manifest.json',application_version='0.4.0',
        status='optional_strategy_playtest_pending',manifest_sha256=digest('models/v5/manifest.json'))
    save('models/registry.json',registry)


if __name__=='__main__':main()
