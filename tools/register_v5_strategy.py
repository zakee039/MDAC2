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
        application_version='0.5.0',release_date='2026-10-21',status='optional_strategy',
        parent_model='v4',parent_manifest_sha256=digest('models/v4/manifest.json'),
        strategy_manifest='recipes/strategies/v5.json',strategy_manifest_sha256=digest('recipes/strategies/v5.json'),
        checkpoints=weights['checkpoints'],postprocessing=strategy['postprocessing'],
        policy=dict(version='studio-v5-boss-span-1ms-v10',hold_detection=['vocals','mix','weak phrase heads accepted when energy rises','heads pulled back to the detected phrase start'],
            hold_interior='notes allowed; same-lane floor doubled inside one hold',
            hold_edges='head/tail count as ordinary notes; no extra clearance beyond the slider floor',
            hold_reroute='min hold 500 ms; first opposite-lane note 50%, second certain, then settled',
            dense_segments='discarded / retained > 0.8 only; otherwise delete excess',
            discard_to_retain_threshold=.8,boss_merge_gap_ms=500,boss_merge_comparison='strictly_less',
            boss_span='span < 200 ms kept as a fixed 100 ms accent; 200-2000 ms demoted to density discarding; > 2000 ms untouched',
            boss_short_below_ms=200,boss_short_exported_ms=100,boss_demoted_span_ms=[200,2000],
            same_lane_spacing_ms='75-175 slider, default 125',hold_interior_spacing_ms='2x same-lane slider',
            boss_near_hold='forbidden within hold +/- 500 ms',
            masher_target='2N; merged interval source notes counted once',
            masher_export='original interval; no game-enforced custom hit counter',
            soft_edges='first/last 5 seconds, local power below 3% of song p90',storage_ms=1),
        validation=dict(human_playtest='accepted_by_user_2026-10-21',game_acceptance='user_playtested'),
        version_note='V10 admits Bosses by measured duration instead of 2N hits. V9 keeps weak vocal phrase heads (rise test instead of a flat 6% gate) and pulls late hold heads back to the detected phrase start. V4 weights and onset/phrase stages; configurable 75-175 ms same-lane spacing doubled inside holds, probability interior reroute, Bosses kept away from holds, hold head/tail as ordinary notes, thresholded dense segments, Bosses demoted by duration (200-2000 ms) into density discarding, sub-200 ms accents normalised to 100 ms, quiet-edge collectibles. No retraining. V4 stays default.'))
    registry=read('models/registry.json')
    registry['strategies']['v5']=dict(manifest='recipes/strategies/v5.json',weight_model='v3',
        status='optional_strategy',manifest_sha256=digest('recipes/strategies/v5.json'))
    registry['models']['v5']=dict(manifest='models/v5/manifest.json',application_version='0.5.0',
        status='optional_strategy',manifest_sha256=digest('models/v5/manifest.json'))
    save('models/registry.json',registry)


if __name__=='__main__':main()
