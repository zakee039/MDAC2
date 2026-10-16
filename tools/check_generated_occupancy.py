"""Recheck one existing prediction under v2 occupancy, without training/exporting a game package."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation import generated_constraints
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.generated_constraints import constrain_decoded_events
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--playtest',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    chart=json.loads((args.playtest/'chart.json').read_text(encoding='utf-8'))
    frozen=json.loads(args.frozen.read_text(encoding='utf-8'))
    if chart['prediction_sha256']!=sha(args.playtest/'prediction.npz') or chart['frozen_sha256']!=sha(args.frozen):
        raise ValueError('Prediction/frozen identity changed')
    events=[TrainingEvent.from_dict(e) for e in chart['events']]
    with np.load(args.playtest/'prediction.npz',allow_pickle=False) as prediction:
        selected,audit=constrain_decoded_events(events,prediction,chart['duration'],frozen['decoder'])
    before=audit_playability(events,chart['duration'])
    after=audit_playability(selected,chart['duration'])
    assert after['engineering_playtest_gate_passed']
    original={e.event_id:e for e in events}
    assert all(original[e.event_id]==e for e in selected)
    assert semantics(read_bms(write_bms(selected,chart['duration']),chart['duration']))==semantics(selected)
    held=[e.event_id for e in selected if e.kind=='hold' and any(
        o.kind in ('light','heavy','gemini') and set(o.lanes)&set(e.lanes) and e.start<=o.start<e.end for o in selected)]
    value=dict(policy_id=generated_constraints.POLICY_ID,constraint_code_sha256=sha(Path(generated_constraints.__file__)),
        original_chart_sha256=sha(args.playtest/'chart.json'),prediction_sha256=chart['prediction_sha256'],
        frozen_sha256=chart['frozen_sha256'],selection=audit,before=before,after=after,
        unchanged_retained_events=True,bms_semantic_roundtrip=True,holds_with_clicks_retained=held,
        events=[e.to_dict() for e in selected],game_package_written=False,
        coverage_quality_fixed=False,game_quality_verified=False,locked_test_used=False)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=dict(before_count=len(events),after_count=len(selected),
        before_exact_attack_stacks=before['exact_attack_stack_lane_times'],
        after_exact_attack_stacks=after['exact_attack_stack_lane_times'],
        after_nearby_attack_stacks=len(after['nearby_attack_stacks']),
        holds_with_clicks_retained=held,after_gap_count=len(after['no_attack_or_hold_intervals']),
        bms_semantic_roundtrip=True,game_package_written=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
