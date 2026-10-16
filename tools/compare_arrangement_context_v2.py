"""Same-song comparison and explicit collapse checks; never changes predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path


def distribution(events):
    lanes={lane:sum(lane in e['lanes'] for e in events) for lane in ('ground','air')}
    return dict(lanes=lanes,ground_fraction=lanes['ground']/max(1,sum(lanes.values())),
                classes=dict(Counter(e['kind'] for e in events)),logical_objects=len(events))


def main(old,new,manifest,evaluation=None):
    before=json.loads((old/'evaluation/validation.json').read_text(encoding='utf-8'))
    evaluation=evaluation or new/'evaluation'
    after=json.loads((evaluation/'validation.json').read_text(encoding='utf-8'))
    rows=json.loads(manifest.read_text(encoding='utf-8'))['charts'];rows={r['chart_sha256']:r for r in rows if r['training_eligible'] and r['split']=='validation'}
    records=[]
    for key,row in rows.items():
        reference=json.loads((manifest.parent/row['labels']).read_text(encoding='utf-8'))
        a=json.loads((old/'evaluation/charts-predicted_onsets'/f'{key}.json').read_text(encoding='utf-8'))
        b=json.loads((evaluation/'charts-predicted_onsets'/f'{key}.json').read_text(encoding='utf-8'))
        r=distribution(reference['events']);p=distribution(b['events']);o=distribution(a['events'])
        # Diagnostic thresholds declared for the revision, not forced decoding.
        severe_lane_bias=abs(p['ground_fraction']-r['ground_fraction'])>.25
        tokens=[d['token'] for d in b['audit']['decisions']]
        fractions=Counter(tokens);dominance=max(fractions.values(),default=0)/max(1,len(tokens))
        records.append(dict(chart_sha256=key,title=row['title'],before=o,after=p,reference=r,
                            severe_lane_bias=severe_lane_bias,dominant_pattern_fraction=dominance,
                            model_choices_changed_by_rules=b['audit']['rule_rejection_count']))
    metrics={}
    for mode in ('reference_onsets','predicted_onsets'):
        metrics[mode]={version:{k:report['free_running'][mode][k] for k in ('six_classes','macro_class_f1','lanes','holds','sampling')}
                       for version,report in [('before',before),('after',after)]}
    first=json.loads((old/'review-v1/review.json').read_text(encoding='utf-8'))['examples'][0]['chart_sha256']
    first=next(r for r in records if r['chart_sha256']==first)
    result=dict(scope='same 22 validation charts; identical onset checkpoint and threshold; no locked test',evaluation=evaluation.resolve().as_posix(),
                selection_uses_validation=True,charts=records,original_failed_preview=first,metrics=metrics,
                reference_onset_accuracy_before=before['free_running']['reference_onsets']['pattern_accuracy'],
                reference_onset_accuracy_after=after['free_running']['reference_onsets']['pattern_accuracy'],
                remaining_severe_lane_bias_charts=sum(r['severe_lane_bias'] for r in records),
                listening='pending human review',gameplay='not evaluated',production_promoted=False)
    (new/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('charts','metrics')},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--evaluation',type=Path)
    a=p.parse_args();main(a.old,a.new,a.manifest,a.evaluation)
