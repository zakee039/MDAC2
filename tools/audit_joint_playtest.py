"""Record a failed listening/playtest audit without changing frozen predictions."""
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import read_bms,semantics


def main():
    import argparse
    import soundfile as sf
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--playtest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    chart=json.loads((args.playtest/'chart.json').read_text(encoding='utf-8'))
    package=json.loads((args.playtest/'package-report.json').read_text(encoding='utf-8'))
    events=[TrainingEvent.from_dict(e) for e in chart['events']]
    result=audit_playability(events,chart['duration'])
    archive_path=Path(package['package'])
    with zipfile.ZipFile(archive_path) as archive:
        back=read_bms(archive.read('map2.bms').decode('utf-8'),chart['duration'])
    result['package_prediction_identical']=semantics(events)==semantics(back)
    result['package_sha256_unchanged']=sha(archive_path)==package['package_sha256']
    result['source_chart_sha256']=sha(args.playtest/'chart.json')
    waveform,rate=sf.read(args.playtest/'mix.wav',dtype='float32',always_2d=True)
    with np.load(args.playtest/'prediction.npz',allow_pickle=False) as pred:
        for gap in result['no_attack_or_hold_intervals']:
            first,last=int((gap['start']+.05)*100),int((gap['end']-.05)*100)
            part=waveform[int(gap['start']*rate):int(gap['end']*rate)]
            gap['mix_rms']=float(np.sqrt(np.mean(part**2)))
            gap['light_heavy_gemini_max_probability']=float(pred['onsets'][first:last,:][:,:,[0,1,3]].max())
            gap['vocal_activity_prior_mean']=float(pred['vocal_activity_prior'][first:last].mean())
            gap['interpretation']='audio/model evidence only; no annotated onset truth'
    # The old clips already contain stacking; short audio mixing and overlapping
    # SVG glyphs are insufficient to review their actual control requirements.
    review=ROOT/'tools/private/community/supervised-joint-v1-20260927/review-v2/review.json'
    examples=json.loads(review.read_text(encoding='utf-8'))['examples']
    result['old_preview_audits']=[]
    for example in examples:
        audit=audit_playability([TrainingEvent.from_dict(e) for e in example['predicted']],example['end'])
        result['old_preview_audits'].append(dict(focus=example['focus'],title=example['title'],
            interval=[example['start'],example['end']],logical_events=audit['logical_events'],
            exact_attack_stack_lane_times=audit['exact_attack_stack_lane_times'],
            gemini_count=audit['gemini_count'],gemini_with_nearby_attack=len(audit['gemini_with_nearby_attack'])))
    result['user_feedback']=dict(verdict='failed',general='效果非常糟糕，和预览的8秒的情况完全不同',
        specifics='空档非常多经常好几秒无操作，而且gemini通常两三组叠在一起',
        scope='user-reported play experience for the delivered song; not self-observed game execution')
    result['frozen_model_or_threshold_changed']=False
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(gap_count=len(result['no_attack_or_hold_intervals']),
        max_gap=max(g['seconds'] for g in result['no_attack_or_hold_intervals']),
        total_gap_seconds=result['total_reported_gap_seconds'],
        exact_attack_stack_lane_times=result['exact_attack_stack_lane_times'],
        nearby_attack_stacks=len(result['nearby_attack_stacks']),
        gemini_with_nearby_attack=len(result['gemini_with_nearby_attack']),
        package_prediction_identical=result['package_prediction_identical'],
        preview=result['old_preview_audits']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
