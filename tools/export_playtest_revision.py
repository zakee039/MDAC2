"""Package an audited event revision with byte-identical media; preserve its parent."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation import generated_constraints
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.generated_constraints import POLICY_ID
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.six_export import PROFILE_ID,write_bms,read_bms,semantics
from mdac2.data_preparation.training_chart import TrainingEvent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--selection-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--version',required=True)
    p.add_argument('--delivery',type=Path,required=True)
    args=p.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*',args.version):
        raise ValueError('Explicit version label required')
    if args.output.exists() or args.delivery.exists():
        raise ValueError('Revision output/delivery exists; refusing to overwrite')
    original=read(args.parent/'chart.json')
    parent=read(args.parent/'package-report.json')
    source_audio=read(args.parent/'audio.json')
    selected=read(args.selection_report)
    parent_path=Path(parent['package'])
    if sha(parent_path)!=parent['package_sha256'] or sha(source_audio['source'])!=source_audio['source_sha256']:
        raise ValueError('Parent package or source audio changed')
    if selected['original_chart_sha256']!=sha(args.parent/'chart.json') or selected['prediction_sha256']!=sha(args.parent/'prediction.npz'):
        raise ValueError('Selection does not match parent prediction')
    if selected['frozen_sha256']!=original['frozen_sha256'] or selected['constraint_code_sha256']!=sha(Path(generated_constraints.__file__)):
        raise ValueError('Frozen model/constraint provenance changed')
    if selected['policy_id']!=POLICY_ID:
        raise ValueError('Unsupported selection policy')
    events=[TrainingEvent.from_dict(e) for e in selected['events']]
    old_events=[TrainingEvent.from_dict(e) for e in original['events']]
    old_by_id={e.event_id:e for e in old_events}
    if not all(old_by_id.get(e.event_id)==e for e in events):
        raise ValueError('Selection changed retained events or invented a note')
    audit=audit_playability(events,original['duration'])
    if not audit['engineering_playtest_gate_passed']:
        raise ValueError('Revision failed occupancy checks')
    with zipfile.ZipFile(parent_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt parent package')
        resources={name:archive.read(name) for name in parent['resources']}
    import hashlib
    digest=lambda data:hashlib.sha256(data).hexdigest()
    if any(digest(data)!=parent['resources'][name] for name,data in resources.items()):
        raise ValueError('Parent resource identity changed')
    if semantics(read_bms(resources['map2.bms'].decode('utf-8'),original['duration']))!=semantics(old_events):
        raise ValueError('Parent BMS and parent events differ')
    title=original['title']+' '+args.version
    text=write_bms(events,original['duration'])
    text=text.replace('#TITLE MDAC2 six-class review','#TITLE '+title)
    text=text.replace('#ARTIST MDAC2','#ARTIST Unknown\n#LEVELDESIGN '+original['creator'])
    text=text.replace('#PLAYLEVEL 9.5',f"#PLAYLEVEL {original['difficulty_input']:g}")
    resources['map2.bms']=text.encode('utf-8')
    info=json.loads(resources['info.json'])
    info['name']=title
    resources['info.json']=encoded(info)
    if semantics(read_bms(text,original['duration']))!=semantics(events):
        raise ValueError('Revision BMS changed event semantics')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(resources.items()):
            entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,data)
    data=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.testzip() is not None or set(archive.namelist())!=set(resources):
            raise ValueError('Revision archive invalid')
        if any(archive.read(name)!=value for name,value in resources.items()):
            raise ValueError('Archived resource differs')
    media_identical={name:digest(resources[name])==parent['resources'][name]
                     for name in ('music.ogg','demo.ogg','cover.png')}
    if not all(media_identical.values()):
        raise ValueError('Revision unexpectedly changed media')
    args.output.mkdir(parents=True,exist_ok=False)
    target=args.output/args.delivery.name
    target.write_bytes(data)
    chart={**original,'title':title,'generation_policy':POLICY_ID,'events':[e.to_dict() for e in events],
        'class_counts':dict(Counter(e.kind for e in events)),
        'audit':{'occupancy_selection':selected['selection']},
        'parent_chart_sha256':sha(args.parent/'chart.json'),'selection_report_sha256':sha(args.selection_report),
        'human_listening':'pending','game_acceptance':'pending'}
    (args.output/'chart.json').write_bytes(encoded(chart))
    (args.output/'audio.json').write_bytes((args.parent/'audio.json').read_bytes())
    (args.output/'playability-audit.json').write_bytes(encoded(audit))
    (args.output/'selection-report.json').write_bytes(args.selection_report.read_bytes())
    with args.delivery.open('xb') as delivered:
        delivered.write(data)
    if sha(args.delivery)!=digest(data) or sha(parent_path)!=parent['package_sha256']:
        raise ValueError('Delivery mismatch or parent changed')
    report=dict(package=str(target.resolve()),delivery=str(args.delivery.resolve()),
        version=args.version,title=title,package_sha256=digest(data),bytes=len(data),
        parent_package=str(parent_path),parent_package_sha256=parent['package_sha256'],parent_unchanged=True,
        profile=PROFILE_ID,generation_policy=POLICY_ID,creator=info['levelDesigner'],scene=info['scene'],
        duration=original['duration'],difficulty=info['difficulty2'],logical_events=len(events),
        physical_lane_events=sum(len(e.lanes) for e in events),class_counts=chart['class_counts'],
        bms_semantic_roundtrip=True,occupancy_passed=True,media_byte_identical_to_parent=media_identical,
        audio_alignment_validation='inherited from the hash-verified parent; OGG bytes unchanged',
        remaining_gap_count=len(audit['no_attack_or_hold_intervals']),
        remaining_gap_seconds=audit['total_reported_gap_seconds'],sub_frame_holds=len(audit['sub_frame_holds']),
        dynamic_threshold_applied=False,game_verified=False,human_listening='pending',production_promoted=False,
        resources={name:digest(value) for name,value in resources.items()},
        actual_command=[sys.executable,*sys.argv],tool_sha256=sha(Path(__file__)))
    (args.output/'package-report.json').write_bytes(encoded(report))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
