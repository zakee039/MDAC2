"""Read-only acceptance of the completed r2 outputs; writes a new audit only."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path('E:/project/MDAC2')
E = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.vocal_priority_r2 import LookaheadPolicy, lookahead_mode
from mdac2.data_preparation.fixed_grid_r2 import GRID, on_grid, issues
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_grid_export_r2 import read_bms, semantics
from mdac2.data_preparation.six_export import write_bms as old_writer
from mdac2.data_preparation.playability_audit import audit_playability

read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
save = lambda p,v: p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (E/'completion.json').exists(), 'Do not overwrite completed acceptance'
manifest = read(ROOT/'models/v2/manifest.json')
assert read(ROOT/'models/registry.json')['current_model']=='v2'
checkpoint_identity = {}
for name, item in manifest['checkpoints'].items():
    assert sha(ROOT/item['path']) == item['sha256'] == sha(ROOT/'models/v1/weights'/f'{name}.pt')
    checkpoint = torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
    for path,digest in checkpoint['identity']['code'].items():
        assert sha(ROOT/'src/mdac2/data_preparation'/path)==digest
    checkpoint_identity[name]=dict(sha256=item['sha256'],bound_sources=checkpoint['identity']['code'])
post = manifest['postprocessing']
assert sha(ROOT/post['prior_path'])==post['prior_sha256']
for path,meta in post['sources'].items():
    assert sha(ROOT/path)==sha(ROOT/meta['snapshot'])==meta['sha256']
for package in manifest['playtests']:
    assert sha(Path(package['delivery']))==package['sha256']
old = read(ROOT/'tools/private/community/vocal-priority-r1-20261016/completion.json')
for path,digest in old['source_sha256'].items():
    assert sha(ROOT/path)==digest
for package in old['packages']:
    assert sha(Path(package['delivery']))==package['sha256']

v = read(E/'validation/validation.json')
assert v['status']=='completed' and len(v['charts'])==22 and not v['identity']['locked_test_opened']
assert sha(ROOT/'models/v2/manifest.json')==v['identity']['model_manifest_sha256']
assert sha(ROOT/'tools/private/community/training-preparation-v3/manifest.json')==v['identity']['manifest_sha256']
for path,digest in v['identity']['code'].items():
    assert sha(ROOT/path)==digest
for slug in ['validation','pipa','snow']:
    identity=read(E/slug/'identity.json')
    assert identity['code']==v['identity']['code'] and identity['policy']==v['identity']['policy']
    for path,digest in identity['code'].items():
        assert sha(E/slug/'source'/Path(path).name)==digest

policy = LookaheadPolicy()
directories = [E/'validation'/r['chart_sha256'] for r in v['charts']] + [E/'pipa',E/'snow']
checks = []
for directory in directories:
    chart = read(directory/'chart.json')
    events = [TrainingEvent.from_dict(e) for e in chart['events']]
    assert not issues(events,chart['duration'])
    assert all(on_grid(t) for e in events for t in [e.start,e.end])
    bms = (directory/'experimental.bms').read_text(encoding='utf-8')
    assert '#BPM 240' in bms
    for line in bms.splitlines():
        if re.match(r'^#\d{3}(13|14|53|54):',line): assert len(line.split(':')[1])==400
    assert semantics(events)==semantics(read_bms(bms,chart['duration']))
    post = read(directory/'postprocessing.json')
    assert post['stage_order'][-2:]==['similar_passage_mirroring','validation_only']
    assert not post['alignment']['old_250ms_tail_extension']
    assert post['snapping']['event_count_preserved']
    assert all(abs(e['shift_ms'])<=7.500001 for e in post['snapping']['edits'])
    for edit in post['alignment']['edits']:
        assert max(edit['sources'])-min(edit['sources'])<=.10000001
        assert all(abs(edit['target']-t)<=.10000001 for t in edit['sources'])
    with np.load(directory/'routing.npz') as a:
        expected,coverage,decisions=lookahead_mode(a['active'],policy)
        np.testing.assert_array_equal(expected,a['vocal_mode'])
        np.testing.assert_allclose(coverage,a['future_coverage'])
        for start,end in np.flatnonzero(np.diff(np.r_[False,expected,False].astype(np.int8))).reshape(-1,2):
            assert a['active'][start] and a['active'][end-1]
        selected=a['selected_frames'];voice=expected[selected]
        assert np.all(a['support'][selected[voice]]>=policy.novelty_threshold)
        assert np.all(a['density_probability'][expected & (a['support']<policy.novelty_threshold)]==0)
    checks.append(dict(chart=directory.name,events=len(events),grid_and_BMS=True,
        lookahead_exact=True,release_latch_absent=True,voice_evidence_gate=True,density_gate=True,
        no_250ms_extension=True,changed_holds=post['alignment']['changed_holds'],
        extended=post['alignment']['extended'],shortened=post['alignment']['shortened'],
        rejected_alignments=len(post['alignment']['rejected']),grid_repairs=len(post['snapping']['repairs']),
        max_grid_shift_ms=max([abs(x['shift_ms']) for x in post['snapping']['edits']],default=0)))

http=[];packages=[];commands=[v['identity']['command']]
for slug in ['pipa','snow']:
    report=read(E/f'{slug}-package/package-report.json')
    assert sha(Path(report['package']))==sha(Path(report['delivery']))==report['package_sha256']
    with zipfile.ZipFile(report['delivery']) as z:
        assert z.testzip() is None
        for name,digest in report['resources'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
        info=json.loads(z.read('info.json'))
        assert str(info['bpm'])=='240' and info['levelDesigner']=='mikee' and info['scene']=='scene_01'
        chart=read(E/slug/'chart.json');events=[TrainingEvent.from_dict(e) for e in chart['events']]
        assert semantics(events)==semantics(read_bms(z.read('map2.bms').decode(),chart['duration']))
        new_bytes=len(z.read('map2.bms'))
    assert all(report['media_byte_identical_to_parent'].values())
    assert sha(Path(chart['source_audio']['source']))==chart['source_audio']['source_sha256']
    preview=read(E/'review'/slug/'audit.json')
    assert sha(ROOT/'tools/build_vocal_grid_review_r2.py')==preview['source_tool_sha256']
    assert sha(ROOT/'tools/package_grid_playtest_r2.py')==report['tool_sha256']
    for name,digest in preview['files'].items():assert sha(E/'review'/slug/name)==digest
    assert len(set(preview['frames'].values()))==1
    page=(E/'review'/slug/'index.html').read_text(encoding='utf-8')
    js='\n'.join(re.findall(r'<script>(.*?)</script>',page,re.S))
    subprocess.run(['node','--check','-'],input=js,text=True,capture_output=True,check=True,timeout=30)
    url=f'http://127.0.0.1:8783/{slug}/index.html'
    with urllib.request.urlopen(url,timeout=10) as response:
        assert response.status==200 and response.read()==(E/'review'/slug/'index.html').read_bytes()
    http.append(dict(url=url,status=200,bytes_equal=True,javascript_syntax=True))
    for name in preview['frames']:
        local=E/'review'/slug/name;size=local.stat().st_size
        for start,end in [(0,127),(size//2,size//2+127)]:
            req=urllib.request.Request(f'http://127.0.0.1:8783/{slug}/{name}',headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=10) as response:
                assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{size}'
                with local.open('rb') as f:f.seek(start);assert response.read()==f.read(end-start+1)
        http.append(dict(file=f'{slug}/{name}',status=206,ranges=2,bytes_equal=True))
    r=read(E/slug/'routing.json');post=read(E/slug/'report.json')
    old_r=read(ROOT/'tools/private/community/vocal-priority-r1-20261016'/slug/'routing.json')
    gaps={}
    for name,path in [('v2',E/slug/'base-chart.json'),('r1',ROOT/'tools/private/community/vocal-priority-r1-20261016'/slug/'chart.json'),('r2',E/slug/'chart.json')]:
        data=read(path);audit=audit_playability([TrainingEvent.from_dict(e) for e in data['events']],data['duration'])
        gaps[name]=audit['no_attack_or_hold_intervals']
    packages.append(dict(song=slug,delivery=report['delivery'],sha256=report['package_sha256'],
        events=report['logical_events'],classes=report['class_counts'],selected_onsets=r['selected_candidates'],
        voice_selected=r['voice_selected'],fallback_selected=r['accompaniment_selected'],
        vocal_mode_seconds=r['vocal_mode_seconds'],r1_vocal_mode_seconds=old_r['vocal_mode_seconds'],
        changed_holds=post['alignment']['changed_holds'],extended=post['alignment']['extended'],
        shortened=post['alignment']['shortened'],rejected=len(post['alignment']['rejected']),
        gaps=gaps,preview=url,bms_bytes=new_bytes,
        same_events_old_format_bytes=len(old_writer(events,chart['duration']).encode()),
        human_listening='pending',game_verified=False))
    commands += [read(E/slug/'identity.json')['command'],report['actual_command'],preview['command']]

suite=ET.parse(E/'full-tests.xml').getroot().find('testsuite')
tests={k:int(suite.attrib[k]) for k in ['tests','errors','failures','skipped']}
assert tests==dict(tests=593,errors=0,failures=0,skipped=0)
ablation=read(E/'onset-ablation.json');grid=read(E/'grid-diagnostics.json')
assert ablation['source_sha256']==sha(ROOT/'tools/ablate_vocal_lookahead_r2.py')
assert grid['source_sha256']==sha(ROOT/'tools/analyze_grid_effects_r2.py') and grid['final_events_exact_match']
assert grid['metrics']['final']['sampling']==v['metrics']['vocal_trial']['sampling']
commands += [ablation['command'],grid['command'],[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','tests','-q','--junitxml=tools/private/community/vocal-grid-r2-20261017/full-tests.xml']]
sources=[ROOT/p for p in v['identity']['code']]+[ROOT/'tools'/n for n in ['build_vocal_grid_review_r2.py','package_grid_playtest_r2.py','ablate_vocal_lookahead_r2.py','analyze_grid_effects_r2.py']]+[ROOT/'tests/unit/test_vocal_grid_r2.py',Path(__file__)]
(E/'source-final').mkdir(exist_ok=False)
for path in sources:shutil.copyfile(path,E/'source-final'/path.name)
save(E/'http-check.json',dict(status='passed',checks=http))
result=dict(status='completed',completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    actual_executor='root; no Sol delegation; no training',current_model='v2',
    baseline_manifest_sha256=sha(ROOT/'models/v2/manifest.json'),data_manifest_sha256=v['identity']['manifest_sha256'],
    checkpoint_identity=checkpoint_identity,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
    policy=v['identity']['policy'],export_grid=GRID,validation_metrics=v['metrics'],
    validation_sha256=sha(E/'validation/validation.json'),ablation_sha256=sha(E/'onset-ablation.json'),
    grid_diagnostics_sha256=sha(E/'grid-diagnostics.json'),checks=checks,packages=packages,tests=tests,
    http=dict(status='passed',pages=2,media_files=12,range_requests=24),commands=commands,
    training_started=False,locked_test_opened=False,production_promoted=False,
    engineering='passed',human_listening='pending',game_acceptance='pending',
    objective_quality='higher onset recall/F1 versus r1, more false positives; grid/hold editing alone slightly reduces reference accuracy',
    vocal_source_quality='not established; no source labels; stricter detector may fall back during real voice',
    failures=[dict(stage='review generation initial attempt',interpreter='.venv/Scripts/python.exe',
        error="ModuleNotFoundError: No module named 'torch'",resolution='reran with designated pretraining-py313; no partial output created')],
    limitations=['5ms is a storage/quantization grid, not measured musical beats or a proven stutter remedy.',
        'Source heuristics can miss real vocal activity or accept instrument leakage; low thresholds increase false positives.',
        'The unchanged joint onset network cannot recover starts with no candidate response; fallback is not a mix-only model.',
        'Route-only ablation has higher raw F1 than full lowered-threshold r2, while r2 prioritizes recall; not claimed best configuration.',
        'Hold alignment has lower reference head-tail F1 than before alignment; subjective preference still pending.',
        'Repeated validation is not independent test; two user songs were not used to choose thresholds.',
        'Separate CLI experiment and packages; frozen release v2 and default desktop generation unchanged.'])
save(E/'completion.json',result)
print(json.dumps(dict(status='completed',charts=len(checks),tests=tests,packages=[{k:p[k] for k in ['song','events','changed_holds','extended','shortened','rejected','bms_bytes','same_events_old_format_bytes']} for p in packages])))
