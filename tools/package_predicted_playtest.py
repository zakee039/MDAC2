"""Build an independent experimental MDM from an exact audited prediction and verified cached media."""
import argparse,hashlib,io,json,sys,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics,PROFILE_ID

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def encoded(v):return (json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--title',required=True);a=p.parse_args()
 if a.output.exists() or a.delivery.exists():raise FileExistsError('Refuse to overwrite existing package or output')
 if any(c in a.title for c in '\r\n'):raise ValueError('Single-line title required')
 chart=read(a.prediction/'chart.json');parent=read(a.parent/'package-report.json');audio=read(a.parent/'audio.json');parent_path=Path(parent['package'])
 if sha(parent_path)!=parent['package_sha256']:raise ValueError('Parent package changed')
 if sha(Path(audio['source']))!=audio['source_sha256'] or sha(a.parent/'mix.wav')!=audio['wav_sha256']:raise ValueError('Original source/cached audio changed')
 if chart['source_audio']!=audio:raise ValueError('Prediction and cached audio identity differ')
 for stage,c in chart['checkpoint_identity'].items():
  if sha(Path(c['path']))!=c['sha256']:raise ValueError('Prediction checkpoint changed: '+stage)
 events=[TrainingEvent.from_dict(e) for e in chart['events']];audit=audit_playability(events,chart['duration'])
 if not audit['engineering_playtest_gate_passed']:raise ValueError('Prediction occupancy gate failed')
 preview=(a.prediction/'experimental.bms').read_text(encoding='utf-8')
 if semantics(events)!=semantics(read_bms(preview,chart['duration'])):raise ValueError('Preview BMS differs from predicted events')
 with zipfile.ZipFile(parent_path) as z:
  if z.testzip() is not None:raise ValueError('Corrupt parent archive')
  resources={name:z.read(name) for name in parent['resources']}
 digest=lambda b:hashlib.sha256(b).hexdigest()
 if any(digest(b)!=parent['resources'][name] for name,b in resources.items()):raise ValueError('Parent media/resources changed')
 info=json.loads(resources['info.json']);info.update(name=a.title,levelDesigner=chart['creator'],scene=chart['scene'])
 if info['scene']!='scene_01' or info['levelDesigner']!='mikee':raise ValueError('Unexpected requested metadata')
 bms=write_bms(events,chart['duration']).replace('#TITLE MDAC2 six-class review','#TITLE '+a.title).replace('#ARTIST MDAC2','#ARTIST Unknown\n#LEVELDESIGN '+chart['creator'])
 if semantics(read_bms(bms,chart['duration']))!=semantics(events):raise ValueError('Export BMS changed event time/type/lane')
 resources['map2.bms']=bms.encode('utf-8');resources['info.json']=encoded(info)
 identical={name:digest(resources[name])==parent['resources'][name] for name in ('music.ogg','demo.ogg','cover.png')}
 if not all(identical.values()):raise ValueError('Media changed unexpectedly')
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name,b in sorted(resources.items()):
   entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,b)
 data=stream.getvalue()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  if z.testzip() is not None or set(z.namelist())!=set(resources) or any(z.read(k)!=v for k,v in resources.items()):raise ValueError('Invalid archive')
 a.output.mkdir(parents=True,exist_ok=False);target=a.output/a.delivery.name;target.write_bytes(data)
 with a.delivery.open('xb') as f:f.write(data)
 if sha(target)!=sha(a.delivery) or sha(parent_path)!=parent['package_sha256']:raise ValueError('Delivery differs or parent changed')
 report=dict(status='completed',package=str(target.resolve()),delivery=str(a.delivery.resolve()),package_sha256=digest(data),bytes=len(data),title=a.title,creator=chart['creator'],scene=chart['scene'],difficulty=info['difficulty2'],duration=chart['duration'],logical_events=len(events),physical_lane_events=sum(len(e.lanes) for e in events),class_counts=dict(Counter(e.kind for e in events)),prediction_chart_sha256=sha(a.prediction/'chart.json'),prediction_bms_sha256=sha(a.prediction/'experimental.bms'),checkpoints=chart['checkpoint_identity'],profile=PROFILE_ID,preview_semantics_identical=True,bms_roundtrip=True,occupancy_passed=True,media_byte_identical_to_parent=identical,audio_alignment_offset_samples=parent['music_alignment_offset_samples'],audio_length_error_samples=parent['music_length_error_samples'],audio_validation='Inherited zero-offset/length checks from verified parent OGG; byte-identical original music, no preview click overlay',parent_package_sha256=parent['package_sha256'],source_audio_sha256=audio['source_sha256'],resources={k:digest(v) for k,v in resources.items()},game_verified=False,human_listening='pending',production_promoted=False,experimental_playtest_only=True,actual_command=[sys.executable,*sys.argv],tool_sha256=sha(Path(__file__)))
 (a.output/'package-report.json').write_bytes(encoded(report));(a.output/'chart.json').write_bytes((a.prediction/'chart.json').read_bytes());(a.output/'playability-audit.json').write_bytes(encoded(audit));(a.output/'source-tool.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
