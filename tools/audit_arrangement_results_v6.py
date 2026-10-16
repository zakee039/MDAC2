"""Verify frozen identities, media and generated event legality; record honest acceptance."""
import importlib,json,hashlib,sys
from pathlib import Path
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.generated_constraints import validate_generated_events
from mdac2.data_preparation.six_export import read_bms,semantics

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for variant in ('v6',):
 r=ROOT/f'tools/private/community/supervised-dual-{variant}-20260930';config=json.loads((r/'arrangement.json').read_text(encoding='utf-8'))
 module=importlib.import_module('mdac2.data_preparation.arrangement_context_training_'+variant)
 saved=torch.load(r/'formal-arrangement/best.pt',map_location='cpu',weights_only=False)
 assert saved['identity']==module.identity(config) and saved['config']==config
 assert digest(Path(config['manifest']))=='34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71'
 media=[];review=r/f'review-{variant}';records=json.loads((review/'review.json').read_text(encoding='utf-8'))
 for example in records['examples']:
  for key,clip in example['clips'].items():
   p=review/clip['file'];info=sf.info(p);assert digest(p)==clip['sha256'];assert abs(info.duration-(example['actual_end']-example['actual_start']))<.001
   media.append(dict(file=p.relative_to(r).as_posix(),sha256=digest(p),seconds=info.duration))
 pipa=r/f'pipaxing-review-{variant}';chart=json.loads((pipa/'chart.json').read_text(encoding='utf-8'));events=[TrainingEvent.from_dict(e) for e in chart['events']]
 assert not validate_generated_events(events,chart['duration'])
 assert semantics(events)==semantics(read_bms((pipa/'experimental.bms').read_text(encoding='utf-8'),chart['duration']))
 for p in pipa.glob('*.wav'):
  info=sf.info(p);assert abs(info.duration-chart['duration'])<.001;media.append(dict(file=p.relative_to(r).as_posix(),sha256=digest(p),seconds=info.duration))
 for p in review.glob('*.bms'):
  chart2=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));events2=[TrainingEvent.from_dict(e) for e in chart2['events']]
  assert not validate_generated_events(events2,chart2['duration']);assert semantics(events2)==semantics(read_bms(p.read_text(encoding='utf-8'),chart2['duration']))
 comparison=json.loads((r/'comparison.json').read_text());metrics=comparison['versions'][variant]
 diagnosis=json.loads((r/'hierarchy-diagnosis.json').read_text());pipa_report=json.loads((pipa/'report.json').read_text())
 quality='pending_objective_review_and_human_acceptance'
 acceptance=dict(training_execution='passed',pipeline='completed',global_arrangement_quality=quality,
    relative_macro_improved=metrics['macro_five_class_f1']>comparison['versions']['v4b']['macro_five_class_f1'],
    five_class_macro_f1=metrics['macro_five_class_f1'],five_class_micro_f1=metrics['micro_five_class']['f1'],hold_joint_f1=metrics['holds']['hold_joint']['f1'],
    orientation_with_correct_type_accuracy=diagnosis['orientation_with_correct_type_accuracy'],
    checkpoint_sha256=digest(r/'formal-arrangement/best.pt'),best_epoch=saved['state']['best_epoch'],
    pipeline_commands_file='completion-status.json',checkpoint_identity_verified=True,manifest_unchanged=True,
    media_count=len(media),bms_roundtrip=True,human_listening='pending',game_acceptance='not_evaluated',production_promoted=False,new_mdm_created=False,locked_test_opened=False)
 atomic_json(r/'acceptance.json',acceptance);atomic_json(r/'media-audit.json',dict(passed=True,media=media))
 description='音频上下文与自主生成历史联合排键实验。训练执行与媒体检查通过不等于听感及真实游戏验收，完整分项和历史消融见实验报告。'
 warning=f'<aside style="padding:18px;background:#fee2e2;border:3px solid #b91c1c"><strong>{variant} 实验待整体排键审阅。</strong><p>{description} 此页为实际预测，待人工听评、未实机验收。</p></aside>'
 html_audit=[]
 for folder,name in ((review,'预测听评.html'),(pipa,'琵琶行-实际预测.html')):
  p=folder/name;before=digest(p);page=p.read_text(encoding='utf-8')
  if variant=='v4b':page=page.replace('排键 v4：','排键 v4b：').replace('排键 v4 实际','排键 v4b 实际').replace('v4齿轮','v4b齿轮')
  if warning not in page:
   ix=page.index('</h1>')+5;page=page[:ix]+warning+page[ix:];p.write_text(page,encoding='utf-8')
  html_audit.append(dict(file=p.relative_to(r).as_posix(),before_sha256=before,after_sha256=digest(p)))
 atomic_json(r/'preview-annotation.json',dict(reason='honest acceptance warning; audio and chart unchanged',files=html_audit))
 supervisor=json.loads((r/'supervision.json').read_text(encoding='utf-8-sig'));supervisor.update(status='completed_pending_review',last_written_by='root',acceptance='acceptance.json');atomic_json(r/'supervision.json',supervisor)
 print(json.dumps(dict(variant=variant,**acceptance)))
