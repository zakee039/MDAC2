"""Add aligned mix/vocal/accompaniment listening to training-only review cards."""
import argparse
import html
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.community import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();root=a.manifest.parent;m=json.loads(a.manifest.read_text(encoding='utf-8'))
    report=json.loads((a.cache/'report.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or report['errors'] or report['manifest_sha256']!=digest(a.manifest):
        raise ValueError('Complete matching feature report required')
    examples=json.loads((root/'review-examples.json').read_text(encoding='utf-8'))
    rows={r['chart_sha256']:r for r in m['charts']};players=[];records=[];player_count=0
    for example in examples:
        row=rows[example['chart_sha256']]
        if row['split']!='train':raise ValueError('Review must remain training-role only')
        start=example['start'];end=example['end'];directory=a.cache/row['pcm']['sha256']
        sources=[('原曲',Path(row['audio_path_absolute']))]
        if row['training_eligible']:sources.extend([('人声',directory/'vocals.wav'),('伴奏',directory/'accompaniment.wav')])
        player_count+=len(sources)
        parts=[]
        for label,path in sources:
            if not path.is_file():raise ValueError('Missing listening source')
            uri=html.escape(path.resolve().as_uri()+f'#t={start:.3f},{end:.3f}',quote=True)
            parts.append(f'<div><h3>{label}</h3><audio controls preload="none" data-start="{start:.3f}" data-end="{end:.3f}" src="{uri}"></audio><button type="button">从片段起点播放</button></div>')
        note='' if row['training_eligible'] else '<p>此谱隔离待复核，只提供原曲，未加入训练及分轨缓存。</p>'
        players.append(note+'<div class="streams">'+''.join(parts)+'</div>')
        records.append({'chart_sha256':row['chart_sha256'],'pcm_sha256':row['pcm']['sha256'],'role':'train',
                        'training_eligible':row['training_eligible'],'start':start,'end':end})
    page=(root/'归并审阅.html').read_text(encoding='utf-8')
    if len(re.findall(r'<audio\b[^>]*></audio>',page))!=len(players):raise ValueError('Review template changed')
    replacements=iter(players)
    page=re.sub(r'<audio\b[^>]*></audio>',lambda _:next(replacements),page)
    page=page.replace('<title>六类标签归并审阅</title>','<title>人声与伴奏对照审阅</title>')
    page=page.replace('<h1>六类标签归并审阅</h1>','<h1>人声与伴奏对照审阅</h1><p>同一片段比较原曲、人声与伴奏，观察下方谱面落点。每次只播放一条音轨；未作各轨响度归一化。本页仅提供审阅材料，未预填听评结论。</p>')
    page+='''<style>.streams{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}button{padding:8px 12px;cursor:pointer}h3{margin-bottom:8px}@media(max-width:800px){.streams{grid-template-columns:1fr}}</style>
<script>
const tracks=[...document.querySelectorAll('audio')];
for(const track of tracks){
 track.addEventListener('play',()=>{for(const other of tracks)if(other!==track)other.pause();});
 track.addEventListener('timeupdate',()=>{if(track.currentTime>=Number(track.dataset.end))track.pause();});
 track.nextElementSibling.addEventListener('click',()=>{
  const play=()=>{track.currentTime=Number(track.dataset.start);track.play().catch(()=>{});};
  if(track.readyState>=1)play();else{track.addEventListener('loadedmetadata',play,{once:true});track.load();}
 });
}
</script>'''
    output=root/'分轨审阅.html';output.write_text(page,encoding='utf-8')
    write_json(root/'stem-review.json',{'manifest_sha256':digest(a.manifest),'html_sha256':digest(output),
        'examples':records,'human_review':'pending','training_started':False})
    print(json.dumps({'examples':len(records),'audio_players':player_count,'human_review':'pending'}))


if __name__=='__main__':main()
