"""Build an explicitly versioned review using the exact same example selection."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_dual_review import main as build


def main(experiment,evaluation,output):
    build(experiment,evaluation,output)
    path=output/'预测听评.html';page=path.read_text(encoding='utf-8')
    page=page.replace('双模型：实际预测审阅','排键修复 v2：实际预测审阅').replace('<title>MDAC2 双模型实际预测审阅</title>','<title>MDAC2 排键修复 v2 对照</title>')
    comparison=json.loads((experiment/'comparison.json').read_text(encoding='utf-8'))
    first=comparison['original_failed_preview'];a,b=first['before'],first['after']
    info=f'''<aside class="pending"><strong>本页是新排键 v2，采音保持原版本。</strong>
<p>仍采用相同验证歌曲和片段，不挑换成功例子。原失败整曲《{html.escape(first['title'])}》：
旧地面／天空 {a['lanes']['ground']}／{a['lanes']['air']}；
新地面／天空 {b['lanes']['ground']}／{b['lanes']['air']}。
全部22首中，按与参考地面占比相差超过25个百分点检查，仍有 {comparison['remaining_severe_lane_bias_charts']} 首明显偏轨。</p>
<p>没有强制交替、随机换轨或补点。轨道分布变化不等于排键、听感或游戏验收通过，六类和长条仍需分别审阅。</p></aside>'''
    page=page.replace('<h1>排键修复 v2：实际预测审阅</h1>','<h1>排键修复 v2：实际预测审阅</h1>'+info)
    path.write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.evaluation.resolve(),a.output.resolve())
