import runpy
from pathlib import Path
from test_similarity import features
from mdac2.domain.structure import Span


def test_one_to_one_precision_and_fixture_comparison():
    module=runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/experiments/similarity/compare.py'))
    pair=(Span(0,8),Span(8,16))
    result=module['metrics']([pair,pair],[pair])
    assert result['precision']==.5 and result['recall']==1
    report=module['compare'](features([0,1,0,1]),[pair])
    assert all(m['true_positives']==1 for m in report['methods'].values())
    assert all(m['wall_seconds']>=0 for m in report['methods'].values())
