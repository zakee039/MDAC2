import pytest
from mdac2.application.studio_difficulty import estimate, level


def chart(step=1.,kind='light',lanes=('ground',),duration=30.):
    return dict(duration=duration,events=[dict(event_id=str(i),kind=kind,lanes=list(lanes),
        start=i*step,end=i*step,source_ids=[str(i)],source_code='',source_role='',exclusive=False,group_id=None)
        for i in range(int(duration/step))])


def test_density_and_double_inputs_raise_estimate_without_model_labels():
    easy=chart();fast=chart(.125);double=chart(1.,'gemini',('ground','air'))
    assert estimate(easy)['value'] < estimate(double)['value'] < estimate(fast)['value']
    assert estimate(easy)==estimate(easy|dict(model_id='r2',metadata={'title':'Anything'},timing={'bpm':240}))
    assert estimate(easy)['calibrated'] is False


def test_silent_and_short_chart_and_peak_burst():
    assert estimate(dict(duration=30,events=[]))['value']==1
    short=chart(.1,duration=.5)
    assert 1<=estimate(short)['value']<=15
    sparse=chart(2.)
    burst=chart(.1,duration=2.)
    mixed=dict(duration=30,events=sparse['events']+burst['events'])
    assert estimate(mixed)['value']>estimate(sparse)['value']
    assert estimate(mixed)['features']['peak_2s']>estimate(sparse)['features']['peak_2s']


@pytest.mark.parametrize('value',[None,True,'',0,21,'nan','inf',9.55])
def test_rating_rejects_invalid_input(value):
    with pytest.raises(ValueError):level(value)


def test_level_and_invalid_chart():
    assert level('8.3')=='8.3' and level(10)=='10.0' and level('9')=='9.0'
    with pytest.raises(ValueError):estimate(chart()|dict(duration=0))
