import json
import pytest
from mdac2.data_preparation.community import digest, resolve_music
from mdac2.data_preparation import community_selection
from mdac2.data_preparation.community_selection import in_range, group_families


@pytest.mark.parametrize('value,expected',[('8.49',False),('8.5',True),('9',True),('10.5',True),('10.51',False),('10?',False),('NaN',False)])
def test_user_range_has_exact_inclusive_boundaries(value,expected):
    assert in_range(value,'8.5','10.5') is expected


def test_invalid_range_rejected():
    with pytest.raises(ValueError):in_range('9','10.5','8.5')
    with pytest.raises(ValueError):in_range('9','8.5','NaN')


def test_family_transitivity_prevents_song_and_pcm_leakage():
    rows=[{'chart_sha256':'a'*64,'title':'Song!','pcm':{'sha256':'x'}},
          {'chart_sha256':'b'*64,'title':'SONG','pcm':{'sha256':'y'}},
          {'chart_sha256':'c'*64,'title':'Different name','pcm':{'sha256':'y'}},
          {'chart_sha256':'d'*64,'title':'Other','pcm':{'sha256':'z'}}]
    group_families(rows)
    assert len({r['provisional_family'] for r in rows[:3]})==1
    assert rows[3]['provisional_family']!=rows[0]['provisional_family']


def test_main_mp3_supported_without_selecting_preview(tmp_path):
    (tmp_path/'demo.mp3').write_bytes(b'preview')
    song=tmp_path/'MUSIC.MP3';song.write_bytes(b'song')
    assert resolve_music(tmp_path)==song
    (tmp_path/'music.ogg').write_bytes(b'other song')
    with pytest.raises(ValueError,match='ambiguous'):
        resolve_music(tmp_path)


def test_selection_preserves_inputs_and_exposes_audio_failure(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir()
    rows=[]
    for package in ('mp3','missing'):
        folder=root/'expanded'/package;folder.mkdir(parents=True)
        chart=folder/'map2.bms';chart.write_text('#TITLE '+package,encoding='utf-8')
        raw=root/(package+'.json')
        raw.write_text(json.dumps({'cells':[{'channel':'13','code':'23'}]}),encoding='utf-8')
        rows.append({'package_hash':package,'chart_path':'map2.bms','chart_sha256':digest(chart),
            'raw_label_path':raw.name,'rating':'9','title':package,'selected_9_10':True,
            'issues':['missing_or_ambiguous_music'],'parse_errors':[], 'rating_sources':{'bms':'9'}})
    (root/'expanded/mp3/music.mp3').write_bytes(b'song')
    inventory=root/'inventory.json'
    inventory.write_text(json.dumps({'status':'completed','charts':rows}),encoding='utf-8')
    original=inventory.read_bytes()
    decoder=tmp_path/'decoder';decoder.write_bytes(b'decoder')
    decoded=[]
    def decode(path,*args):
        decoded.append(path.name)
        return {'sha256':'pcm','seconds':60}
    monkeypatch.setattr(community_selection,'pcm_identity',decode)
    result=community_selection.select_range(root,tmp_path/'result',minimum='8.5',maximum='10.5',
        ffmpeg=decoder,ffprobe=decoder)
    assert result['status']=='completed_with_errors'
    assert result['summary']['audio_verified_charts']==1
    assert len(result['errors'])==1 and decoded==['music.mp3']
    success,failed=result['charts']
    assert success['audio_status']=='verified' and success['issues']==[]
    assert success['resolved_preparation_issues']==['missing_or_ambiguous_music']
    assert failed['audio_status']=='error' and 'pcm' not in failed
    assert all(not r['training_eligible'] for r in result['charts'])
    assert inventory.read_bytes()==original
    with pytest.raises(FileExistsError):
        community_selection.select_range(root,tmp_path/'result',minimum='8.5',maximum='10.5',
            ffmpeg=decoder,ffprobe=decoder)
