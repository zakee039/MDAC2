import json
from pathlib import Path
import zipfile
import pytest
from mdac2.data_preparation.community import extract, inspect_bms, prepare, rating_value


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/drive', 'a/../../outside', 'CON.txt', 'a./file'])
def test_extraction_fences_paths_and_never_publishes(tmp_path,name):
    archive=tmp_path/'a.mdm'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'bad')
    with pytest.raises(ValueError):extract(archive,tmp_path/'expanded')
    assert not (tmp_path/'expanded').exists()
    assert not (tmp_path/'outside').exists()


def test_duplicate_and_symlink_rejected(tmp_path):
    archive=tmp_path/'a.mdm'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('Map2.bms',b'first');z.writestr('map2.bms',b'second')
    with pytest.raises(ValueError,match='Duplicate'):extract(archive,tmp_path/'first')
    with zipfile.ZipFile(archive,'w') as z:
        member=zipfile.ZipInfo('link');member.external_attr=(0o120777<<16);z.writestr(member,b'outside')
    with pytest.raises(ValueError,match='Unsafe'):extract(archive,tmp_path/'second')


def test_unknown_objects_and_alphanumeric_channels_preserved(tmp_path):
    chart=tmp_path/'map2.bms'
    chart.write_text('#TITLE\n#BPM 120\n#PLAYLEVEL 10?\n#WAV0G Masher\n#001D3:000G\n#00253:0F000F00\n#00302:0.5\n#00413:xyz\n',encoding='utf-8')
    result=inspect_bms(chart)
    assert result['cells'][0]['channel']=='D3' and result['cells'][0]['code']=='0G'
    assert result['cells'][0]['index']==1 and result['cells'][0]['divisions']==2
    assert result['cells'][-1]['raw']=='0.5'
    assert result['errors'] and not result['gold_eligible']
    assert result['headers']['TITLE']==['']
    assert rating_value('10?') is None and rating_value('NaN') is None


def test_full_copy_nested_extract_and_rating_provenance(tmp_path):
    source=tmp_path/'source';source.mkdir()
    nested=tmp_path/'nested.zip'
    with zipfile.ZipFile(nested,'w') as z:z.writestr('readme.txt','inner')
    with zipfile.ZipFile(source/'pack.mdm','w') as z:
        z.writestr('info.json',json.dumps({'difficulty2':'9.5'}))
        z.writestr('map2.bms','#BPM 120\n#PLAYLEVEL 11\n#00113:01')
        z.writestr('nested.zip',nested.read_bytes())
    report=prepare(source,tmp_path/'target',ffmpeg=Path('unused'),ffprobe=Path('unused'))
    assert report['summary']['extracted_packages']==2
    assert report['summary']['selected_9_10']==1
    chart=report['charts'][0]
    assert 'difficulty_metadata_conflict' in chart['issues']
    assert 'missing_or_ambiguous_music' in chart['issues']
    assert chart['rating']=='9.5' and not chart['training_eligible']
    assert (tmp_path/'target/copied/pack.mdm').read_bytes()==(source/'pack.mdm').read_bytes()
    with pytest.raises(FileExistsError):prepare(source,tmp_path/'target',ffmpeg=Path('unused'),ffprobe=Path('unused'))
    with pytest.raises(ValueError):prepare(source,source/'target',ffmpeg=Path('unused'),ffprobe=Path('unused'))


def test_crc_corruption_does_not_publish(tmp_path):
    archive=tmp_path/'a.mdm'
    payload=b'unique-payload-for-crc-test'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('map2.bms',payload)
    data=bytearray(archive.read_bytes());position=data.index(payload);data[position]^=1;archive.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile):extract(archive,tmp_path/'expanded')
    assert not (tmp_path/'expanded').exists()
