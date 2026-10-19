import io
import struct
from pathlib import Path
import pytest
from PIL import Image,ImageDraw
from mdac2.application.studio_imagegen import fallback,text_layout
from mdac2.application.studio_cover import embedded


@pytest.mark.parametrize('title,artist',[
    ('游京 (奋斗进行曲)','DJ筱轩'),('poi',''),('初音ミクの歌'*30,'作者'*80),
    ('A very long song title with words and punctuation '*5,'Multiple composers and singers'),
])
def test_text_layout_large_clear_and_fits_circle(title,artist):
    draw=ImageDraw.Draw(Image.new('RGB',(512,512)))
    layout=text_layout(title,artist)
    if title.startswith('游京'):assert layout[0][1].size>=48 and len(layout)>=3
    assert Image.open(io.BytesIO(fallback(title,artist))).size==(512,512)
    for line,font,y,color in layout:
        left,top,right,bottom=draw.textbbox((256,y),line,font=font,anchor='mt')
        for x in [left,right]:
            for yy in [top,bottom]:assert (x-256)**2+(yy-256)**2<240**2


def test_ncm_local_artwork_only(tmp_path):
    png=fallback('Embedded','')
    number=lambda n:struct.pack('<I',n)
    source=tmp_path/'song.ncm'
    source.write_bytes(b'CTENFDAM'+b'\0'*2+number(3)+b'key'+number(8)+b'metadata'+b'\0'*5+number(len(png)+12)+number(len(png))+png+b'\0'*12)
    root=Path(__file__).resolve().parents[2]
    assert Image.open(io.BytesIO(embedded(root,source))).size==(512,512)
    source.write_bytes(b'CTENFDAM'+b'\0'*2+number(9999999))
    assert embedded(root,source) is None
