"""Compact fixed 240 BPM export. 200 cells per 1-second measure = 5ms."""
from collections import defaultdict

from .six_export import CODE,read_bms,semantics
from .fixed_grid_r2 import GRID,on_grid,issues

PROFILE_ID='customalbums-six-fixed240-grid5ms-r2'


def write_bms(events,duration):
    if not events or issues(events,duration):raise ValueError('Nonempty legal grid chart required')
    cells=defaultdict(list)
    for event in events:
        for lane in event.lanes:
            channel=('53' if lane=='air' else '54') if event.kind=='hold' else ('13' if lane=='air' else '14')
            for t in ([event.start,event.end] if event.kind=='hold' else [event.start]):
                if not on_grid(t):raise ValueError('Off-grid event; exporter must not silently quantize')
                measure,index=divmod(round(t*200),200)
                if measure>999:raise ValueError('BMS measure bound exceeded')
                cells[measure,channel].append((index,CODE[event.kind]))
    lines=['#PLAYER 3','#GENRE scene_01','#TITLE MDAC2 six-class review','#ARTIST MDAC2',
           '#BPM 240','#PLAYLEVEL 9.5','#RANK 2','#LNTYPE 1','#WAV10 music.ogg',
           '#WAV01 Small','#WAV0A Large1','#WAV11 BossMelee1','#WAV0E Gemini',
           '#WAV0F Hold','#WAV0H Gear','#00001:10']
    for (measure,channel),marks in sorted(cells.items()):
        buckets=[]
        for index,code in marks:
            bucket=next((b for b in buckets if index not in b),None)
            if bucket is None:bucket={};buckets.append(bucket)
            bucket[index]=code
        for bucket in buckets:
            data=['00']*200
            for index,code in bucket.items():data[index]=code
            lines.append(f'#{measure:03d}{channel}:'+''.join(data))
    return '\n'.join(lines)+'\n'
