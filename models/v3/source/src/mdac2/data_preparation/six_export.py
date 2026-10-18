"""Experimental six-class BMS round-trip, separate from the legacy profile.

This verifies representation, not target-game acceptance. Every event retains
its time; arbitrary metadata is not emitted as executable BMS directives.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import tempfile
from .community import inspect_bms
from .community_labels import convert
from .training_chart import TrainingEvent,validate_events

PROFILE_ID='customalbums-bdfc7e6-six-experimental-v1'
CODE={'light':'01','heavy':'0A','extra_heavy':'11','gemini':'0E','hold':'0F','gear':'0H'}


def write_bms(events,duration):
    if not events:raise ValueError('Empty chart')
    if validate_events(events,duration):raise ValueError('Unresolved event conflicts prevent export')
    cells=defaultdict(list)
    for event in events:
        for lane in event.lanes:
            channel=('53' if lane=='air' else '54') if event.kind=='hold' else ('13' if lane=='air' else '14')
            for seconds in ([event.start,event.end] if event.kind=='hold' else [event.start]):
                value=Fraction(str(seconds))*1000
                nearest=round(value)
                # Addition of a 1ms tap-hold duration may leave binary float
                # noise (e.g. 1.2009999999999998). This is not sub-ms timing.
                if abs(value-nearest)>Fraction(1,100000000):
                    raise ValueError('Explicit millisecond precision required; no silent quantization')
                millisecond=nearest;measure,index=divmod(millisecond,2000)
                if measure>999:raise ValueError('BMS measure bound exceeded')
                cells[measure,channel].append((index,CODE[event.kind]))
    lines=['#PLAYER 3','#GENRE scene_01','#TITLE MDAC2 six-class review','#ARTIST MDAC2',
           '#BPM 120','#PLAYLEVEL 9.5','#RANK 2','#LNTYPE 1','#WAV10 music.ogg',
           '#WAV01 Small','#WAV0A Large1','#WAV11 BossMelee1','#WAV0E Gemini',
           '#WAV0F Hold','#WAV0H Gear','#00001:10']
    for (measure,channel),marks in sorted(cells.items()):
        buckets=[]
        for index,code in marks:
            bucket=next((b for b in buckets if index not in b),None)
            if bucket is None:bucket={};buckets.append(bucket)
            bucket[index]=code
        for bucket in buckets:
            # Duplicate lines preserve a hold tail and next head at the same
            # time. They must not be overwritten in a single cell map.
            data=['00']*2000
            for index,code in bucket.items():data[index]=code
            lines.append(f'#{measure:03d}{channel}:'+''.join(data))
    return '\n'.join(lines)+'\n'


def read_bms(text,duration):
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/'map.bms';path.write_text(text,encoding='utf-8')
        raw=inspect_bms(path)
    result=convert(raw,chart_hash='0'*64,duration=duration)
    if not result['training_eligible']:raise ValueError('Six-class read-back has unresolved semantics')
    return [TrainingEvent.from_dict(e) for e in result['events']]


def semantics(events):
    # BMS represents a both-lane hold as two physical holds and cannot recover
    # its logical grouping without the retained TrainingChart sidecar. Gemini
    # stays a paired logical event; compare holds by actual lane occupancy.
    return sorted((e.kind,lanes,round(e.start,9),round(e.end,9),e.exclusive) for e in events
                  for lanes in ([(lane,) for lane in e.lanes] if e.kind=='hold' else [e.lanes]))
