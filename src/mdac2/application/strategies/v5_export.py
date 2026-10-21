"""CustomAlbums 0G interval / 23 collectible export, with MDAC-only 2N metadata.

The loader pairs identical 0G codes in the same channel to obtain length. It
does not consume target_hits: that field is deliberately a comment, not a claim
of an enforceable game hit cap. Mapping reference: MDMods/CustomAlbums Bms.cs.
"""
from collections import defaultdict
import json
import re
from mdac2.data_preparation.training_chart import TrainingEvent
from .v5_events import SpecialEvent,issues

CODE=dict(light='01',heavy='0A',extra_heavy='11',gemini='0E',hold='0F',gear='0H',masher='0G',music='23')


def write_bms(events,duration):
    if not events or issues(events,duration):raise ValueError('Invalid V5 chart')
    cells=defaultdict(list);targets=[]
    for e in events:
        for lane in e.lanes:
            channel=('53' if lane=='air' else '54') if e.kind in ('hold','masher') else ('13' if lane=='air' else '14')
            for t in ((e.start,e.end) if e.kind in ('hold','masher') else (e.start,)):
                measure,index=divmod(round(t*1000),2000)
                if measure>999:raise ValueError('BMS measure limit')
                cells[measure,channel].append((index,CODE[e.kind]))
        if e.kind=='masher':targets.append(dict(start_ms=round(e.start*1000),end_ms=round(e.end*1000),lane=e.lanes[0],source_note_count=e.source_note_count,target_hits=e.target_hits))
    lines=['#PLAYER 3','#GENRE scene_01','#TITLE MDAC2 six-class review','#ARTIST MDAC2',
        '#BPM 120','#PLAYLEVEL 9.5','#RANK 2','#LNTYPE 1','#WAV10 music.ogg','#WAV01 Small',
        '#WAV0A Large1','#WAV11 BossMelee1','#WAV0E Gemini','#WAV0F Hold','#WAV0H Gear',
        '#WAV0G Masher','#WAV23 Note','#00001:10',
        '// MDAC2_V5_TARGET_HITS are authoring targets only; game behavior is duration-based.']
    lines += ['// MDAC2_V5_MASHER '+json.dumps(t,separators=(',',':')) for t in targets]
    for (measure,channel),marks in sorted(cells.items()):
        buckets=[]
        for index,code in marks:
            bucket=next((b for b in buckets if index not in b),None)
            if bucket is None:bucket={};buckets.append(bucket)
            bucket[index]=code
        for bucket in buckets:
            data=['00']*2000
            for index,code in bucket.items():data[index]=code
            lines.append(f'#{measure:03d}{channel}:'+''.join(data))
    return '\n'.join(lines)+'\n'


def read_bms(text,duration):
    """Read our explicit 120 BPM/1ms profile; reject unsupported timebases."""
    marks=[];targets={};codes={v:k for k,v in CODE.items()}
    if '#BPM 120' not in text.splitlines():raise ValueError('Expected V5 fixed storage tempo')
    for line in text.splitlines():
        if line.startswith('// MDAC2_V5_MASHER '):
            t=json.loads(line[len('// MDAC2_V5_MASHER '):]);key=(t['start_ms'],t['end_ms'],t['lane'])
            if key in targets:raise ValueError('Duplicate masher target')
            targets[key]=t
        m=re.fullmatch(r'#(\d{3})(13|14|53|54):([0-9A-Z]+)',line)
        if not m:continue
        measure,channel,data=m.groups()
        if len(data)!=4000:raise ValueError('Expected 1ms cells')
        for i in range(0,len(data),2):
            code=data[i:i+2]
            if code=='00':continue
            if code not in codes:raise ValueError('Unsupported V5 code')
            marks.append((int(measure)*2000+i//2,channel,code))
    pending={};geminis=defaultdict(list);out=[];used=set()
    for index,(ms,channel,code) in enumerate(sorted(marks)):
        lane='air' if channel in ('13','53') else 'ground';kind=codes[code]
        if kind=='gemini':geminis[ms].append(lane);continue
        identity=f'v5-read-{index}';t=ms/1000
        if kind in ('hold','masher'):
            key=(channel,code)
            if key not in pending:pending[key]=t;continue
            start=pending.pop(key)
            if kind=='hold':out.append(TrainingEvent(identity,kind,(lane,),start,t,(identity,),code,'bms'))
            else:
                key=(round(start*1000),ms,lane)
                if key not in targets:raise ValueError('Missing authoring target sidecar')
                target=targets[key];used.add(key)
                out.append(SpecialEvent(identity,kind,(lane,),start,t,(identity,),code,'bms',source_note_count=target['source_note_count'],target_hits=target['target_hits']))
        elif kind=='music':out.append(SpecialEvent(identity,kind,(lane,),t,t,(identity,),code,'bms'))
        else:out.append(TrainingEvent(identity,kind,(lane,),t,t,(identity,),code,'bms',kind=='extra_heavy'))
    if pending or set(targets)!=used:raise ValueError('Unpaired hold/masher or unused targets')
    for ms,lanes in geminis.items():
        if sorted(lanes)!=['air','ground']:raise ValueError('Unpaired Gemini')
        identity=f'v5-gemini-{ms}';out.append(TrainingEvent(identity,'gemini',('ground','air'),ms/1000,ms/1000,(identity,),'0E','bms'))
    if issues(out,duration):raise ValueError('Invalid V5 readback')
    return out


def semantics(events):
    return sorted((e.kind,lane,round(e.start,9),round(e.end,9),e.exclusive,getattr(e,'target_hits',0))
        for e in events for lane in ([(l,) for l in e.lanes] if e.kind=='hold' else [e.lanes]))
