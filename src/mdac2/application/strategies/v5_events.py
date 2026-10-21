"""V5 output-only object types. Frozen six-class training labels stay intact."""
from dataclasses import dataclass
import math
from mdac2.data_preparation.training_chart import TrainingEvent,validate_events


@dataclass(frozen=True)
class SpecialEvent(TrainingEvent):
    source_note_count: int = 0
    target_hits: int = 0

    def __post_init__(self):
        if self.kind not in ('masher','music'):raise ValueError('Unknown V5 special')
        if not self.event_id or not self.source_ids or self.lanes not in (('ground',),('air',)):
            raise ValueError('Invalid V5 event identity/lane')
        if not all(math.isfinite(t) for t in (self.start,self.end)) or self.start<0 or self.end<self.start:
            raise ValueError('Invalid V5 event time')
        if self.exclusive or self.group_id:raise ValueError('V5 special cannot be exclusive/grouped')
        if self.kind=='masher':
            if self.end<=self.start or type(self.source_note_count)!=int or self.source_note_count<2 or self.target_hits!=self.source_note_count*2:
                raise ValueError('Masher must cover a dense group and retain its 2N target')
        elif self.end!=self.start or self.source_note_count or self.target_hits:
            raise ValueError('Music collectible must be instantaneous')


def from_dict(value):
    return (SpecialEvent if value['kind'] in ('masher','music') else TrainingEvent).from_dict(value)


def ordinary(event):
    if event.kind=='music':
        d=event.to_dict();d.pop('source_note_count');d.pop('target_hits');d['kind']='light'
        return TrainingEvent.from_dict(d)
    return event


def issues(events,duration):
    if len({e.event_id for e in events})!=len(events):raise ValueError('Duplicate V5 identities')
    result=validate_events([ordinary(e) for e in events if e.kind!='masher'],duration)
    mashers=[e for e in events if e.kind=='masher']
    for e in events:
        if e.end>duration+1e-6:result.append(dict(code='outside_audio',event_ids=[e.event_id]))
        if any(abs(t*1000-round(t*1000))>1e-7 for t in (e.start,e.end)):
            result.append(dict(code='sub_ms_time',event_ids=[e.event_id]))
        # Notes inside a sustained lane are legal; only their spacing is bounded.
    for i,a in enumerate(mashers):
        for b in mashers[i+1:]:
            if max(a.start,b.start)<min(a.end,b.end):result.append(dict(code='overlapping_mashers',event_ids=[a.event_id,b.event_id]))
    return result
