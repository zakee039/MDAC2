"""Versioned community BMS interpretation with exact musical coordinates.

Source profile: CustomAlbums bdfc7e6. Unsupported timing quarantines a chart;
unresolved object spans are explicit, never silently converted into empty labels.
"""
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import math

from .training_chart import TrainingEvent, POLICY_ID, validate_events

LOADER_COMMIT = 'bdfc7e6c2f4be0c8b5c5b405ba26b287397f956f'
AIR = {'11', '13', '31', '33', '51', '53', 'D1', 'D3'}
GROUND = {'12', '14', '32', '34', '52', '54', 'D2', 'D4'}
EVENT_CHANNELS = AIR | GROUND | {'15', '16', '18'}
LIGHT = {'01','02','03','04','05','06','07','08','09','0C','0I','13','14','15','23',
         '30','31','32','33','34','35','36','37','3C','3D','3E','3F'}
HEAVY = {'0A','0B','0D','0J','38','39','3A','3B'}
CONTROL = {'0O','0P','0Q','0R','0S','0T','0U','0V','0W','10',
           '1A','1B','1C','1D','1E','1F','1G','1H','1J','1K','1L','1M',
           '1O','1P','1Q','1R','1S','1T','1U','1V','1X',
           '25','26','27','28','29','2A','2B','2C','2D','2E','2F','2G','2H','2I','2J','2K',
           '2N','2O','2P','2Q','2R','2S'}
PAIRED = {'0F','0G','16','17'}
EXCLUDED = {'20': 'item', '21': 'ghost', '22': 'heart'}


def identity(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:24]


class SourceTiming:
    def __init__(self, raw):
        if raw['errors']:
            raise ValueError('Raw syntax errors')
        headers = raw['headers']
        if len(headers.get('BPM', [])) != 1:
            raise ValueError('Exactly one initial BPM required')
        self.initial_bpm = F(headers['BPM'][0])
        if not 0 < self.initial_bpm <= 10000:
            raise ValueError('Invalid initial BPM')
        self.ratios = {}
        changes = {F(0): self.initial_bpm}
        self.change_lines = {F(0): -1}
        self.ratio_lines = {}
        for c in raw['cells']:
            channel = c['channel']
            if channel == '02':
                ratio = F(c['raw'])
                if not 0 < ratio <= 64 or c['measure'] in self.ratios:
                    raise ValueError('Invalid or duplicate measure ratio')
                self.ratios[c['measure']] = ratio
                self.ratio_lines[c['measure']] = c['line']
            elif channel in {'03','08'}:
                tick = self.tick(c)
                if channel == '03':
                    bpm = F(int(c['code'], 16))
                else:
                    values = headers.get('BPM'+c['code'], [])
                    if len(values) != 1:
                        raise ValueError('Unresolved BPM lookup')
                    bpm = F(values[0])
                if bpm <= 0 or bpm > 10000 or (tick in changes and tick != 0):
                    raise ValueError('Invalid or simultaneous tempo changes')
                changes[tick] = bpm
                self.change_lines[tick] = c['line']
            elif channel not in EVENT_CHANNELS | {'01','SC'}:
                raise ValueError('Unsupported channel '+channel)
        self.changes = sorted(changes.items())
        self.max_measure = max((c['measure'] for c in raw['cells']), default=0) + 1
        if self.max_measure > 1000:
            raise ValueError('Measure bound exceeded')
        self.starts = [F(0)]
        for m in range(self.max_measure+1):
            self.starts.append(self.starts[-1] + 4*self.ratios.get(m, F(1)))
        # The pinned loader evaluates time while reading. Reject files whose
        # late tempo/measure declarations would disagree with full-map timing.
        for c in raw['cells']:
            if c['channel'] not in EVENT_CHANNELS | {'01'}:
                continue
            tick = self.tick(c)
            if any(t < tick and line > c['line'] for t,line in self.change_lines.items()):
                raise ValueError('Late tempo declaration changes loader timing')
            if any(F(m) < tick and line > c['line'] for m,line in self.ratio_lines.items()):
                raise ValueError('Late measure declaration changes loader timing')
        music = [c for c in raw['cells'] if c['channel']=='01' and c.get('code')=='10']
        if len(music) > 1:
            raise ValueError('Multiple music starts')
        self.music_tick = self.tick(music[0]) if music else F(0)
        self.music_start = self.absolute(self.music_tick)
        self.origin_basis = 'scene_music_10' if music else 'implicit_zero_no_music_marker'

    @staticmethod
    def tick(c):
        return F(c['measure']) + F(c.get('index', 0), c.get('divisions', 1))

    def quarter(self, tick):
        m = int(tick)
        return self.starts[m] + (tick-m)*4*self.ratios.get(m, F(1))

    def absolute(self, tick):
        result = F(0)
        for i,(start,bpm) in enumerate(self.changes):
            if start >= tick:
                break
            end = min(tick, self.changes[i+1][0]) if i+1 < len(self.changes) else tick
            result += (self.quarter(end)-self.quarter(start))*60/bpm
        return result

    def seconds(self, c):
        # Game-facing note ticks are rounded to milliseconds by the loader.
        return round(float(self.absolute(self.tick(c)) - self.music_start), 3)

    def to_dict(self):
        return {'initial_bpm': str(self.initial_bpm), 'music_tick': str(self.music_tick),
                'music_start_seconds': float(self.music_start), 'origin_basis': self.origin_basis,
                'measure_ratios': {str(m):str(v) for m,v in self.ratios.items()},
                'tempos': [{'measure_tick':str(t),'quarter':str(self.quarter(t)),
                            'seconds':float(self.absolute(t)-self.music_start),'bpm':str(b)} for t,b in self.changes],
                'rounding': 'audio-relative milliseconds; rational integration',
                'game_verified': False}


def convert(raw, *, chart_hash, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Positive audio duration required')
    out = {'schema_version': 1, 'policy': POLICY_ID, 'loader_commit': LOADER_COMMIT,
           'chart_sha256': chart_hash, 'duration': duration, 'events': [], 'excluded': [],
           'unresolved': [], 'controls': [], 'issues': [], 'training_eligible': False,
           'gold_eligible': False, 'review': 'automatic_unreviewed'}
    try:
        timing = SourceTiming(raw)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        out['issues'].append({'code':'timing_unresolved', 'detail': str(exc)})
        return out
    out['timing'] = timing.to_dict()
    cells = []
    for c in raw['cells']:
        if c['channel'] not in EVENT_CHANNELS:
            continue
        cell = {**c, 'id': identity(chart_hash, c['line'], c.get('index',0)),
                'seconds': timing.seconds(c), 'quarter': str(timing.quarter(timing.tick(c)))}
        cells.append(cell)
    cells.sort(key=lambda c: (c['seconds'], c['channel'] != '15', c['line'], c.get('index',0)))
    out['source_events'] = cells
    consumed = set()
    paired_groups = defaultdict(list)
    geminis = defaultdict(list)
    for c in cells:
        if c['code'] in PAIRED and not c['channel'].startswith('D'):
            paired_groups[c['channel'],c['code']].append(c)
        if c['code'] == '0E':
            geminis[c['seconds']].append(c)
    ends = {}
    for group in paired_groups.values():
        for a,b in zip(group[::2],group[1::2]):
            ends[a['id']] = b
            consumed.add(b['id'])

    def exclude(c, reason, end=None, unresolved=False):
        out['unresolved' if unresolved else 'excluded'].append({
            'source_id': c['id'], 'source_code': c['code'], 'channel':c['channel'],
            'start': max(0., c['seconds']), 'end': max(0., c['seconds'] if end is None else end),
            'reason':reason})

    def emit(c, kind, lanes, end=None, refs=None, role=None, group=None):
        start = c['seconds']
        if start < 0 or start >= duration:
            exclude(c, 'outside_audio', end, unresolved=True)
            return
        last = start if end is None else end
        if last < start or last > duration+1e-6:
            exclude(c, 'invalid_or_outside_end', last, unresolved=True)
            return
        event = TrainingEvent(identity(c['id'],kind,','.join(lanes)), kind, tuple(lanes), start, last,
                              tuple(refs or [c['id']]), c['code'], role or c['code'],
                              kind=='extra_heavy', group)
        out['events'].append(event.to_dict())

    for c in cells:
        if c['id'] in consumed:
            continue
        code,channel,t = c['code'],c['channel'],c['seconds']
        lane = 'air' if channel in AIR else 'ground'
        if code in CONTROL:
            out['controls'].append(c)
        elif code in EXCLUDED:
            exclude(c, EXCLUDED[code])
        elif code == '0E':
            group = [g for g in geminis[t] if g['id'] not in consumed]
            ground = [g for g in group if g['channel'] not in AIR]
            air = [g for g in group if g['channel'] in AIR]
            if len(ground) == len(air) == 1:
                emit(c, 'gemini', ('ground','air'), refs=[ground[0]['id'],air[0]['id']], role='gemini_pair')
                consumed.update(g['id'] for g in group)
            else:
                # A singleton colored Gemini is not evidence of a two-lane hit.
                exclude(c, 'unpaired_or_ambiguous_gemini', unresolved=True)
        elif code in PAIRED:
            tap_form = channel.startswith('D')
            end = t+.001 if tap_form else ends.get(c['id'], {}).get('seconds')
            refs = [c['id']] + ([ends[c['id']]['id']] if c['id'] in ends else [])
            if code == '0G':
                if not tap_form and (end is None or end <= t):
                    exclude(c, 'unclosed_masher', unresolved=True)
                    continue
                for side in ('ground','air'):
                    emit(c, 'heavy', (side,), refs=refs, role='masher_start', group=c['id'])
                # Independent real Hold events remain; Mul length alone is not
                # evidence for creating an additional held input.
            elif code in {'16','17'}:
                if tap_form or end is None or end <= t:
                    exclude(c, 'boss_masher_without_sustained_interval', unresolved=True)
                else:
                    emit(c, 'hold', ('ground','air'), end, refs, role='boss_masher')
            elif end is None or end <= t:
                exclude(c, 'unclosed_hold', unresolved=True)
            else:
                emit(c, 'hold', (lane,), end, refs, role='tap_hold' if tap_form else 'hold')
        elif code in LIGHT:
            # Lane-shifting projectile variants need a runtime pathway profile.
            if code in {'33','37','3D','3F'}:
                exclude(c, 'unverified_lane_shift', unresolved=True)
            else:
                emit(c, 'light', (lane,), role='note' if code=='23' else code)
        elif code in HEAVY:
            if code == '3B':
                exclude(c, 'unverified_lane_shift', unresolved=True)
            else:
                emit(c, 'heavy', (lane,))
        elif code in {'11','12'}:
            emit(c, 'extra_heavy', (lane,), role='boss_single_melee')
        elif code in {'0H','18'}:
            emit(c, 'gear', (lane,))
        else:
            exclude(c, 'unknown_object', unresolved=True)
    # Two source lanes can describe one simultaneous Masher. Expanding each
    # independently would invent four heavy hits (or duplicate both-lane holds).
    # Merge only an exact cross-lane pair of the same code AND source interval;
    # preserve every marker identity. Same-lane duplicates remain errors.
    source_by_id={c['id']:c for c in cells}
    candidates=defaultdict(list)
    for e in out['events']:
        if e['source_role'] not in {'masher_start','boss_masher'}:
            continue
        origin=source_by_id[e['source_ids'][0]]
        endpoint=source_by_id[e['source_ids'][-1]]['seconds']
        key=(e['source_role'],e['source_code'],e['start'],endpoint,tuple(e['lanes']))
        candidates[key].append((e,'air' if origin['channel'] in AIR else 'ground'))
    removed=set();merged=[]
    for group in candidates.values():
        if len(group)!=2 or {side for _,side in group}!={'air','ground'}:
            continue
        events=[e for e,_ in group]
        references=sorted({ref for e in events for ref in e['source_ids']})
        merged_group=identity('cross_lane_masher',*references)
        merged.append({**events[0],'event_id':identity(merged_group,*events[0]['lanes']),
                       'source_ids':references,'group_id':merged_group})
        removed.update(e['event_id'] for e in events)
    out['events']=[e for e in out['events'] if e['event_id'] not in removed]+merged
    out['events'].sort(key=lambda e:(e['start'],e['event_id']))
    events = [TrainingEvent.from_dict(e) for e in out['events']]
    out['issues'].extend(validate_events(events, duration))
    # Partial outputs are review artifacts. Only completely interpretable,
    # conflict-free charts enter the initial supervised subset.
    out['training_eligible'] = bool(events) and not out['unresolved'] and not out['issues']
    return out
