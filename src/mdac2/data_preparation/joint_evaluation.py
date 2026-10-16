"""Tolerance-based full-song metrics; reference charts are not human gold."""
from collections import defaultdict
import numpy as np
from .joint_decoding import pair_points
from .training_chart import CLASSES, LANES


def counts(predicted, reference, tolerance=.05):
    p = sorted(predicted)
    r = sorted(reference)
    pairs = pair_points([(v,) for v in p], [(v,) for v in r], tolerance)
    errors = [abs(p[a] - r[b]) for a, b in pairs]
    return np.array([len(pairs), len(p)-len(pairs), len(r)-len(pairs)], np.int64), errors


def result(count):
    tp, fp, fn = map(int, count)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn), reference_count=tp+fn, predicted_count=tp+fp)


def atoms(events, mask=None):
    values = []
    for e in events:
        for lane in e.lanes:
            side, kind = LANES.index(lane), CLASSES.index(e.kind)
            frame = min(len(mask)-1, int(e.start*100+1e-7)) if mask is not None else 0
            if mask is None or mask[frame, side, kind]:
                values.append((e.start, lane, e.kind, e.end))
    return values


class EventMetrics:
    def __init__(self, tolerance=.05):
        self.tolerance = tolerance
        self.cells = defaultdict(lambda: np.zeros(3, np.int64))
        self.errors = []
        self.duration = 0.
        self.charts = 0
        self.rules = 0
        self.candidates = 0
        self.decode_rejections = defaultdict(int)

    def add(self, predicted, reference, duration, audit, onset_mask=None):
        p, r = atoms(predicted, onset_mask), atoms(reference, onset_mask)
        self.duration += duration
        self.charts += 1
        self.rules += audit['rule_rejection_count']
        self.candidates += audit['candidate_count']
        for rejected in audit['rejected']:
            self.decode_rejections[rejected['code']] += 1
        # Sampling is a distinct time, independent of lane and kind. Co-located
        # Gemini/double/heavy+hold contribute one time to this sampling metric.
        for name, selector in [('sampling', lambda a: True)] + [(lane, lambda a, lane=lane: a[1]==lane) for lane in LANES]:
            pt = sorted(set(a[0] for a in p if selector(a)))
            rt = sorted(set(a[0] for a in r if selector(a)))
            count, errors = counts(pt, rt, self.tolerance)
            self.cells[name] += count
            if name == 'sampling':
                self.errors.extend(errors)
        for kind in CLASSES:
            for lane in LANES:
                pt = [a[0] for a in p if a[1]==lane and a[2]==kind]
                rt = [a[0] for a in r if a[1]==lane and a[2]==kind]
                count, _ = counts(pt, rt, self.tolerance)
                self.cells['class:'+kind] += count
                self.cells['lane_class:'+lane+':'+kind] += count
        for lane in LANES:
            ph = sorted((a[0],a[3]) for a in p if a[1]==lane and a[2]=='hold')
            rh = sorted((a[0],a[3]) for a in r if a[1]==lane and a[2]=='hold')
            head_count, _ = counts([a[0] for a in ph], [a[0] for a in rh], self.tolerance)
            tail_count, _ = counts([a[1] for a in ph], [a[1] for a in rh], .1)
            self.cells['hold_head'] += head_count
            self.cells['hold_tail'] += tail_count
            pairs = pair_points(ph, rh, self.tolerance)
            matched = sum(abs(ph[a][1]-rh[b][1]) <= .1 for a,b in pairs)
            self.cells['hold_joint'] += (matched, len(ph)-matched, len(rh)-matched)

    def report(self):
        classes = {kind: result(self.cells['class:'+kind]) for kind in CLASSES}
        extra = {**classes['extra_heavy'], 'false_positives_per_minute': int(self.cells['class:extra_heavy'][1])/max(.001,self.duration/60)}
        return dict(charts=self.charts, audio_seconds=self.duration, sampling=result(self.cells['sampling']),
            lanes={lane: result(self.cells[lane]) for lane in LANES}, six_classes=classes,
            lane_classes={lane: {k: result(self.cells['lane_class:'+lane+':'+k]) for k in CLASSES} for lane in LANES},
            macro_class_f1=sum(v['f1'] for v in classes.values())/6,
            holds={k: result(self.cells[k]) for k in ('hold_head','hold_tail','hold_joint')},
            extra_heavy=extra,
            matched_onset_mean_absolute_ms=float(np.mean(self.errors)*1000) if self.errors else None,
            matched_onset_p95_ms=float(np.percentile(self.errors,95)*1000) if self.errors else None,
            rule_rejections=self.rules, candidate_count=self.candidates,
            rule_rejection_rate=self.rules/max(1,self.candidates), decode_rejections=dict(self.decode_rejections),
            onset_tolerance_ms=self.tolerance*1000, hold_tail_tolerance_ms=100,
            unit='six-class and hold metrics use actual lane entities; sampling collapses exact co-located starts',
            masks='excluded ghost/item cells and quantization collisions excluded from scoring',
            human_review='pending; community reference is not gold')
