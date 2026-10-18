"""Experimental vocal-first selection on frozen onset scores, in absolute time.

The separated stem is evidence, not a ground-truth singing detector. These
scores are routing scores, never calibrated vocal probabilities. No beat grid.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .joint_decoding import peaks


@dataclass(frozen=True)
class VocalPolicy:
    version: str = 'vocal-priority-experiment-r1'
    hop_seconds: float = .01
    release_seconds: float = 2.
    attack_frames: int = 3
    vocal_relative_db: float = -26.
    vocal_floor_below_mix_peak_db: float = 45.
    absolute_mel_power_floor_db: float = -50.
    novelty_threshold: float = .22
    novelty_prominence: float = .06
    novelty_floor: float = .08
    support_radius_frames: int = 4
    baseline_threshold: float = .3
    vocal_model_threshold: float = .12
    vocal_spacing_frames: int = 8


class VocalLatch:
    """Persistent frame-clock state; release at the first full 2-second gap.

    Initial mode is accompaniment until first positive voice evidence. The
    active frame occupies [i*hop,(i+1)*hop); that interval is not silence.
    """
    def __init__(self, policy=VocalPolicy()):
        self.release = int(math.ceil(policy.release_seconds/policy.hop_seconds-1e-9))
        if self.release <= 0:
            raise ValueError('Positive release duration required')
        self.voice = False
        self.silent_frames = 0

    def feed(self, active):
        out = np.zeros(len(active), bool)
        for i, present in enumerate(active):
            if present:
                self.voice = True
                self.silent_frames = 0
            elif self.voice:
                # Output describes mode at the beginning of the current cell.
                if self.silent_frames >= self.release:
                    self.voice = False
                self.silent_frames += 1
            out[i] = self.voice
        return out


def intervals(mask, duration, hop=.01):
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [[round(a*hop, 3), round(min(b*hop, duration), 3)] for a, b in indices.reshape(-1, 2)]


def evidence(features, policy=VocalPolicy()):
    x = np.asarray(features, np.float32)
    if x.ndim != 3 or x.shape[:2] != (3, 96) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError('Expected finite nonnegative mix/vocal/accompaniment log-mel cache')
    if not x.shape[-1]:
        raise ValueError('Empty acoustic cache')
    power = np.expm1(x.astype(np.float64)).mean(axis=1)/10000.
    power = uniform_filter1d(power, 3, axis=1, mode='nearest')
    db = 10*np.log10(np.maximum(power, 1e-12))
    floor = max(policy.absolute_mel_power_floor_db,
                float(np.percentile(db[0], 95))-policy.vocal_floor_below_mix_peak_db)
    raw = (db[1] >= floor) & (db[1]-db[0] >= policy.vocal_relative_db)
    active = np.zeros(len(raw), bool)
    consecutive = 0
    for i, flag in enumerate(raw):
        consecutive = consecutive+1 if flag else 0
        active[i] = consecutive >= policy.attack_frames
    # Positive spectral change over 20 ms, with a neighbouring-frequency
    # maximum to suppress vibrato. Constant loud vowels do not make new notes.
    previous = np.concatenate([x[1, :, :1].repeat(2, axis=1), x[1, :, :-2]], axis=1)[:, :x.shape[-1]]
    previous = maximum_filter1d(previous, 3, axis=0, mode='nearest')
    flux = np.maximum(0, x[1]-previous).mean(axis=0)
    flux = uniform_filter1d(flux, 3, mode='nearest')
    scale = max(policy.novelty_floor, float(np.percentile(flux[active], 95)) if active.any() else 0.)
    novelty = np.clip(flux/scale, 0, 1)
    # Use actual peaks with prominence, not all frames of a loud plateau.
    ids, _ = find_peaks(novelty, height=policy.novelty_threshold,
                       prominence=policy.novelty_prominence, distance=4)
    sparse = np.zeros(len(raw), np.float32)
    acoustic_active = maximum_filter1d(active.astype(np.uint8), 5, mode='nearest') > 0
    ids = ids[acoustic_active[ids]]
    sparse[ids] = novelty[ids]
    support = maximum_filter1d(sparse, 2*policy.support_radius_frames+1, mode='constant')
    return dict(active=active, novelty=novelty.astype(np.float32), support=support,
                vocal_db=db[1].astype(np.float32), relative_db=(db[1]-db[0]).astype(np.float32),
                mel_power_floor_db=floor)


def select_onsets(features, prediction, duration, policy=VocalPolicy()):
    if not np.isfinite(duration) or duration <= 0 or policy.hop_seconds != .01:
        raise ValueError('Positive duration and fixed 10 ms feature contract required')
    p = np.asarray(prediction['probability'], np.float32)
    offsets = np.asarray(prediction['offsets'], np.float32)
    if p.ndim != 1 or offsets.shape != p.shape or len(p) != features.shape[-1]:
        raise ValueError('Score/feature shape mismatch')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(offsets)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid onset scores')
    if abs(len(p)*.01-duration) > .011 or np.any((offsets < 0) | (offsets > 1)):
        raise ValueError('Time axis or offsets mismatch')
    ev = evidence(features, policy)
    mode = VocalLatch(policy).feed(ev['active'])
    base = peaks(p, policy.baseline_threshold, 3)
    voice = [i for i in peaks(p, policy.vocal_model_threshold, 3)
             if mode[i] and ev['support'][i] >= policy.novelty_threshold]
    # Rank close candidates by human-voice onset support, retaining the neural
    # frame/offset rather than snapping a note onto a volume maximum.
    chosen = []
    for i in sorted(voice, key=lambda i: (-float(ev['support'][i]), -float(p[i]), i)):
        if all(abs(i-j) >= policy.vocal_spacing_frames for j in chosen):
            chosen.append(i)
    chosen += [i for i in base if not mode[i]]
    chosen.sort()
    # Cross-mode boundaries still obey the baseline's 30 ms peak exclusion.
    selected = []
    for i in chosen:
        if selected and i-selected[-1] <= 3:
            j = selected[-1]
            if (bool(mode[i]), float(ev['support'][i]), float(p[i])) > (bool(mode[j]), float(ev['support'][j]), float(p[j])):
                selected[-1] = i
        else:
            selected.append(i)
    last = np.floor(duration*1000)/1000-.001
    times = [max(0., min(last, round(float((i+offsets[i])*.01), 3))) for i in selected]
    if len(times) != len(set(times)):
        raise ValueError('Duplicate exported onset')
    # Later high-intensity densification must obey the same voice evidence.
    # This retains accompaniment candidates only after the 2 s release.
    density_probability = p.copy()
    density_probability[mode & (ev['support'] < policy.novelty_threshold)] = 0.
    summary = dict(policy=asdict(policy), baseline_candidates=len(base), selected_candidates=len(selected),
        retained=len(set(base)&set(selected)), removed=len(set(base)-set(selected)), added=len(set(selected)-set(base)),
        vocal_mode_seconds=float(mode.sum()*.01), detected_vocal_seconds=float(ev['active'].sum()*.01),
        voice_selected=sum(bool(mode[i]) for i in selected), accompaniment_selected=sum(not mode[i] for i in selected),
        vocal_intervals=intervals(ev['active'], duration), vocal_mode_intervals=intervals(mode, duration),
        fallback_intervals=intervals(~mode, duration), mel_power_floor_db=ev['mel_power_floor_db'],
        detector='fixed separated-stem energy/share plus spectral novelty; heuristic, no voice-source truth',
        initial_mode='accompaniment until first detected vocal', fallback='unchanged frozen joint onset model at threshold 0.3',
        peak_timing='original neural frame and offset; no beat snapping', voice_priority='vocal-supported candidates only while latched',
        vocal_novelty_support_is_not_ground_truth=True, model_retrained=False, production_promoted=False)
    arrays = {k: v for k, v in ev.items() if isinstance(v, np.ndarray)}
    arrays.update(vocal_mode=mode, selected_frames=np.asarray(selected, np.int64), times=np.asarray(times),
                  density_probability=density_probability)
    return times, arrays, summary
