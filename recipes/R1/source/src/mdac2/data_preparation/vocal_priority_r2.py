"""One-second lookahead vocal evidence. No release latch or tempo estimator."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .vocal_priority import VocalPolicy, evidence, intervals
from .joint_decoding import peaks


@dataclass(frozen=True)
class LookaheadPolicy(VocalPolicy):
    version: str = 'vocal-lookahead-r2'
    release_seconds: float = 0.
    lookahead_seconds: float = 1.
    minimum_coverage: float = .25
    minimum_segment_seconds: float = .08
    maximum_internal_gap_seconds: float = .08
    minimum_occupied_fraction: float = .125
    copied_stem_similarity: float = .995
    copied_stem_relative_db: float = -12.
    vocal_model_threshold: float = .08
    novelty_threshold: float = .16
    vocal_spacing_frames: int = 6


def lookahead_mode(raw, policy=LookaheadPolicy()):
    """Confirm from future evidence, retaining the actual first/last activity.

    Only internal gaps up to 80 ms are bridged. Never pre-activate before the
    first evidence or retain a mode after its final evidence. Whole-song input
    is required so lookahead does not reset at inference window boundaries.
    """
    raw = np.asarray(raw, bool)
    n = len(raw)
    window = round(policy.lookahead_seconds/policy.hop_seconds)
    if window < 1:
        raise ValueError('Positive lookahead required')
    spans = np.flatnonzero(np.diff(np.r_[False, raw, False].astype(np.int8))).reshape(-1, 2)
    joined = []
    maximum_gap = round(policy.maximum_internal_gap_seconds/policy.hop_seconds)
    for a,b in spans:
        if joined and a-joined[-1][1] <= maximum_gap:
            joined[-1][1] = int(b)
        else:
            joined.append([int(a),int(b)])
    cumsum = np.r_[0,np.cumsum(raw)]
    ends = np.minimum(np.arange(n)+window,n)
    coverage = (cumsum[ends]-cumsum[np.arange(n)])/np.maximum(1,ends-np.arange(n))
    mode = np.zeros(n,bool)
    decisions = []
    for a,b in joined:
        enough = (b-a)*policy.hop_seconds >= policy.minimum_segment_seconds-1e-9
        accepted = enough and coverage[a] >= policy.minimum_coverage
        if accepted:
            mode[a:b] = True
        decisions.append(dict(start=a*.01,end=b*.01,accepted=bool(accepted),
            future_coverage=float(coverage[a]),reason='confirmed' if accepted else 'short_or_uncertain'))
    return mode, coverage.astype(np.float32), decisions


def vocal_evidence(features, policy=LookaheadPolicy()):
    ev = evidence(features,policy)
    x = np.asarray(features,np.float64)
    power = np.expm1(x)/10000.
    mean = uniform_filter1d(power.mean(axis=1),3,axis=1,mode='nearest')
    db = 10*np.log10(np.maximum(mean,1e-12))
    # Frequency occupancy and weak near-identical leakage are additional
    # evidence only; neither constitutes a trained singing classifier.
    occupied = power[1] > np.maximum(power[1].max(axis=0)*.01,1e-10)
    fraction = occupied.mean(axis=0)
    bands = sum(occupied[a:b].mean(axis=0) >= .10 for a,b in [(0,32),(32,64),(64,96)])
    v,a = np.sqrt(power[1]),np.sqrt(power[2])
    similarity = (v*a).sum(axis=0)/np.maximum(1e-20,np.linalg.norm(v,axis=0)*np.linalg.norm(a,axis=0))
    copied = (similarity >= policy.copied_stem_similarity)&(db[1]-db[0] < policy.copied_stem_relative_db)
    raw = ((db[1] >= ev['mel_power_floor_db']) & (db[1]-db[0] >= policy.vocal_relative_db)
           & (fraction >= policy.minimum_occupied_fraction) & (bands >= 2) & ~copied)
    mode,coverage,decisions = lookahead_mode(raw,policy)
    ev.update(active=raw,vocal_mode=mode,future_coverage=coverage,frequency_occupancy=fraction.astype(np.float32),
              stem_similarity=similarity.astype(np.float32),copied_stem=copied,segments=decisions)
    return ev


def select_onsets(features,prediction,duration,policy=LookaheadPolicy()):
    p=np.asarray(prediction['probability'],np.float32)
    offsets=np.asarray(prediction['offsets'],np.float32)
    if p.ndim!=1 or offsets.shape!=p.shape or len(p)!=features.shape[-1] or abs(len(p)*.01-duration)>.011:
        raise ValueError('Score/feature time mismatch')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(offsets)) or np.any((p<0)|(p>1)|(offsets<0)|(offsets>1)):
        raise ValueError('Invalid scores')
    ev=vocal_evidence(features,policy);mode=ev['vocal_mode']
    base=peaks(p,policy.baseline_threshold,3)
    candidates=peaks(p,policy.vocal_model_threshold,3)
    supported=[i for i in candidates if mode[i] and ev['support'][i]>=policy.novelty_threshold]
    chosen=[]
    for i in sorted(supported,key=lambda j:(-float(ev['support'][j]),-float(p[j]),j)):
        if all(abs(i-j)>=policy.vocal_spacing_frames for j in chosen):chosen.append(i)
    chosen.extend(i for i in base if not mode[i]);chosen.sort()
    selected=[]
    for i in chosen:
        if selected and i-selected[-1]<=3:
            j=selected[-1]
            if (bool(mode[i]),float(ev['support'][i]),float(p[i]))>(bool(mode[j]),float(ev['support'][j]),float(p[j])):
                selected[-1]=i
        else:selected.append(i)
    last=np.floor(duration*1000)/1000-.001
    times=[max(0.,min(last,round(float((i+offsets[i])*.01),3))) for i in selected]
    if len(times)!=len(set(times)):raise ValueError('Duplicate onsets')
    density=p.copy();density[mode & (ev['support']<policy.novelty_threshold)]=0
    summary=dict(policy=asdict(policy),baseline_candidates=len(base),selected_candidates=len(selected),
        retained=len(set(base)&set(selected)),removed=len(set(base)-set(selected)),added=len(set(selected)-set(base)),
        vocal_mode_seconds=float(mode.sum()*.01),detected_vocal_seconds=float(ev['active'].sum()*.01),
        voice_selected=sum(bool(mode[i]) for i in selected),accompaniment_selected=sum(not mode[i] for i in selected),
        vocal_intervals=intervals(ev['active'],duration),vocal_mode_intervals=intervals(mode,duration),
        fallback_intervals=intervals(~mode,duration),segments=ev['segments'],
        rejected=dict(insufficient_vocal_onset_evidence=int(sum(mode[i] and ev['support'][i]<policy.novelty_threshold for i in candidates)),
                      close_vocal_candidates=int(len(supported)-sum(mode[i] for i in selected))),
        detector='one-second separated-stem evidence, frequency occupancy and weak-copy rejection; no source truth',
        fallback='unchanged v2 joint onset candidates at threshold 0.3, not mix-only model',
        model_retrained=False,production_promoted=False)
    arrays={k:v for k,v in ev.items() if isinstance(v,np.ndarray)}
    arrays.update(selected_frames=np.asarray(selected,np.int64),times=np.asarray(times),density_probability=density)
    return times,arrays,summary
