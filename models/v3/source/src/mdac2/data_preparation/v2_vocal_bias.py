"""V2 inference iteration: soft vocal onset bias, no hard vocal replacement.

Existing baseline peaks are retained unless a stronger vocal-supported peak
competes within the same 30ms exclusion region. Quiet/no-voice score unchanged.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .vocal_priority_r2 import vocal_evidence,LookaheadPolicy
from .joint_decoding import peaks


@dataclass(frozen=True)
class VocalBias:
    version: str='v2-vocal-soft-bias-v1'
    maximum_odds_multiplier: float=2.0
    baseline_threshold: float=.3
    minimum_support: float=.16


def select(features,prediction,duration,policy=VocalBias()):
    p=np.asarray(prediction['probability'],np.float32)
    offsets=np.asarray(prediction['offsets'],np.float32)
    if p.shape!=offsets.shape or p.ndim!=1 or len(p)!=features.shape[-1] or abs(len(p)*.01-duration)>.011:
        raise ValueError('Feature/score/time mismatch')
    if not np.isfinite(p).all() or not np.isfinite(offsets).all() or np.any((p<0)|(p>1)|(offsets<0)|(offsets>1)):
        raise ValueError('Invalid onset output')
    ev=vocal_evidence(features,LookaheadPolicy())
    strength=np.where(ev['vocal_mode'] & (ev['support']>=policy.minimum_support),ev['support'],0.)
    multiplier=np.power(policy.maximum_odds_multiplier,strength)
    score=(p*multiplier/np.maximum(1e-12,1-p+p*multiplier)).astype(np.float32)
    chosen=dict(probability=score,offsets=offsets)
    last=np.floor(duration*1000)/1000-.001
    times=[min(last,round(float((i+offsets[i])*.01),3)) for i in peaks(score,policy.baseline_threshold,3)]
    base=set(peaks(p,.3,3));selected=set(peaks(score,.3,3))
    summary=dict(policy=asdict(policy),baseline_candidates=len(base),selected_candidates=len(selected),
        added=len(selected-base),removed_by_local_peak_competition=len(base-selected),
        boosted_frames=int(np.sum(strength>0)),vocal_mode_seconds=float(ev['vocal_mode'].sum()*.01),
        hard_vocal_mode_replacement=False,base_scores_suppressed=False,model_weights_changed=False,
        score_is_calibrated_probability=False,vocal_source_accuracy_verified=False)
    return times,chosen,summary
