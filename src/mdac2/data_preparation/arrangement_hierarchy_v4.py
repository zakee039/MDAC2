"""Separate mirrored composition from lane orientation; never independent lane firing."""
import torch
from .arrangement_data_v3 import PATTERNS

def mirror(pattern):
    g,a,h=pattern
    return a,g,((h&1)<<1)|((h&2)>>1)
GROUPS=tuple(sorted({min(p,mirror(p)) for p in PATTERNS}))
GROUP_IDS=tuple(GROUPS.index(min(p,mirror(p))) for p in PATTERNS)
ORIENTATIONS=tuple(int(p!=GROUPS[g]) for p,g in zip(PATTERNS,GROUP_IDS))
ASYMMETRIC=tuple(mirror(p)!=p for p in PATTERNS)

def hierarchical_scores(group_logits,orientation_logits):
    # Best orientation contributes zero to its group score, so uncertain lane
    # choice cannot halve the evidence for single notes relative to Gemini.
    groups=torch.as_tensor(GROUP_IDS,device=group_logits.device)
    sides=torch.as_tensor(ORIENTATIONS,device=group_logits.device)
    asymmetric=torch.as_tensor(ASYMMETRIC,device=group_logits.device)
    orientation=orientation_logits-orientation_logits.amax(-1,keepdim=True)
    offsets=orientation[...,groups,sides]*asymmetric
    return group_logits[...,groups]+offsets
