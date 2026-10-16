"""Loss contract only. No optimizer or parameter updates are performed here."""
import torch
from torch.nn import functional as F


def masked_mean(values,mask):
    mask=mask.to(values.dtype)
    return (values*mask).sum()/mask.sum().clamp_min(1)


def joint_losses(output,target,*,class_weights=None,positive_weights=None,hold_end_positive_weights=None):
    logits=output['onset_logits']
    weights=torch.ones(6,device=logits.device,dtype=logits.dtype) if class_weights is None else class_weights
    if weights.shape!=(6,) or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Six finite positive training-only class weights required')
    for value,shape in ((positive_weights,(6,)),(hold_end_positive_weights,(2,))):
        if value is not None and (value.shape!=shape or not torch.isfinite(value).all() or (value<=0).any()):
            raise ValueError('Finite positive training-only positive weights required')
    losses={
        'onsets':masked_mean(F.binary_cross_entropy_with_logits(logits,target['onsets'],reduction='none',pos_weight=positive_weights)*
                            weights.reshape(1,1,1,6),target['onset_mask']),
        'offsets':masked_mean(F.smooth_l1_loss(output['offsets'],target['offsets'],reduction='none'),target['offset_mask']),
        'hold_ends':masked_mean(F.binary_cross_entropy_with_logits(output['hold_end_logits'],target['hold_ends'],reduction='none',pos_weight=hold_end_positive_weights),target['hold_end_mask']),
        'hold_end_offsets':masked_mean(F.smooth_l1_loss(output['hold_end_offsets'],target['hold_end_offsets'],reduction='none'),target['hold_end_offset_mask']),
        'occupancy':masked_mean(F.binary_cross_entropy_with_logits(output['occupancy_logits'],target['occupancy'],reduction='none'),target['occupancy_mask'])}
    # Initial contract weights are explicit, not tuned against validation/test.
    losses['total']=losses['onsets']+.25*losses['offsets']+.5*losses['hold_ends']+.25*losses['hold_end_offsets']+.25*losses['occupancy']
    return losses
