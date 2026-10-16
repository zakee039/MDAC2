"""Reviewable lane/Hold repair commands recomputed from a fixed chart baseline."""
from dataclasses import dataclass,replace
from uuid import uuid4
from mdac2.analysis.arrangement_baseline import arrange,ArrangementPolicy
from mdac2.analysis.hold_semantics import propose_holds,HoldPolicy
from mdac2.domain.artifact import content_hash,canonical_json,strict_json
from mdac2.domain.validation import validate_chart


@dataclass(frozen=True,slots=True)
class OptimizationPreview:
    base_hash:str
    timing_fingerprint:str
    options:bytes
    events:tuple
    changes:tuple[tuple[str,tuple[str,...]],...]
    reasons:tuple[tuple[str,str],...]
    conflicts:tuple[tuple[str,str],...]
    before:tuple[tuple[str,str,str|None],...]
    after:tuple[tuple[str,str,str|None],...]

    @property
    def fingerprint(self):return content_hash({'base':self.base_hash,'timing':self.timing_fingerprint,'options':strict_json(self.options),
        'events':[e.to_dict() for e in self.events],'changes':[[i,list(fields)] for i,fields in self.changes],
        'reasons':[list(r) for r in self.reasons],'conflicts':[list(c) for c in self.conflicts],'before':[list(i) for i in self.before],'after':[list(i) for i in self.after]})


def rule_report(chart,timing):
    issues=[(i.code,i.severity,i.event_id) for i in validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True).issues]
    if any(severity=='error' for _,severity,_ in issues):return tuple(issues)
    ordered=sorted(chart.events,key=lambda e:(e.q,e.event_id));left=0
    for i,e in enumerate(ordered):
        while timing.mapping.t(e.q)-timing.mapping.t(ordered[left].q)>=1:left+=1
        if i-left+1>8:issues.append(('HIGH_LOCAL_NPS','warning',e.event_id))
        if any(set(e.lane_mask)&set(other.lane_mask) and (other.q==e.q or other.semantic=='hold' and other.q<=e.q<other.end_q) for other in ordered[:i]):
            issues.append(('LANE_OCCUPANCY_REVIEW','warning',e.event_id))
    return tuple(issues)


def preview_optimization(chart,timing,*,arrangement=False,hold_ids=(),evidence=None,span=None,arrangement_policy=ArrangementPolicy(),hold_policy=HoldPolicy()):
    if type(arrangement) is not bool or type(hold_ids) is not tuple or len(set(hold_ids))!=len(hold_ids):raise ValueError('Explicit optimizer selection required')
    if chart.timing_revision!=timing.revision_id or timing.review!='accepted' or len(chart.events)>2048:raise ValueError('Current approved bounded optimization input required')
    before=rule_report(chart,timing);events=chart.events;reasons=[];conflicts=[]
    if any(severity=='error' for _,severity,_ in before):raise ValueError('Resolve invalid chart before optimization')
    if arrangement:
        result=arrange(chart,timing,span=span,policy=arrangement_policy);events=result.events;reasons.extend(result.decisions);conflicts.extend(result.conflicts)
    if hold_ids:
        if evidence is None:raise ValueError('Sustained evidence required for selected Hold repairs')
        candidates={c.event_id:c for c in propose_holds(replace(chart,events=events),timing,evidence,policy=hold_policy)}
        if not set(hold_ids)<=set(candidates):raise ValueError('Unknown requested Hold event')
        updated=[]
        for e in events:
            if e.event_id not in hold_ids:updated.append(e);continue
            candidate=candidates[e.event_id]
            if span is not None and not span.contains(e.q):raise ValueError('Hold selection outside optimization region')
            reasons.append((e.event_id,candidate.reason))
            if candidate.end_q is None:conflicts.append((e.event_id,'HOLD_REVIEW_REQUIRED'));updated.append(e)
            else:updated.append(replace(e,semantic='hold',duration_q=candidate.end_q-e.q))
        events=tuple(updated)
    changes=[]
    for old,new in zip(chart.events,events):
        fields=tuple(key for key in old.to_dict() if old.to_dict()[key]!=new.to_dict()[key])
        if fields:
            if old.locked or old.source_kind=='manual' or not set(fields)<={'lane_mask','semantic','duration_q'}:raise ValueError('Optimizer exceeded declared writable fields')
            changes.append((old.event_id,fields))
    after=rule_report(replace(chart,events=events),timing)
    conflicts.extend((event or 'chart',code) for code,severity,event in after if severity=='error')
    options=canonical_json({'order':['arrangement','hold_semantics'],'arrangement':arrangement,'hold_ids':list(hold_ids),
        'arrangement_policy':arrangement_policy.fingerprint,'hold_policy':hold_policy.fingerprint,
        'evidence':None if evidence is None else evidence.fingerprint,'span':None if span is None else span.to_dict(),
        'read_fields':['q','lane_mask','semantic','duration_q','locked','source_kind'],'write_fields':['lane_mask','semantic','duration_q']})
    return OptimizationPreview(content_hash(chart.to_dict()),timing.fingerprint,options,events,tuple(changes),tuple(reasons),tuple(conflicts),before,after)


def apply_optimization(chart,timing,*,expected_preview_fingerprint,**options):
    p=preview_optimization(chart,timing,**options)
    if p.fingerprint!=expected_preview_fingerprint:raise ValueError('Stale optimization preview')
    if p.conflicts:raise ValueError('Unresolved optimization conflicts')
    return replace(chart,revision_id=str(uuid4()),events=p.events,user_patch_ids=(*chart.user_patch_ids,p.fingerprint))


def undo_optimization(current,base,preview):
    if type(preview) is not OptimizationPreview or content_hash(base.to_dict())!=preview.base_hash:raise ValueError('Undo base mismatch')
    expected=replace(base,revision_id=current.revision_id,events=preview.events,user_patch_ids=(*base.user_patch_ids,preview.fingerprint))
    if current!=expected:raise ValueError('Undo would overwrite later edits')
    return replace(base,revision_id=str(uuid4()),user_patch_ids=(*current.user_patch_ids,content_hash({'undo':preview.fingerprint,'current':content_hash(current.to_dict())})))
