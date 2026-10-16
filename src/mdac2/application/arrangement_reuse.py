"""Fixed-snapshot arrangement-only reuse; never moves or creates onsets."""
from dataclasses import dataclass,replace
from fractions import Fraction as F
import math
from uuid import uuid4
from mdac2.domain.artifact import content_hash
from mdac2.domain.structure import validate_reuse
from mdac2.domain.validation import validate_chart
from mdac2.analysis.phrase_alignment import AlignmentResult


def alignment_map(result,group,target):
    if type(result) is not AlignmentResult or not result.accepted:raise ValueError('Accepted local alignment required')
    p=result.artifact.payload
    if p['proposal_fingerprint']!=target.proposal_fingerprint or p['timing_fingerprint']!=group.timing_fingerprint:raise ValueError('Stale target alignment')
    anchors=tuple((F(a),F(b)) for a,b in p['anchors'])
    if anchors!=result.anchors or len(anchors)<2 or anchors[0]!=(group.primary.start_q,target.span.start_q) or anchors[-1]!=(group.primary.end_q,target.span.end_q) or any(a[0]>=b[0] or a[1]>=b[1] for a,b in zip(anchors,anchors[1:])):raise ValueError('Alignment does not cover reviewed boundaries')
    def mapped(q):
        if not anchors[0][0]<=q<=anchors[-1][0]:raise ValueError('Source outside alignment')
        for (a,b),(c,d) in zip(anchors,anchors[1:]):
            if a<=q<=c:return b+(q-a)*(d-b)/(c-a)
        raise ValueError('Uncovered source position')
    return mapped


@dataclass(frozen=True,slots=True)
class ReusePreview:
    base_hash:str
    group_hash:str
    timing_hash:str
    options:bytes
    proposed_events:tuple
    decisions:tuple[tuple[str,str,str|None],...]
    conflicts:tuple[tuple[str,str|None],...]
    before_issues:tuple[tuple[str,str,str|None],...]
    after_issues:tuple[tuple[str,str,str|None],...]

    @property
    def fingerprint(self):
        from mdac2.domain.artifact import strict_json
        return content_hash({'base':self.base_hash,'group':self.group_hash,'timing':self.timing_hash,'options':strict_json(self.options),
            'events':[e.to_dict() for e in self.proposed_events],'decisions':[list(d) for d in self.decisions],
            'conflicts':[list(c) for c in self.conflicts],'before':[list(i) for i in self.before_issues],'after':[list(i) for i in self.after_issues]})


def reports(chart,events,timing):
    report=lambda c:tuple((i.code,i.severity,i.event_id) for i in validate_chart(c,timing.mapping,timing_revision=timing.revision_id).issues)
    return report(chart),report(replace(chart,events=tuple(events)))


def preview_reuse(chart,timing,group,*,target_ids,alignments,mode='arrangement',max_match_seconds=.06):
    from mdac2.domain.artifact import canonical_json
    if mode not in ('arrangement','mirror'):raise ValueError('Arrangement-only mode required')
    if type(max_match_seconds) not in (int,float) or not math.isfinite(max_match_seconds) or not 0<=max_match_seconds<=.2:raise ValueError('Bounded matching tolerance required')
    source,targets=validate_reuse(chart,timing,group,target_ids=target_ids,mode=mode)
    if len(chart.events)>10000 or len(source)>512:raise ValueError('Reuse event limit exceeded')
    changes={};decisions=[];conflicts=[]
    for target in targets:
        mapped=alignment_map(alignments[target.target_id],group,target)
        events=[e for e in chart.events if target.span.contains(e.q)]
        if len(events)>512:raise ValueError('Target matching limit exceeded')
        edges=[]
        for a in source:
            for b in events:
                # Semantic/cardinality compatibility keeps Gemini and Holds intact.
                if a.semantic!=b.semantic or len(a.lane_mask)!=len(b.lane_mask):continue
                distance=abs(timing.mapping.t(mapped(a.q))-timing.mapping.t(b.q))
                if distance<=F(str(max_match_seconds)):edges.append((distance,a.event_id,b.event_id,a,b))
        used_source=set();matched={}
        for _,aid,bid,a,b in sorted(edges,key=lambda e:e[:3]):
            if aid in used_source or bid in matched:continue
            used_source.add(aid);matched[bid]=a
        for b in events:
            a=matched.get(b.event_id)
            if a is None:decisions.append((b.event_id,'UNMATCHED_PRESERVED',None));continue
            lanes=a.lane_mask
            if mode=='mirror' and len(lanes)==1:lanes=tuple('air' if lane=='ground' else 'ground' for lane in lanes)
            if set(lanes)==set(b.lane_mask):decisions.append((b.event_id,'UNCHANGED',a.event_id));continue
            if b.locked:decisions.append((b.event_id,'LOCKED_PRESERVED',a.event_id));continue
            if b.semantic=='hold' and b.end_q>target.span.end_q:
                conflicts.append(('HOLD_CROSSES_TARGET',b.event_id));continue
            changes[b.event_id]=replace(b,lane_mask=lanes)
            decisions.append((b.event_id,'LANES_CHANGED',a.event_id))
    events=tuple(changes.get(e.event_id,e) for e in chart.events)
    before,after=reports(chart,events,timing)
    conflicts.extend((code,event) for code,severity,event in after if severity=='error')
    options=canonical_json({'mode':mode,'targets':list(target_ids),'max_match_seconds':max_match_seconds,
        'alignments':{t.target_id:alignments[t.target_id].artifact.artifact_hash for t in targets},'matching':'nearest-compatible-one-to-one-1','locked':'preserve'})
    return ReusePreview(content_hash(chart.to_dict()),group.fingerprint,timing.fingerprint,options,events,tuple(decisions),tuple(conflicts),before,after)


def apply_reuse(chart,timing,group,*,expected_preview_fingerprint,**options):
    preview=preview_reuse(chart,timing,group,**options)
    if preview.fingerprint!=expected_preview_fingerprint:raise ValueError('Stale reuse preview')
    if preview.conflicts:raise ValueError('Unresolved reuse conflicts')
    return replace(chart,revision_id=str(uuid4()),events=preview.proposed_events,user_patch_ids=(*chart.user_patch_ids,preview.fingerprint))
