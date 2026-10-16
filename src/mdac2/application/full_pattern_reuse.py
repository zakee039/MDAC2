"""Explicit full replacement previews with boundary policy and reversible snapshots."""
from dataclasses import dataclass,replace
from uuid import uuid4
from mdac2.domain.artifact import content_hash,canonical_json
from mdac2.domain.structure import validate_reuse
from mdac2.application.arrangement_reuse import ReusePreview,alignment_map,reports


@dataclass(frozen=True,slots=True)
class FullReusePreview:
    reuse:ReusePreview
    deleted:tuple
    added:tuple
    modified:tuple

    @property
    def fingerprint(self):
        return content_hash({'reuse':self.reuse.fingerprint,'deleted':[e.to_dict() for e in self.deleted],
            'added':[e.to_dict() for e in self.added],'modified':[[a.to_dict(),b.to_dict()] for a,b in self.modified]})


def preview_full_reuse(chart,timing,group,*,target_ids,alignments,source_hold='reject',target_hold='reject',locked='reject'):
    if source_hold not in ('reject','skip','truncate') or target_hold not in ('reject','preserve','truncate') or locked not in ('reject','skip_target'):raise ValueError('Explicit supported boundary/lock policy required')
    source,targets=validate_reuse(chart,timing,group,target_ids=target_ids,mode='full_pattern')
    if len(chart.events)>10000 or len(source)*len(targets)>4096:raise ValueError('Full reuse resource limit exceeded')
    conflicts=[];decisions=[];prepared=[];base_hash=content_hash(chart.to_dict());group_hash=group.fingerprint
    # Includes incoming Holds so no boundary ambiguity is silently ignored.
    for e in chart.events:
        overlaps=e.semantic=='hold' and e.q<group.primary.end_q and e.end_q>group.primary.start_q
        if not group.primary.contains(e.q) and not overlaps:continue
        crossing=overlaps and (e.q<group.primary.start_q or e.end_q>group.primary.end_q)
        if crossing:
            if source_hold=='reject':conflicts.append(('SOURCE_HOLD_BOUNDARY',e.event_id));continue
            if source_hold=='skip':decisions.append((e.event_id,'SOURCE_HOLD_SKIPPED',None));continue
            q=max(e.q,group.primary.start_q);end=min(e.end_q,group.primary.end_q)
            prepared.append(replace(e,q=q,duration_q=end-q));decisions.append((e.event_id,'SOURCE_HOLD_TRUNCATED',None))
        else:prepared.append(e)
    if not prepared:conflicts.append(('EMPTY_EFFECTIVE_PRIMARY',None))
    changes={};removed=set();added=[]
    for target in targets:
        mapped=alignment_map(alignments[target.target_id],group,target)
        touched=[e for e in chart.events if target.span.contains(e.q) or e.semantic=='hold' and e.q<target.span.end_q and e.end_q>target.span.start_q]
        if any(e.locked for e in touched):
            if locked=='skip_target':decisions.append((target.target_id,'LOCKED_TARGET_SKIPPED',None));continue
            conflicts.extend(('LOCKED_TARGET_EVENT',e.event_id) for e in touched if e.locked);continue
        for e in touched:
            crossing=e.semantic=='hold' and (e.q<target.span.start_q or e.end_q>target.span.end_q)
            if crossing:
                if target_hold=='reject':conflicts.append(('TARGET_HOLD_BOUNDARY',e.event_id));continue
                if target_hold=='preserve':decisions.append((e.event_id,'TARGET_HOLD_PRESERVED',None));continue
                # Keep only outside fragments; a spanning Hold can yield two.
                fragments=[]
                if e.q<target.span.start_q:fragments.append(replace(e,duration_q=target.span.start_q-e.q))
                if e.end_q>target.span.end_q:fragments.append(replace(e,q=target.span.end_q,duration_q=e.end_q-target.span.end_q))
                if fragments:
                    changes[e.event_id]=fragments[0]
                    if len(fragments)>1:
                        identity=content_hash({'base':base_hash,'group':group_hash,'target':target.target_id,'split':e.event_id})
                        added.append(replace(fragments[1],event_id=identity,source_kind='event',source_id=e.event_id))
                    decisions.append((e.event_id,'TARGET_HOLD_TRUNCATED',None))
                else:removed.add(e.event_id)
            else:removed.add(e.event_id)
        for e in prepared:
            q=mapped(e.q);end=mapped(e.end_q)
            identity=content_hash({'base':base_hash,'group':group_hash,'target':target.target_id,'source':e.event_id,
                'alignment':alignments[target.target_id].artifact.artifact_hash,'source_hold':source_hold})
            added.append(replace(e,event_id=identity,q=q,duration_q=end-q,source_kind='event',source_id=e.event_id,locked=False))
            decisions.append((identity,'ADDED_FROM_PRIMARY',e.event_id))
    events=tuple(changes.get(e.event_id,e) for e in chart.events if e.event_id not in removed)+tuple(added)
    if len({e.event_id for e in events})!=len(events):raise ValueError('Generated event identity collision')
    before,after=reports(chart,events,timing)
    conflicts.extend((code,event) for code,severity,event in after if severity=='error')
    preserved={identity for identity,reason,_ in decisions if reason=='TARGET_HOLD_PRESERVED'}
    for e in chart.events:
        if e.event_id not in preserved:continue
        if any(set(e.lane_mask)&set(a.lane_mask) and (e.q<=a.q<e.end_q or a.semantic=='hold' and a.q<=e.q<a.end_q) for a in added):
            conflicts.append(('PRESERVED_HOLD_COLLISION',e.event_id))
    # Cross-target spanning Holds need a dedicated boundary edit, never order-dependent truncation.
    for e in chart.events:
        if e.semantic=='hold' and sum(e.q<t.span.end_q and e.end_q>t.span.start_q for t in targets)>1:
            conflicts.append(('HOLD_SPANS_MULTIPLE_TARGETS',e.event_id))
    options=canonical_json({'mode':'full_pattern','targets':list(target_ids),'source_hold':source_hold,'target_hold':target_hold,'locked':locked,
        'alignments':{t.target_id:alignments[t.target_id].artifact.artifact_hash for t in targets}})
    reuse=ReusePreview(base_hash,group_hash,timing.fingerprint,options,events,tuple(decisions),tuple(conflicts),before,after)
    return FullReusePreview(reuse,tuple(e for e in chart.events if e.event_id in removed),tuple(added),tuple((e,changes[e.event_id]) for e in chart.events if e.event_id in changes))


def apply_full_reuse(chart,timing,group,*,expected_preview_fingerprint,**options):
    preview=preview_full_reuse(chart,timing,group,**options)
    if preview.fingerprint!=expected_preview_fingerprint:raise ValueError('Stale full replacement preview')
    if preview.reuse.conflicts:raise ValueError('Unresolved full replacement conflicts')
    return replace(chart,revision_id=str(uuid4()),events=preview.reuse.proposed_events,user_patch_ids=(*chart.user_patch_ids,preview.fingerprint))


def undo_full_reuse(current,base,preview):
    if type(preview) is not FullReusePreview or content_hash(base.to_dict())!=preview.reuse.base_hash or not current.user_patch_ids or current.user_patch_ids[-1]!=preview.fingerprint:raise ValueError('Undo base/operation mismatch')
    expected=replace(base,revision_id=current.revision_id,events=preview.reuse.proposed_events,user_patch_ids=(*base.user_patch_ids,preview.fingerprint))
    if current!=expected:raise ValueError('Undo requires unchanged applied revision')
    return replace(base,revision_id=str(uuid4()),user_patch_ids=(*current.user_patch_ids,content_hash({'undo':preview.fingerprint,'current':content_hash(current.to_dict())})))
