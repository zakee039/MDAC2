"""Audio proposals, human relations and chart reuse are separate contracts."""
from dataclasses import asdict,dataclass
from fractions import Fraction
import math
from mdac2.domain.artifact import _hash,content_hash
from mdac2.domain.timing import rational


def text(value):
    if type(value) is not str or not value.strip():raise ValueError('Explicit structure identity/provenance required')


@dataclass(frozen=True,slots=True)
class Span:
    start_q:Fraction
    end_q:Fraction

    def __post_init__(self):
        object.__setattr__(self,'start_q',rational(self.start_q));object.__setattr__(self,'end_q',rational(self.end_q))
        if self.end_q<=self.start_q:raise ValueError('Nonempty increasing span required')

    def overlaps(self,other):return self.start_q<other.end_q and other.start_q<self.end_q
    def contains(self,q):return self.start_q<=q<self.end_q
    def to_dict(self):return {'start_q':str(self.start_q),'end_q':str(self.end_q)}


@dataclass(frozen=True,slots=True)
class SimilarityProposal:
    proposal_id:str
    source:Span
    target:Span
    audio_hash:str
    timing_fingerprint:str
    feature_hash:str
    similarity_by_source:tuple[tuple[str,float],...]
    boundary_uncertainty_seconds:float
    alignment_anchors:tuple[tuple[Fraction,Fraction],...]=()
    warnings:tuple[str,...]=()

    def __post_init__(self):
        text(self.proposal_id)
        if type(self.source) is not Span or type(self.target) is not Span or self.source.overlaps(self.target):raise ValueError('Disjoint source/target required')
        for h in (self.audio_hash,self.timing_fingerprint,self.feature_hash):_hash(h)
        if type(self.similarity_by_source) is not tuple or not self.similarity_by_source:raise ValueError('Immutable similarity evidence required')
        names=[]
        for entry in self.similarity_by_source:
            if type(entry) is not tuple or len(entry)!=2:raise ValueError('Named similarity required')
            name,value=entry;text(name);names.append(name)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Bounded similarity required')
        if len(set(names))!=len(names):raise ValueError('Duplicate similarity source')
        if type(self.boundary_uncertainty_seconds) not in (float,int) or not math.isfinite(self.boundary_uncertainty_seconds) or self.boundary_uncertainty_seconds<0:raise ValueError('Finite boundary uncertainty required')
        if type(self.alignment_anchors) is not tuple:raise ValueError('Immutable alignment required')
        anchors=[]
        for pair in self.alignment_anchors:
            if type(pair) is not tuple or len(pair)!=2:raise ValueError('Alignment pair required')
            a,b=map(rational,pair)
            if not self.source.start_q<=a<=self.source.end_q or not self.target.start_q<=b<=self.target.end_q:raise ValueError('Alignment outside spans')
            anchors.append((a,b))
        if anchors and (len(anchors)<2 or any(a[0]>=b[0] or a[1]>=b[1] for a,b in zip(anchors,anchors[1:]))):raise ValueError('Monotone alignment required')
        object.__setattr__(self,'alignment_anchors',tuple(anchors))
        if type(self.warnings) is not tuple:raise ValueError('Immutable warnings required')
        for warning in self.warnings:text(warning)

    @property
    def fingerprint(self):
        return content_hash({'id':self.proposal_id,'source':self.source.to_dict(),'target':self.target.to_dict(),
            'audio':self.audio_hash,'timing':self.timing_fingerprint,'features':self.feature_hash,
            'similarity':[[n,v] for n,v in self.similarity_by_source],'uncertainty':self.boundary_uncertainty_seconds,
            'alignment':[[str(a),str(b)] for a,b in self.alignment_anchors],'warnings':list(self.warnings)})


MODES=frozenset(('arrangement','mirror','full_pattern'))


@dataclass(frozen=True,slots=True)
class ConfirmedTarget:
    target_id:str
    span:Span
    proposal_fingerprint:str
    relation:str
    allowed_modes:tuple[str,...]=()

    def __post_init__(self):
        text(self.target_id);_hash(self.proposal_fingerprint)
        if type(self.span) is not Span or self.relation not in ('exact_repeat','variation','rhythmic_similar','unknown'):raise ValueError('Explicit target relation required')
        if type(self.allowed_modes) is not tuple or len(set(self.allowed_modes))!=len(self.allowed_modes) or any(m not in MODES for m in self.allowed_modes):raise ValueError('Explicit reuse permissions required')
        if self.relation=='unknown' and self.allowed_modes:raise ValueError('Unknown relation cannot authorize reuse')
        if 'full_pattern' in self.allowed_modes and self.relation!='exact_repeat':raise ValueError('Full replacement requires exact-repeat confirmation')


@dataclass(frozen=True,slots=True)
class ConfirmedStructureGroup:
    group_id:str
    primary:Span
    targets:tuple[ConfirmedTarget,...]
    timing_fingerprint:str
    actor:str
    reason:str
    review:str='confirmed'

    def __post_init__(self):
        for value in (self.group_id,self.actor,self.reason):text(value)
        _hash(self.timing_fingerprint)
        if self.review!='confirmed' or type(self.primary) is not Span or type(self.targets) is not tuple or not self.targets or any(type(t) is not ConfirmedTarget for t in self.targets):raise ValueError('Confirmed immutable targets required')
        if len({t.target_id for t in self.targets})!=len(self.targets):raise ValueError('Duplicate target ID')
        spans=[self.primary,*[t.span for t in self.targets]]
        if any(a.overlaps(b) for i,a in enumerate(spans) for b in spans[i+1:]):raise ValueError('Primary/targets must not overlap')

    @property
    def fingerprint(self):
        return content_hash({'id':self.group_id,'primary':self.primary.to_dict(),'targets':[
            {'id':t.target_id,'span':t.span.to_dict(),'proposal':t.proposal_fingerprint,'relation':t.relation,'allowed_modes':list(t.allowed_modes)} for t in self.targets],
            'timing':self.timing_fingerprint,'actor':self.actor,'reason':self.reason,'review':self.review})


def validate_relations(groups):
    if type(groups) is not tuple or any(type(g) is not ConfirmedStructureGroup for g in groups):raise ValueError('Immutable groups required')
    if len({g.group_id for g in groups})!=len(groups):raise ValueError('Duplicate structure group')
    targets=[t.span for g in groups for t in g.targets]
    if any(a.overlaps(b) for i,a in enumerate(targets) for b in targets[i+1:]):raise ValueError('Overlapping targets across groups')
    graph={g.group_id:[other.group_id for other in groups if any(t.span.overlaps(other.primary) for t in g.targets)] for g in groups}
    visiting=set();visited=set()
    def visit(identity):
        if identity in visiting:raise ValueError('Cyclic structure relations')
        if identity in visited:return
        visiting.add(identity)
        for other in graph[identity]:visit(other)
        visiting.remove(identity);visited.add(identity)
    for identity in graph:visit(identity)


def validate_reuse(chart,timing,group,*,target_ids,mode):
    if type(group) is not ConfirmedStructureGroup:raise ValueError('Only confirmed groups permit reuse')
    if group.timing_fingerprint!=timing.fingerprint or chart.timing_revision!=timing.revision_id:raise ValueError('Stale structure timing')
    if timing.review!='accepted':raise ValueError('Approved timing required for reuse')
    if type(target_ids) is not tuple or not target_ids or len(set(target_ids))!=len(target_ids):raise ValueError('Explicit unique targets required')
    targets={t.target_id:t for t in group.targets}
    if mode not in MODES or any(i not in targets or mode not in targets[i].allowed_modes for i in target_ids):raise ValueError('Target has not authorized this reuse mode')
    for span in (group.primary,*[targets[i].span for i in target_ids]):timing.mapping.t(span.start_q);timing.mapping.t(span.end_q)
    source=tuple(e for e in chart.events if group.primary.contains(e.q))
    if not source:raise ValueError('Empty primary cannot replace target notes')
    return source,tuple(targets[i] for i in target_ids)
