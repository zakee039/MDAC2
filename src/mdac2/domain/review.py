"""Human decisions bind a precise revision and quality report, never a score."""
from dataclasses import dataclass,asdict,replace
import math
import re
from mdac2.domain.artifact import content_hash
from mdac2.domain.timing_edits import TimingRevision,EditConflict


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Content fingerprint required')


@dataclass(frozen=True,slots=True)
class QualityItem:
    name:str
    status:str
    value:float|None
    unit:str
    reasons:tuple[str,...]=()

    def __post_init__(self):
        if type(self.name) is not str or not self.name or self.status not in {'pass','warn','fail','unknown'}:raise ValueError('Named quality status required')
        if self.value is not None and (type(self.value) not in (int,float) or not math.isfinite(self.value)):raise ValueError('Finite metric required')
        if type(self.unit) is not str or not self.unit:raise ValueError('Explicit metric unit required')
        if type(self.reasons) is not tuple or any(type(r) is not str or not r for r in self.reasons):raise ValueError('Immutable reasons required')


@dataclass(frozen=True,slots=True)
class TimingQuality:
    revision_fingerprint:str
    observation_hash:str
    policy_hash:str
    items:tuple[QualityItem,...]

    def __post_init__(self):
        for h in (self.revision_fingerprint,self.observation_hash,self.policy_hash):_hash(h)
        if type(self.items) is not tuple or not self.items or any(type(i) is not QualityItem for i in self.items):raise ValueError('Quality components required')
        if len({i.name for i in self.items})!=len(self.items):raise ValueError('Duplicate quality component')

    @property
    def fingerprint(self):
        data=asdict(self);data['items']=[{**asdict(item),'reasons':list(item.reasons)} for item in self.items]
        return content_hash(data)


@dataclass(frozen=True,slots=True)
class ReviewDecision:
    revision_id:str
    base_fingerprint:str
    reviewed_fingerprint:str
    quality_fingerprint:str
    decision:str
    actor:str
    reason:str
    timestamp:str

    def __post_init__(self):
        for text in (self.revision_id,self.actor,self.reason,self.timestamp):
            if type(text) is not str or not text.strip():raise ValueError('Explicit review provenance required')
        if self.decision not in {'accepted','rejected'}:raise ValueError('Explicit review decision required')
        for h in (self.base_fingerprint,self.reviewed_fingerprint,self.quality_fingerprint):_hash(h)

    def is_current(self,revision,quality):
        return (revision.revision_id==self.revision_id and revision.fingerprint==self.reviewed_fingerprint and
            quality.fingerprint==self.quality_fingerprint and quality.revision_fingerprint==self.base_fingerprint and revision.review==self.decision)


def review_timing(revision:TimingRevision,quality:TimingQuality,*,expected_fingerprint,decision,actor,reason,timestamp):
    if revision.fingerprint!=expected_fingerprint or quality.revision_fingerprint!=revision.fingerprint:
        raise EditConflict('Review refers to a different revision/quality report')
    if decision=='accepted' and any(i.status=='fail' for i in quality.items):raise ValueError('Failed quality gate must be resolved before acceptance')
    after=replace(revision,review=decision)
    record=ReviewDecision(revision.revision_id,revision.fingerprint,after.fingerprint,quality.fingerprint,decision,actor,reason,timestamp)
    return after,record
