"""Explainable selection constraints, separate from acoustic evidence."""
from dataclasses import asdict,dataclass
import math
from mdac2.domain.artifact import content_hash,_hash


@dataclass(frozen=True,slots=True)
class OnsetPolicy:
    difficulty:str='normal'
    minimum_score:float=.2
    minimum_gap_seconds:float=.12
    max_notes_per_second:int=5
    allowed_denominators:tuple[int,...]=(1,2,4)
    version:str='selection-1'

    def __post_init__(self):
        if self.version!='selection-1' or self.difficulty not in ('easy','normal','hard','custom'):raise ValueError('Explicit difficulty/policy version required')
        if type(self.minimum_score) not in (int,float) or not math.isfinite(self.minimum_score) or not 0<=self.minimum_score<=1:raise ValueError('Bounded score required')
        if type(self.minimum_gap_seconds) not in (int,float) or not math.isfinite(self.minimum_gap_seconds) or not 0<self.minimum_gap_seconds<=5:raise ValueError('Positive minimum gap required')
        if type(self.max_notes_per_second) is not int or not 1<=self.max_notes_per_second<=32:raise ValueError('Bounded local density required')
        if type(self.allowed_denominators) is not tuple or not self.allowed_denominators or len(set(self.allowed_denominators))!=len(self.allowed_denominators) or any(type(d) is not int or not 1<=d<=192 for d in self.allowed_denominators):raise ValueError('Explicit grid complexity required')

    @classmethod
    def preset(cls,difficulty):
        values={'easy':(.3,.2,3,(1,2)),'normal':(.2,.12,5,(1,2,4)),'hard':(.12,.07,8,(1,2,3,4,6,8))}
        if difficulty not in values:raise ValueError('Unknown difficulty')
        return cls(difficulty,*values[difficulty])

    @property
    def fingerprint(self):return content_hash({**asdict(self),'allowed_denominators':list(self.allowed_denominators)})


@dataclass(frozen=True,slots=True)
class SelectionDecision:
    candidate_id:str
    selected:bool
    reasons:tuple[str,...]

    def __post_init__(self):
        _hash(self.candidate_id)
        if type(self.selected) is not bool or type(self.reasons) is not tuple or not self.reasons or any(type(r) is not str or not r for r in self.reasons):raise ValueError('Explicit immutable selection explanation required')


@dataclass(frozen=True,slots=True)
class OnsetSelection:
    onset_fingerprint:str
    policy_hash:str
    decisions:tuple[SelectionDecision,...]

    def __post_init__(self):
        _hash(self.onset_fingerprint);_hash(self.policy_hash)
        if type(self.decisions) is not tuple or any(type(d) is not SelectionDecision for d in self.decisions):raise ValueError('Immutable decisions required')
        if len({d.candidate_id for d in self.decisions})!=len(self.decisions):raise ValueError('Duplicate decision')

    @property
    def selected_ids(self):return tuple(d.candidate_id for d in self.decisions if d.selected)
    @property
    def fingerprint(self):return content_hash({'onsets':self.onset_fingerprint,'policy':self.policy_hash,'decisions':[{**asdict(d),'reasons':list(d.reasons)} for d in self.decisions]})
