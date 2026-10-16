"""Fitting quality is independent from a user's review and export permission."""
from dataclasses import dataclass,asdict
from mdac2.domain.artifact import content_hash
from mdac2.domain.review import TimingQuality,QualityItem


@dataclass(frozen=True,slots=True)
class TimingReviewPolicy:
    version:str='1'
    minimum_coverage:float=.8

    def __post_init__(self):
        if self.version!='1' or type(self.minimum_coverage) not in (float,int) or not 0<self.minimum_coverage<=1:raise ValueError('Invalid timing review policy')

    @property
    def fingerprint(self):return content_hash(asdict(self))


def assess_fit(revision,fit,*,policy=TimingReviewPolicy()):
    if not fit.accepted or fit.mapping is None or revision.mapping!=fit.mapping:raise ValueError('Quality must bind the actual fitted candidate')
    duration=float(revision.mapping.audio_duration)
    coverage=(fit.coverage_seconds[1]-fit.coverage_seconds[0])/duration if duration else 0
    items=(QualityItem('residual_p95','pass',fit.p95_seconds,'seconds'),
        QualityItem('residual_max','pass',fit.max_seconds,'seconds'),
        QualityItem('coverage','pass' if coverage>=policy.minimum_coverage else 'warn',coverage,'ratio',('OBSERVATION_COVERAGE_ONLY',)),
        QualityItem('pulse_meter','unknown',None,'interpretation',('EXPLICIT_CANDIDATE_REQUIRES_LISTENING',)),
        QualityItem('exportability','unknown',None,'profile',('REQUIRES_EXPORT_PROFILE_VALIDATION',)))
    return TimingQuality(revision.fingerprint,fit.observation_hash,
        content_hash({'review_policy':policy.fingerprint,'fit_policy':fit.policy_hash,'interpretation':fit.interpretation_hash}),items)
