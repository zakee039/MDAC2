"""Song/family split and reference qualification, bound to immutable identities."""
from dataclasses import asdict,dataclass
import math
from mdac2.domain.artifact import ArtifactEnvelope,_hash


def text(value):
    if type(value) is not str or not value.strip():raise ValueError('Explicit dataset provenance required')


@dataclass(frozen=True,slots=True)
class Reference:
    label_hash:str
    version:str
    kind:str
    origin:str
    annotation_hash:str
    annotator:str
    review:str
    reviewer:str|None
    lineage:tuple[str,...]=()
    uncertainty_ms:float=0.

    def __post_init__(self):
        _hash(self.label_hash);_hash(self.annotation_hash);text(self.version);text(self.annotator)
        if self.kind not in ('timing','onsets','chart','structure') or self.origin not in ('human_annotation','automatic','automatic_confirmed') or self.review not in ('pending','reviewed','disputed'):raise ValueError('Explicit reference type/origin/review required')
        if self.reviewer is not None:text(self.reviewer)
        if self.review=='reviewed' and self.reviewer is None:raise ValueError('Reviewed reference requires reviewer')
        if type(self.lineage) is not tuple:raise ValueError('Immutable label lineage required')
        for h in self.lineage:_hash(h)
        if type(self.uncertainty_ms) not in (int,float) or not math.isfinite(self.uncertainty_ms) or not 0<=self.uncertainty_ms<=1000:raise ValueError('Bounded reference uncertainty required')

    @property
    def gold_eligible(self):return self.origin=='human_annotation' and self.review=='reviewed' and self.reviewer!=self.annotator

    def to_dict(self):return {**asdict(self),'lineage':list(self.lineage)}


@dataclass(frozen=True,slots=True)
class DatasetSong:
    song_id:str
    audio_hash:str
    pcm_hash:str
    work_family:str
    role:str
    license_basis:str
    usage_scope:tuple[str,...]
    difficulty:str
    slices:tuple[str,...]
    pretrained_overlap:str
    references:tuple[Reference,...]

    def __post_init__(self):
        for value in (self.song_id,self.work_family,self.license_basis,self.difficulty):text(value)
        _hash(self.audio_hash);_hash(self.pcm_hash)
        if self.role not in ('train','validation','locked_test') or self.pretrained_overlap not in ('known_overlap','known_disjoint','unknown'):raise ValueError('Explicit split/overlap status required')
        for values in (self.usage_scope,self.slices):
            if type(values) is not tuple or not values or len(set(values))!=len(values):raise ValueError('Nonempty unique usage/slices required')
            for value in values:text(value)
        required='training' if self.role=='train' else 'evaluation'
        if required not in self.usage_scope:raise ValueError('Dataset role outside declared usage scope')
        if type(self.references) is not tuple or not self.references or any(type(r) is not Reference for r in self.references):raise ValueError('Versioned immutable references required')
        if len({r.label_hash for r in self.references})!=len(self.references):raise ValueError('Duplicate reference identity')

    def to_dict(self):return {**asdict(self),'usage_scope':list(self.usage_scope),'slices':list(self.slices),'references':[r.to_dict() for r in self.references]}


def validate_splits(songs):
    if type(songs) is not tuple or not songs or len(songs)>100000 or any(type(s) is not DatasetSong for s in songs):raise ValueError('Bounded immutable song manifest required')
    for field in ('song_id','audio_hash','pcm_hash'):
        if len({getattr(s,field) for s in songs})!=len(songs):raise ValueError('Duplicate song/normalized audio; group references under one song')
    family_roles={};label_roles={}
    for song in songs:
        prior=family_roles.setdefault(song.work_family,song.role)
        if prior!=song.role:raise ValueError('Work family leakage across dataset roles')
        for r in song.references:
            for h in (r.label_hash,r.annotation_hash,*r.lineage):
                prior=label_roles.setdefault(h,song.role)
                if prior!=song.role:raise ValueError('Reference lineage leakage across roles')


@dataclass(frozen=True,slots=True)
class DatasetManifest:
    songs:tuple[DatasetSong,...]
    artifact:ArtifactEnvelope

    @property
    def fingerprint(self):return self.artifact.artifact_hash

    @classmethod
    def freeze(cls,songs,*,version,family_policy,split_seed):
        validate_splits(songs);text(version);text(family_policy)
        if type(split_seed) is not int or split_seed<0:raise ValueError('Explicit nonnegative split seed required')
        payload={'schema_version':1,'version':version,'frozen':True,'family_policy':family_policy,'split_seed':split_seed,'songs':[s.to_dict() for s in songs]}
        inputs=sorted({h for s in songs for h in (s.audio_hash,*[r.label_hash for r in s.references])})
        artifact=ArtifactEnvelope.create(artifact_type='dataset_manifest',producer={'name':'mdac2.dataset-manifest','version':'1','model_hash':None,'dependencies':{}},inputs=inputs,options={'version':version,'family_policy':family_policy,'split_seed':split_seed},payload=payload)
        return cls(songs,artifact)

    @classmethod
    def load(cls,data,*,expected_hash):
        _hash(expected_hash);artifact=ArtifactEnvelope.from_json(data)
        if artifact.artifact_hash!=expected_hash or artifact.to_dict()['artifact_type']!='dataset_manifest':raise ValueError('Frozen dataset identity mismatch')
        p=artifact.payload
        if p.get('schema_version')!=1 or p.get('frozen') is not True:raise ValueError('Frozen manifest schema required')
        songs=[]
        for s in p['songs']:
            refs=tuple(Reference(**{**r,'lineage':tuple(r['lineage'])}) for r in s['references'])
            songs.append(DatasetSong(**{**s,'usage_scope':tuple(s['usage_scope']),'slices':tuple(s['slices']),'references':refs}))
        result=cls.freeze(tuple(songs),version=p['version'],family_policy=p['family_policy'],split_seed=p['split_seed'])
        if result.fingerprint!=artifact.artifact_hash:raise ValueError('Noncanonical/fabricated manifest metadata')
        return result
