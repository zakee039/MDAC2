"""Read-only local checkpoint registry. No download or implicit fallback."""
from dataclasses import dataclass
import hashlib
from importlib import metadata
from pathlib import Path
import re
from mdac2.infrastructure.artifact_store import _plain


@dataclass(frozen=True,slots=True)
class ModelSpec:
    model_id:str
    relative_path:str
    sha256:str
    source:str
    license_id:str
    allowed_uses:tuple[str,...]
    dependencies:tuple[tuple[str,str],...]=()

    def __post_init__(self):
        if not all(type(v) is str and v.strip() for v in (self.model_id,self.relative_path,self.source,self.license_id)):
            raise ValueError('Model identity/source/license required')
        path=Path(self.relative_path)
        if path.is_absolute() or path.drive or '..' in path.parts or ':' in self.relative_path:raise ValueError('Relative checkpoint path required')
        if not re.fullmatch('[0-9a-f]{64}',self.sha256):raise ValueError('Checkpoint SHA256 required')
        if type(self.allowed_uses) is not tuple or not self.allowed_uses or any(type(v) is not str or not v for v in self.allowed_uses):raise ValueError('Explicit allowed uses required')
        if type(self.dependencies) is not tuple or any(type(d) is not tuple or len(d)!=2 or any(type(v) is not str or not v for v in d) for d in self.dependencies):raise ValueError('Pinned dependencies required')
        if len({d[0] for d in self.dependencies})!=len(self.dependencies):raise ValueError('Duplicate dependencies')


@dataclass(frozen=True,slots=True)
class ModelCapability:
    available:bool
    code:str
    model_hash:str
    dependencies:tuple[tuple[str,str],...]=()


class ModelRegistry:
    def __init__(self,root,specs):
        self.root=Path(root).absolute()
        self.specs={s.model_id:s for s in specs}
        if len(self.specs)!=len(specs):raise ValueError('Duplicate model ID')

    def check(self,model_id,*,use='inference'):
        spec=self.specs[model_id]
        def result(code,deps=()):return ModelCapability(code=='AVAILABLE',code,spec.sha256,tuple(deps))
        if use not in spec.allowed_uses:return result('USE_NOT_ALLOWED')
        path=self.root/spec.relative_path
        try:
            for parent in reversed((path,*path.parents)):
                if parent.exists():_plain(parent)
            if not path.is_file():return result('WEIGHTS_MISSING')
            if not path.stat().st_size:return result('WEIGHTS_INVALID')
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=spec.sha256:return result('WEIGHTS_HASH_MISMATCH')
        except (OSError,ValueError):return result('WEIGHTS_UNREADABLE')
        versions=[]
        for name,required in spec.dependencies:
            try:version=metadata.version(name)
            except metadata.PackageNotFoundError:return result('DEPENDENCY_MISSING',versions)
            versions.append((name,version))
            if version!=required:return result('DEPENDENCY_VERSION_MISMATCH',versions)
        return result('AVAILABLE',versions)

    def require(self,model_id,*,use='inference'):
        capability=self.check(model_id,use=use)
        if not capability.available:raise ValueError(capability.code)
        return self.root/self.specs[model_id].relative_path,capability
