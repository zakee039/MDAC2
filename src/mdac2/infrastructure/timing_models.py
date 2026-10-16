"""Explicit local environment manifest, imported without torch in the API."""
from pathlib import Path
import hashlib
import json
from mdac2.infrastructure.model_registry import ModelRegistry,ModelSpec


class TimingModels:
    def __init__(self,root):
        self.root=Path(root).absolute()
        self.manifest=self.root/'tools/experiments/beat_this/models.json'

    def configuration(self,model_id,*,check_dependencies=False):
        if model_id not in ('final0','small0'):raise ValueError('Unsupported timing model')
        if not self.manifest.is_file():raise ValueError('MODEL_CONFIGURATION_MISSING')
        config=json.loads(self.manifest.read_text(encoding='utf-8'))
        if config['schema_version']!=1:raise ValueError('Unknown model configuration')
        if config['python']!='tools/.deps/beat-this-py313/Scripts/python.exe' or config['weights_root']!='tools/private/beat-this':
            raise ValueError('Unsupported local runtime layout')
        record=next(r for r in config['models'] if r['model_id']==model_id)
        python=self.root/config['python'];weights=self.root/config['weights_root']
        if not python.is_file():raise ValueError('MODEL_ENVIRONMENT_MISSING')
        deps=tuple(sorted(config['dependencies'].items()))
        spec=ModelSpec(model_id,record['relative_path'],record['sha256'],record['source'],record['license_id'],
            tuple(record['allowed_uses']),deps if check_dependencies else ())
        registry=ModelRegistry(weights,[spec]);registry.require(model_id)
        backend=self.root/'src/mdac2/adapters/beat_this_local.py'
        producer={'name':'mdac2.timing-analysis','version':'1','model_hash':spec.sha256,
            'dependencies':{**dict(deps),'local_backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest()}}
        return python,registry,producer

    def capabilities(self):
        results=[]
        for model_id in ('final0','small0'):
            try:
                self.configuration(model_id)
                code='CONFIGURED_DEPENDENCIES_CHECKED_AT_RUN';available=True
            except (ValueError,OSError,KeyError,StopIteration):code='LOCAL_MODEL_UNAVAILABLE';available=False
            results.append({'model_id':model_id,'available':available,'code':code,'device':'cpu','precision':'float32'})
        return results
