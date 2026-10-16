"""Audio-separator wrapper identity is distinct from its actual model engine."""
from mdac2.adapters.separation.demucs import LocalSeparationBoundary,SeparationOptions
from mdac2.domain.stems import OutputRole,OutputProfile


VOCAL_PAIR=OutputProfile('separator-native-vocal-pair-pcm16-v1',(
    OutputRole('vocals',('vocals',)),OutputRole('accompaniment',('accompaniment',))))
DERIVED_VOCAL_PAIR=OutputProfile('separator-vocals-derived-accompaniment-pcm16-v1',VOCAL_PAIR.roles)


class AudioSeparatorProvider(LocalSeparationBoundary):
    def __init__(self,registry,model_id,backend,*,native_labels,options=SeparationOptions()):
        # Callers configure the documented checkpoint-specific labels. Filename
        # guesses such as "other" -> accompaniment are deliberately not aliases.
        if type(native_labels) is not tuple or len(native_labels)!=2:raise ValueError('Explicit vocal/accompaniment native labels required')
        if native_labels[1].lower()=='other':raise ValueError('Other is not proven complete accompaniment')
        profile=DERIVED_VOCAL_PAIR if getattr(backend,'policy',{}).get('derivations') else VOCAL_PAIR
        super().__init__(registry,model_id,backend,profile=profile,native_roles=native_labels,options=options)
