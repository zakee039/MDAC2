"""CPU analysis child; no model imports in the web/server environment."""
import argparse
from fractions import Fraction
from pathlib import Path
import socket
import sys
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.timing_workflow import TimingWorkflow
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.infrastructure.timing_models import TimingModels
from mdac2.ports.timing import TimingInput


def main():
    parser=argparse.ArgumentParser()
    for name in ('database','store','root','job','result'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    backend=JobBackend(args.database);store=ArtifactStore(args.store)
    request=backend.get_job(args.job).request.to_dict();options=request['options']
    if request['operation']!='timing_candidate':raise ValueError('Wrong analysis operation')
    _,registry,producer=TimingModels(args.root).configuration(options['model_id'],check_dependencies=True)
    if producer!=request['producer']:raise ValueError('Model runtime changed since enqueue')
    flow=TimingWorkflow(ManualWorkflow(backend,store))
    audio,inputs=flow.context(request['project_id'],request['project_revision'])
    if inputs!=request['inputs']:raise ValueError('Analysis audio changed')
    with wave.open(str(store._destination('audio',audio.artifact_hash)/'audio.wav'),'rb') as wav:
        source=TimingInput(audio.artifact_hash,wav.getframerate(),wav.getnframes(),wav.getnchannels())
        if wav.getsampwidth()!=2:raise ValueError('PCM16 required')
        pcm=wav.readframes(wav.getnframes())
    def forbidden(*args,**kwargs):raise RuntimeError('Inference networking forbidden')
    from mdac2.adapters.beat_this import BeatThisProvider
    from mdac2.adapters.beat_this_local import LocalBeatThisBackend
    socket.socket=forbidden
    observation=BeatThisProvider(registry,options['model_id'],LocalBeatThisBackend()).observe(source,pcm,include_logits=True)
    candidate=flow.propose(request['project_id'],request['project_revision'],observation,
        pulse_unit=Fraction(options['pulse_unit']),first_q=Fraction(options['first_q']),meter=tuple(options['meter']),
        producer=producer,options=options)
    Path(args.result).write_bytes(candidate.to_json())


if __name__=='__main__':main()
