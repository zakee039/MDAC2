"""Bounded mix-only onset analysis in the existing cancellable child process."""
import argparse
from pathlib import Path
import socket

from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.onset_workflow import OnsetWorkflow, PRODUCER, request_options
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def main():
    parser = argparse.ArgumentParser()
    for name in ('database', 'store', 'job', 'result'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    backend = JobBackend(args.database); store = ArtifactStore(args.store)
    request = backend.get_job(args.job).request.to_dict()
    if request['operation'] != 'onset_candidate' or request['producer'] != PRODUCER:
        raise ValueError('Wrong onset operation or runtime identity')
    options = request['options']; difficulty = options['selection_policy']['difficulty']
    if options != request_options(request['project_id'], request['project_revision'], difficulty):
        raise ValueError('Onset runtime policy changed since enqueue')
    flow = OnsetWorkflow(ManualWorkflow(backend, store))
    _, inputs, _, _ = flow.context(request['project_id'], request['project_revision'])
    if inputs != request['inputs']: raise ValueError('Onset inputs changed')
    def forbidden(*args, **kwargs): raise RuntimeError('Analysis networking forbidden')
    socket.socket = forbidden
    artifact = flow.propose(request['project_id'], request['project_revision'], difficulty=difficulty)
    Path(args.result).write_bytes(artifact.to_json())


if __name__ == '__main__': main()
