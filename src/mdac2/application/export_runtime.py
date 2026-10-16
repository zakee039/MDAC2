"""Huey transport dispatch and supervised serial local export processing."""
from pathlib import Path
import logging
import sys
import threading
import time
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.infrastructure.job_backend import HueyTransport
from mdac2.infrastructure.process_runner import ProcessRunner
from mdac2.application.reconciliation import Reconciler


class ExportRuntime:
    def __init__(self,backend,store,decoder,queue_path,*,timing_models=None):
        self.backend,self.store,self.decoder=backend,store,decoder
        self.runner=ProcessRunner(backend,lease_seconds=30)
        self.reconciler=Reconciler(backend,store)
        self.transport=HueyTransport(queue_path,self.execute)
        self.stop_event=threading.Event();self.thread=None
        self.timing_models=timing_models

    def execute(self,job_id,run_id):
        run=self.backend.claim(job_id,run_id,'local-export',now=time.time(),lease_seconds=30)
        if run is None:return
        stage=self.store.staging/run.run_id;stage.mkdir(exist_ok=True)
        result_path=stage/'export-result.json'
        argv=[sys.executable,'-m','mdac2.application.export_worker','--database',str(self.backend.database.absolute()),
            '--store',str(self.store.root),'--ffmpeg',str(self.decoder.ffmpeg),'--ffprobe',str(self.decoder.ffprobe),
            '--job',job_id,'--result',str(result_path)]
        request=self.backend.get_job(job_id).request.to_dict()
        if request['operation']=='onset_candidate':
            argv=[sys.executable,'-m','mdac2.application.onset_worker','--database',str(self.backend.database.absolute()),
                '--store',str(self.store.root),'--job',job_id,'--result',str(result_path)]
        if request['operation']=='structure_candidate':
            argv=[sys.executable,'-m','mdac2.application.structure_worker','--database',str(self.backend.database.absolute()),
                '--store',str(self.store.root),'--job',job_id,'--result',str(result_path)]
        if request['operation']=='timing_candidate' and self.timing_models is not None:
            # Manifest path is local configuration, never supplied by an API request.
            argv=[str(self.timing_models.root/'tools/.deps/beat-this-py313/Scripts/python.exe'),str(Path(__file__).with_name('timing_worker.py')),
                '--database',str(self.backend.database.absolute()),'--store',str(self.store.root),
                '--root',str(self.timing_models.root),'--job',job_id,'--result',str(result_path)]
        def prepare():
            envelope=ArtifactEnvelope.from_json(result_path.read_bytes())
            # Register inputs in dependency order, without treating any file as job success.
            self.reconciler.recover_artifacts()
            if request['operation'] in ('timing_candidate','onset_candidate','structure_candidate'):return self.reconciler.prepare(envelope)
            data=(self.store._destination('export',envelope.artifact_hash)/'chart.mdm').read_bytes()
            return self.reconciler.prepare(envelope,{'chart.mdm':data})
        self.runner.execute(job_id,run_id,argv,timeout=180,prepare_success=prepare)

    def tick(self):
        with self.backend._connection() as c:
            ids=[r[0] for r in c.execute('SELECT job_id FROM jobs WHERE active=1')]
        for job_id in ids:
            run=self.backend.get_job(job_id).runs[-1]
            if run.state=='running' and run.lease_until<=time.time():self.runner.recover(job_id,run.run_id)
            elif run.state=='interrupted' and not run.exit_confirmed:self.runner.recover(job_id,run.run_id)
        self.backend.dispatch(self.transport)
        task=self.transport.huey.dequeue()
        if task is not None:self.transport.huey.execute(task)

    def start(self):
        self.reconciler.recover_artifacts();self.reconciler.recover_queued_delivery()
        def loop():
            while not self.stop_event.is_set():
                try:self.tick()
                except Exception:logging.getLogger(__name__).error('Export dispatch interrupted; persistent run retained')
                self.stop_event.wait(.1)
        self.thread=threading.Thread(target=loop,name='mdac2-export-dispatch',daemon=True);self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.backend._connection() as c:
            ids=[r[0] for r in c.execute('SELECT job_id FROM jobs WHERE active=1')]
        for job_id in ids:
            run=self.backend.get_job(job_id).runs[-1]
            if run.state=='running':self.backend.cancel(job_id,run.run_id)
        if self.thread:self.thread.join(10)
        self.transport.close()
