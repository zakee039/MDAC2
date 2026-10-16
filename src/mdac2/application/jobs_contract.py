"""Immutable schema-1 job requests and fenced execution attempts; no I/O."""
from dataclasses import asdict, dataclass, replace
import math
from uuid import uuid4

from mdac2.domain.artifact import ArtifactEnvelope, canonical_json, content_hash, strict_json


class JobError(ValueError):
    pass


def _text(value):
    if type(value) is not str or not value.strip():
        raise JobError("Nonempty text required")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise JobError("Finite nonnegative time required")


@dataclass(frozen=True, slots=True)
class Resources:
    cpu_slots: int = 1
    gpu_device: int | None = None
    gpu_memory_mb: int = 0

    def __post_init__(self):
        if type(self.cpu_slots) is not int or self.cpu_slots < 1:
            raise JobError("At least one CPU slot required")
        if self.gpu_device is not None and (type(self.gpu_device) is not int or self.gpu_device < 0):
            raise JobError("Invalid GPU device")
        if type(self.gpu_memory_mb) is not int or self.gpu_memory_mb < 0:
            raise JobError("Invalid GPU memory request")
        if (self.gpu_device is None) != (self.gpu_memory_mb == 0):
            raise JobError("GPU device and positive memory reservation must be paired")


@dataclass(frozen=True, slots=True)
class JobRequest:
    encoded: bytes

    def __post_init__(self):
        if type(self.encoded) is not bytes:
            raise JobError("Request must store immutable bytes")
        value = strict_json(self.encoded)
        fields = {"schema_version", "project_id", "project_revision", "operation", "inputs", "producer", "options", "resources"}
        if type(value) is not dict or set(value) != fields or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise JobError("Unsupported job request schema")
        for key in ("project_id", "operation"):
            _text(value[key])
        if value["project_revision"] is not None:
            _text(value["project_revision"])
        if type(value["resources"]) is not dict or set(value["resources"]) != {"cpu_slots", "gpu_device", "gpu_memory_mb"}:
            raise JobError("Invalid resource fields")
        Resources(**value["resources"])
        # Reuse the producer/input contract without confusing request and result IDs.
        ArtifactEnvelope.create(artifact_type=value["operation"], inputs=value["inputs"],
                                producer=value["producer"], options=value["options"], payload=None)
        object.__setattr__(self, "encoded", canonical_json(value))

    @classmethod
    def create(cls, *, project_id, project_revision, operation, inputs, producer, options, resources=Resources()):
        if not isinstance(resources, Resources):
            raise JobError("Resources required")
        return cls(canonical_json({"schema_version": 1, "project_id": project_id,
            "project_revision": project_revision, "operation": operation, "inputs": inputs,
            "producer": producer, "options": options, "resources": asdict(resources)}))

    def to_dict(self):
        return strict_json(self.encoded)

    @property
    def request_key(self):
        return content_hash(self.to_dict())


TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


def public_run(run):
    """No lease, worker identity, traceback, or arbitrary diagnostic text."""
    stages = {"queued", "starting", "process_owned", "computing", *TERMINAL}
    errors = {"TIMEOUT", "PROCESS_EXIT", "CHILD_PROCESS_LEFT_RUNNING"}
    return {"job_id": run.job_id, "run_id": run.run_id, "attempt": run.attempt, "state": run.state,
        "stage": run.stage if run.stage in stages else "processing", "completed_units": run.completed_units,
        "total_units": run.total_units, "cancel_requested": run.cancel_requested,
        "error_code": run.error_code if run.error_code in errors else ("WORKER_ERROR" if run.error_code else None)}


@dataclass(frozen=True, slots=True)
class Run:
    job_id: str
    run_id: str
    attempt: int = 1
    state: str = "queued"
    worker_id: str | None = None
    lease_token: str | None = None
    lease_until: float | None = None
    exit_confirmed: bool = False
    cancel_requested: bool = False
    stage: str = "queued"
    completed_units: int = 0
    total_units: int | None = None
    error_code: str | None = None

    def __post_init__(self):
        for value in (self.job_id, self.run_id, self.stage):
            _text(value)
        if type(self.attempt) is not int or self.attempt < 1 or self.state not in TERMINAL | {"queued", "running"}:
            raise JobError("Invalid attempt/state")
        if type(self.exit_confirmed) is not bool or type(self.cancel_requested) is not bool:
            raise JobError("Boolean flags required")
        if type(self.completed_units) is not int or self.completed_units < 0:
            raise JobError("Invalid progress")
        if self.total_units is not None and (type(self.total_units) is not int or self.total_units < self.completed_units):
            raise JobError("Progress exceeds total")
        lease = (self.worker_id, self.lease_token, self.lease_until)
        if any(value is not None for value in lease):
            _text(self.worker_id)
            _text(self.lease_token)
            _number(self.lease_until)
        if self.state == "running" and (self.lease_token is None or self.exit_confirmed):
            raise JobError("Running attempt needs a live lease")
        if self.state == "queued" and (any(v is not None for v in lease) or self.exit_confirmed or self.cancel_requested):
            raise JobError("Queued attempt cannot hold execution state")
        if self.state in {"succeeded", "failed", "cancelled"} and not self.exit_confirmed:
            raise JobError("Terminal computation must have exited")
        if self.error_code is not None:
            _text(self.error_code)

    def start(self, worker_id, *, now, lease_seconds):
        if self.state != "queued":
            raise JobError("Only queued attempts can start")
        _number(now)
        _number(lease_seconds)
        if lease_seconds == 0:
            raise JobError("Positive lease required")
        return replace(self, state="running", stage="starting", worker_id=worker_id,
                       lease_token=str(uuid4()), lease_until=now + lease_seconds)

    def _fence(self, token, now):
        _number(now)
        if self.state != "running" or token != self.lease_token or now >= self.lease_until:
            raise JobError("Expired or foreign run lease")

    def heartbeat(self, token, *, now, lease_seconds, stage, completed_units=0, total_units=None):
        self._fence(token, now)
        _number(lease_seconds)
        if lease_seconds == 0:
            raise JobError("Positive lease required")
        return replace(self, lease_until=max(self.lease_until, now + lease_seconds), stage=stage,
                       completed_units=completed_units, total_units=total_units)

    def cancel(self):
        if self.state == "queued":
            return replace(self, state="cancelled", stage="cancelled", cancel_requested=True, exit_confirmed=True)
        if self.state != "running":
            raise JobError("Attempt already terminal")
        return replace(self, cancel_requested=True)

    def finish(self, state, token, *, now, exit_confirmed, error_code=None):
        self._fence(token, now)
        if state not in {"succeeded", "failed", "cancelled"} or exit_confirmed is not True:
            raise JobError("Finish requires terminal state and confirmed process exit")
        if self.cancel_requested and state != "cancelled":
            raise JobError("Cancellation must be acknowledged")
        if state == "cancelled" and not self.cancel_requested:
            raise JobError("Cancellation was not requested")
        return replace(self, state=state, stage=state, exit_confirmed=True, error_code=error_code)

    def interrupt(self, *, now, exit_confirmed=False):
        _number(now)
        if self.state != "running" or (now < self.lease_until and exit_confirmed is not True):
            raise JobError("Live process cannot be interrupted before lease expiry")
        return replace(self, state="interrupted", stage="interrupted", exit_confirmed=exit_confirmed)

    def confirm_exit(self):
        if self.state != "interrupted":
            raise JobError("Only interrupted attempts need exit confirmation")
        return replace(self, exit_confirmed=True)

    def retry(self):
        if self.state not in {"failed", "cancelled", "interrupted"} or not self.exit_confirmed:
            raise JobError("Retry requires failed/cancelled/interrupted attempt and confirmed exit")
        return Run(self.job_id, str(uuid4()), attempt=self.attempt + 1)


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    request: JobRequest
    runs: tuple[Run, ...]

    def __post_init__(self):
        _text(self.job_id)
        if not isinstance(self.request, JobRequest) or type(self.runs) is not tuple or not self.runs:
            raise JobError("Request and immutable run history required")
        if any(not isinstance(run, Run) or run.job_id != self.job_id or run.attempt != i + 1 for i, run in enumerate(self.runs)):
            raise JobError("Invalid run lineage")
        if len({run.run_id for run in self.runs}) != len(self.runs):
            raise JobError("Run IDs must be unique")
        if any(run.state not in {"failed", "cancelled", "interrupted"} or not run.exit_confirmed for run in self.runs[:-1]):
            raise JobError("Previous runs must have exited without success")

    @classmethod
    def create(cls, request):
        job_id = str(uuid4())
        return cls(job_id, request, (Run(job_id, str(uuid4())),))

    @property
    def state(self):
        return self.runs[-1].state
