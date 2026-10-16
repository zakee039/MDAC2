"""Read-only basic IR validation; not a game/export acceptance certificate."""

from dataclasses import dataclass

from .chart import ChartRevision
from .timing import TimingError, TimingMap


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    severity: str
    event_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[Issue, ...]
    policy_version: str = "basic-ir-v1"

    @property
    def has_errors(self):
        return any(issue.severity == "error" for issue in self.issues)


def validate_chart(chart: ChartRevision, timing: TimingMap, *, timing_revision: str,
                   allow_empty: bool = False) -> ValidationReport:
    if type(chart) is not ChartRevision or type(timing) is not TimingMap:
        raise TypeError("Expected validated ChartRevision and TimingMap contracts")
    if type(allow_empty) is not bool:
        raise TypeError("allow_empty must be explicit bool")
    if type(timing_revision) is not str or not timing_revision.strip():
        raise ValueError("Explicit timing revision identity required")
    issues = [Issue("GAME_PROFILE_PENDING", "warning", None,
                    "Game-specific conflicts, object mappings and export timing remain unverified.")]
    if chart.timing_revision != timing_revision:
        issues.append(Issue("TIMING_REVISION_MISMATCH", "error", None,
                            "Do not validate musical coordinates against a different timing revision."))
    if not chart.events:
        issues.append(Issue("EMPTY_CHART_ALLOWED" if allow_empty else "EMPTY_CHART",
                            "warning" if allow_empty else "error", None,
                            "Chart has no events; no automatic repair was applied."))
    for event in chart.events:
        lanes = event.lane_mask
        valid_lanes = bool(lanes) and len(set(lanes)) == len(lanes) and set(lanes) <= {"ground", "air"}
        if not valid_lanes:
            issues.append(Issue("INVALID_LANE_MASK", "error", event.event_id,
                                "Use a nonempty, duplicate-free subset of ground/air."))
        if event.semantic == "gemini" and set(lanes) != {"ground", "air"}:
            issues.append(Issue("GEMINI_INCOMPLETE", "error", event.event_id,
                                "Gemini requires both physical lanes."))
        if ((event.semantic == "hold" and event.duration_q <= 0) or
                (event.semantic != "hold" and event.duration_q != 0)):
            issues.append(Issue("INVALID_DURATION", "error", event.event_id,
                                "Hold must have positive duration; non-Hold duration must be zero."))
        if chart.timing_revision == timing_revision:
            for position, label in ((event.q, "head"), (event.end_q, "tail")):
                try:
                    timing.t(position)
                except TimingError:
                    issues.append(Issue("TIME_OUT_OF_RANGE", "error", event.event_id,
                                        f"Event {label} is outside audio/mapping policy bounds."))
                    break
    return ValidationReport(tuple(issues))
