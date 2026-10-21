"""Final V4 gate using the full tempo map and hold/head/tail invariants."""
from mdac2.application.studio_timing import settings
from mdac2.application.studio_timing_v4 import violations


def audit_playability_v4(events, duration, *, timing=None):
    config=settings(timing)
    issues=violations(events,duration,config)
    return dict(schema_version=2,policy_id='generated-v4-spacing',
                logical_events=len(events),duration=duration,
                occupancy_issues=[i for i in issues if i['code'] not in ('same_lane_gap','hold_tail_guard')],
                exact_attack_stacks=[],
                spacing_issues=[i for i in issues if i['code']=='same_lane_gap'],
                tail_guard_issues=[i for i in issues if i['code']=='hold_tail_guard'],
                minimum_gap_seconds=None if config['enabled'] else .15,
                interior_gap_seconds=None if config['enabled'] else .5,
                timing=config,
                spacing_units='local_beats' if config['enabled'] else 'seconds',
                ordinary_gap_beats=1/3 if config['enabled'] else None,
                interior_gap_beats=1 if config['enabled'] else None,
                engineering_playtest_gate_passed=not issues,
                caveats=['V4 game acceptance is pending.'])
