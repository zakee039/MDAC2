from dataclasses import replace
from fractions import Fraction as F

import pytest

from mdac2.application.timing_migration import MigrationError, apply_migration, preview_migration
from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.domain.timing import Anchor, TimingMap
from mdac2.domain.timing_edits import TimingRevision


OLD = TimingRevision("t-old", TimingMap((Anchor(0, 0), Anchor(8, 4)), 4))
NEW = TimingRevision("t-new", TimingMap((Anchor(0, 0), Anchor(16, 4)), 4))


def chart(*, locked=False):
    events = (ChartEvent("hold", 2, ("ground",), "hold", duration_q=2, locked=locked),
              ChartEvent("tap", 2, ("air",), source_kind="candidate", source_id="evidence-1"))
    return ChartRevision("c0", "t-old", events, user_patch_ids=("manual-patch",))


def preview(source, policy="keep_audio_time", new=NEW):
    return preview_migration(source, OLD, new, position_policy=policy)


def apply(source, proposal, policy="keep_audio_time", new=NEW):
    return apply_migration(source, OLD, new, position_policy=policy,
                           expected_preview_fingerprint=proposal.fingerprint)


def test_keep_audio_time_preserves_hold_head_and_tail_seconds():
    source = chart()
    before = source.to_json()
    result = preview(source)
    diff = result.diffs[0]
    assert (diff.old_q, diff.old_end_q, diff.new_q, diff.new_end_q) == (2, 4, 4, 8)
    assert (diff.old_seconds, diff.old_end_seconds) == (diff.new_seconds, diff.new_end_seconds) == (1, 2)
    assert not result.conflicts
    updated = apply(source, result)
    assert updated.revision_id != source.revision_id
    assert updated.timing_revision == NEW.revision_id
    assert updated.events[0].duration_q == 4
    assert updated.events[1].source_id == "evidence-1"
    assert updated.events[0].event_id == "hold"
    assert updated.user_patch_ids == source.user_patch_ids
    assert len(updated.events) == 2
    assert source.to_json() == before


def test_keep_music_preserves_beats_changes_seconds():
    source = chart()
    proposal = preview(source, "keep_musical_position")
    diff = proposal.diffs[0]
    assert (diff.new_q, diff.new_end_q) == (diff.old_q, diff.old_end_q) == (2, 4)
    assert (diff.new_seconds, diff.new_end_seconds) == (F(1, 2), 1)
    assert apply(source, proposal, "keep_musical_position").events == source.events


@pytest.mark.parametrize("policy", ["keep_audio_time", "keep_musical_position"])
def test_locked_changes_conflict_under_both_policies(policy):
    source = chart(locked=True)
    proposal = preview(source, policy)
    assert any(c.code == "LOCKED_EVENT_CHANGED" for c in proposal.conflicts)
    with pytest.raises(MigrationError):
        apply(source, proposal, policy)


def test_identical_mapping_does_not_conflict_with_lock():
    same_mapping = replace(OLD, revision_id="same-content-new-id")
    source = chart(locked=True)
    proposal = preview(source, new=same_mapping)
    assert not proposal.conflicts
    assert apply(source, proposal, new=same_mapping).events == source.events


@pytest.mark.parametrize("policy", ["keep_audio_time", "keep_musical_position"])
def test_out_of_range_not_silently_deleted(policy):
    limited = TimingRevision("limited", TimingMap((Anchor(0, 0), Anchor(2, 1)), 1))
    source = chart()
    proposal = preview(source, policy, limited)
    assert any(c.code == "TIME_OUT_OF_RANGE" for c in proposal.conflicts)
    with pytest.raises(MigrationError):
        apply(source, proposal, policy, limited)
    assert len(source.events) == 2


def test_preview_fingerprint_binds_chart_timing_and_policy():
    source = chart()
    proposal = preview(source)
    changed_chart = replace(source, events=(replace(source.events[0], q=1), source.events[1]))
    with pytest.raises(MigrationError):
        apply(changed_chart, proposal)
    changed_timing = replace(NEW, review="accepted")
    with pytest.raises(MigrationError):
        apply(source, proposal, new=changed_timing)
    with pytest.raises(MigrationError):
        apply(source, proposal, "keep_musical_position")


def test_bad_source_and_identity_fail_closed():
    source = chart()
    with pytest.raises(MigrationError):
        preview(replace(source, timing_revision="other"))
    with pytest.raises(MigrationError):
        preview(source, "guess")
    with pytest.raises(MigrationError):
        preview(source, new=replace(NEW, revision_id=OLD.revision_id))
    bad = replace(source, events=(replace(source.events[0], duration_q=-1),))
    proposal = preview(bad)
    assert any(c.code == "SOURCE_INVALID_DURATION" for c in proposal.conflicts)
    with pytest.raises(MigrationError):
        apply(bad, proposal)


def test_piecewise_mapping_treats_hold_end_independently():
    piecewise = TimingRevision("piecewise", TimingMap((Anchor(0, 0), Anchor(4, 1), Anchor(7, 4)), 4))
    proposal = preview(chart(), new=piecewise)
    diff = proposal.diffs[0]
    assert (diff.new_q, diff.new_end_q) == (4, 5)
    assert proposal.proposed_events[0].duration_q == 1
