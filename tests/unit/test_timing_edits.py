from dataclasses import replace

import pytest

from mdac2.domain.timing import Anchor, TimingError, TimingMap
from mdac2.domain.timing_edits import EditConflict, TimingRevision, edit_timing, undo_timing


META = dict(actor="test-user", reason="manual correction", timestamp="2026-09-15T03:00:00Z")


def base():
    return TimingRevision("t0", TimingMap((Anchor(0, 0), Anchor(4, 2), Anchor(8, 4)), 5),
                          (0, 4), "accepted")


def edit(revision, **kwargs):
    return edit_timing(revision, expected_fingerprint=revision.fingerprint, **META, **kwargs)


def test_insert_drag_delete_and_old_approval_invalidation():
    original = base()
    result = edit(original, operation="insert_beat", q=2, t=1)
    assert len(result.after.mapping.anchors) == 4
    assert original.review == "accepted"
    assert result.after.review == "pending"
    dragged = edit(result.after, operation="drag_beat", q=2, t=1.2)
    assert dragged.after.mapping.t(2) == dragged.after.mapping.anchors[1].t
    deleted = edit(dragged.after, operation="delete_beat", q=2)
    assert deleted.after.mapping == original.mapping
    assert deleted.after.revision_id != original.revision_id
    assert deleted.after.review == "pending"
    assert result.record.base_fingerprint == original.fingerprint
    assert result.record.result_fingerprint == result.after.fingerprint


def test_downbeat_marks_only_meter_boundary():
    marked = edit(base(), operation="set_downbeat", q=8, enabled=True)
    assert marked.after.downbeat_q == (0, 4, 8)
    removed = edit(marked.after, operation="set_downbeat", q=4, enabled=False)
    assert removed.after.downbeat_q == (0, 8)
    inserted = edit(base(), operation="insert_beat", q=2, t=1)
    with pytest.raises(TimingError):
        edit(inserted.after, operation="set_downbeat", q=2, enabled=True)


def test_undo_restores_content_not_approval_or_revision_identity():
    original = base()
    result = edit(original, operation="drag_beat", q=4, t=2.2)
    undone = undo_timing(result.after, result, expected_fingerprint=result.after.fingerprint, **META)
    assert undone.after.mapping == original.mapping
    assert undone.after.downbeat_q == original.downbeat_q
    assert undone.after.review == "pending"
    assert undone.after.revision_id != original.revision_id
    assert undone.record.operation == "undo"


def test_stale_base_and_wrong_undo_refused():
    original = base()
    result = edit(original, operation="drag_beat", q=4, t=2.2)
    with pytest.raises(EditConflict):
        edit_timing(result.after, expected_fingerprint=original.fingerprint,
                    operation="delete_beat", q=8, **META)
    with pytest.raises(EditConflict):
        undo_timing(original, result, expected_fingerprint=original.fingerprint, **META)
    changed = replace(result, record=replace(result.record, base_fingerprint="fake"))
    with pytest.raises(EditConflict):
        undo_timing(result.after, changed, expected_fingerprint=result.after.fingerprint, **META)


@pytest.mark.parametrize("kwargs", [
    dict(operation="insert_beat", q=4, t=2), dict(operation="insert_beat", q=2, t=3),
    dict(operation="drag_beat", q=4, t=0), dict(operation="drag_beat", q=4, t=-1),
    dict(operation="drag_beat", q=8, t=6), dict(operation="drag_beat", q=9, t=4),
    dict(operation="delete_beat", q=4), dict(operation="delete_beat", q=999),
    dict(operation="delete_beat", q=8, t=3), dict(operation="set_downbeat", q=8, enabled=1),
    dict(operation="unsupported", q=4),
])
def test_invalid_operations_leave_base_unchanged(kwargs):
    original = base()
    fingerprint = original.fingerprint
    with pytest.raises(TimingError):
        edit(original, **kwargs)
    assert original.fingerprint == fingerprint


def test_cannot_delete_to_one_anchor():
    revision = TimingRevision("minimal", TimingMap((Anchor(0, 0), Anchor(4, 2)), 2))
    with pytest.raises(TimingError):
        edit(revision, operation="delete_beat", q=4)
