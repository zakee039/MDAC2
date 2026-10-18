from dataclasses import replace
import numpy as np
import pytest

from mdac2.data_preparation.vocal_priority import VocalPolicy, VocalLatch, evidence, select_onsets


def test_release_exactly_two_seconds_and_chunk_state():
    active = np.zeros(500, bool)
    active[5:10] = True  # voice ends at 0.10; fallback begins at 2.10
    whole = VocalLatch().feed(active)
    assert not whole[:5].any()
    assert whole[5:210].all() and not whole[210:].any()
    latch = VocalLatch()
    chunks = np.concatenate([latch.feed(active[:101]), latch.feed(active[101:200]), latch.feed(active[200:])])
    np.testing.assert_array_equal(whole, chunks)


def test_short_pauses_do_not_release_and_return_is_immediate():
    active = np.zeros(700, bool)
    active[0:10] = True
    active[209] = True  # 1.99 s gap, reset timer
    active[510] = True
    mode = VocalLatch().feed(active)
    assert mode[:410].all() and not mode[410:510].any()
    assert mode[510]


def synthetic(n=600, voice=False):
    f = np.zeros((3, 96, n), np.float32)
    f[0] = f[2] = 9.
    if voice:
        f[1, 20:70, 100:300] = 8.
    p = np.zeros(n, np.float32)
    p[np.arange(20, n, 20)] = .7
    return f, dict(probability=p, offsets=np.full(n, .5, np.float32))


def test_no_voice_preserves_baseline_times_even_with_loud_instruments():
    f, p = synthetic()
    times, a, report = select_onsets(f, p, 6.)
    assert times == [round(i*.01+.005, 3) for i in np.arange(20, 600, 20)]
    assert report['removed'] == report['added'] == 0
    assert not a['vocal_mode'].any()


def test_sustained_vowel_is_not_a_metronome_and_density_respects_voice():
    f, p = synthetic(voice=True)
    times, a, report = select_onsets(f, p, 6.)
    assert a['vocal_mode'][150:450].all()
    assert not any(1.2 < t < 4.8 for t in times)
    assert not a['density_probability'][120:480].any()
    assert report['voice_selected'] <= 2
    assert not a['vocal_mode'][510:].any()


def test_quiet_separator_residual_does_not_trigger_voice():
    f, p = synthetic()
    f[1] = .0001
    assert not evidence(f)['active'].any()


def test_real_vocal_rise_can_promote_low_neural_peak():
    f, p = synthetic(voice=True)
    p['probability'][:] = 0.
    p['probability'][101] = .18
    times, a, report = select_onsets(f, p, 6.)
    assert times == [1.015]
    assert report['added'] == 1


def test_silence_produces_no_onsets_and_nonfinite_input_is_rejected():
    f, p = synthetic()
    f[:] = 0
    p['probability'][:] = 0
    assert select_onsets(f, p, 6.)[0] == []
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        select_onsets(f, p, 6.)


def test_routing_is_deterministic_and_rejects_mismatched_time_axis():
    f, p = synthetic(voice=True)
    first = select_onsets(f, p, 6.)
    second = select_onsets(f, p, 6.)
    assert first[0] == second[0] and first[2] == second[2]
    with pytest.raises(ValueError):
        select_onsets(f, p, 5.)
